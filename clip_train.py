import os
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer
from transformers import get_constant_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import SequentialDistributedSampler
from tqdm.auto import tqdm


def distributed_concat(tensor, num_total_examples):
    output_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


class ImageDataset(Dataset):
    def __init__(self, transform_img=None, transform_prt=None):
        path = 'metadata_withcmt.parquet'
        path_img = '/remote-home/share/bxy/SD_2M/bxy_part/images'
        dataframe = pd.read_parquet(path)
        image_names = list(dataframe[dataframe['remove_idx_80']]['image_name'])
        self.image_paths = [os.path.join(path_img, i) for i in image_names]
        self.prompt_list = list(dataframe[dataframe['remove_idx_80']]['prompt'])

        self.transform_img = transform_img
        self.transform_prt = transform_prt

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        prompt = self.prompt_list[idx]
        if self.transform_img:
            image = self.transform_img(images=image, return_tensors="pt", padding=True)
            image = image['pixel_values'].squeeze()
        if self.transform_prt is not None:
            prompt = self.transform_prt.encode(prompt, convert_to_tensor=True)
        return image.to('cpu'), prompt.to('cpu')


def cosine_similarity_loss(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    output = -cos(pred, target).mean()
    return output


def load_pretrained_model():
    model = Net()

    trainable_model_weights = False
    for name, child in model.named_children():
        if name == 'vision':
            for pn, p in child.named_parameters():
                if str(UNFREEZE_START) in pn:
                    """start unfreezing layer , the weights are trainable"""
                    trainable_model_weights = True
                p.requires_grad = trainable_model_weights
                if p.requires_grad:
                    print(f"{pn} is set to be trainable.")

    return model.to(device)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision = clip.vision_model
        self.fc = nn.Linear(1024, 384)

    def forward(self, x):
        out = self.vision(x)['pooler_output']
        return self.fc(out)


"""main training"""
NEPOCH = 25
UNFREEZE_START = 18  # set it to lower number when significantly more samples are included.
BestEpoch = 0
BestSim = 0
writer = SummaryWriter(log_dir='checkpoints_clip')

torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if local_rank == 0:
    print('Preparing models...')

st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = load_pretrained_model()
if local_rank == 0:
    print('Preparing data...')
dataset_all = ImageDataset(transform_img=clip_processor, transform_prt=st_model)
train_size = int(0.8 * len(dataset_all))
test_size = len(dataset_all) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset_all, [train_size, test_size])
train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=32)
test_sampler = SequentialDistributedSampler(test_dataset, batch_size=64)
testloader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler, num_workers=32)

if local_rank == 0:
    print(f"test size: {len(train_dataset)}, train size: {len(test_dataset)}")

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
optimizer.zero_grad()
num_training_steps = NEPOCH * len(train_loader)
lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.01 * num_training_steps))
progress_bar = tqdm(range(num_training_steps), disable=not local_rank == 0)
step = 0
for epoch in range(NEPOCH):
    epoch_loss = 0
    train_sampler.set_epoch(epoch)
    for batch_data in train_loader:
        batch_images, batch_targets = batch_data
        batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
        pred = model(batch_images)
        cosine_loss = cosine_similarity_loss(pred, batch_targets)
        loss = cosine_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += -cosine_loss.item()
        lr_scheduler.step()
        step = step + 1
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)
        writer.add_scalar('train_loss_freeze80', loss, step)
    epoch_loss /= len(train_loader)

    """test loss"""
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        losses = []
        for batch_images, batch_targets in tqdm(testloader, disable=not local_rank == 0):
            batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
            pred = model(batch_images)
            loss = -cosine_similarity_loss(pred, batch_targets)
            losses.append(loss)
        loss_eval = distributed_concat(torch.stack(losses),
                                       len(test_sampler.dataset))
        epoch_loss = torch.mean(loss_eval)
        writer.add_scalar('eval_loss_freeze80', epoch_loss, step)
    if local_rank == 0:
        print(f"epoch: {epoch}, test loss: {epoch_loss}")
    model.train()
    if local_rank == 0:
        if epoch_loss > BestSim:
            BestSim = epoch_loss
            BestEpoch = epoch + 1
            print(f"save best model at {BestSim} with epoch {BestEpoch}")
            torch.save(model.state_dict(), f"best_model_freeze80.pt")
    torch.distributed.barrier()
    if local_rank == 0:
        if epoch - 3 > BestEpoch:
            print(f"early stop at {epoch + 1} with best epoch {BestEpoch} and test similarity {BestSim}.")
            break
