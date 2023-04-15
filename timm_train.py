import pandas as pd
import timm
import json
import os
from pathlib import Path
from torch import nn
from torch.distributed import destroy_process_group
from transformers.trainer_pt_utils import SequentialDistributedSampler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
from transformers import WEIGHTS_NAME, CONFIG_NAME
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from collections import namedtuple
from transformers import get_constant_schedule_with_warmup

Sample = namedtuple("Sample", ["prompt", "image_path"])
writer = SummaryWriter(log_dir='checkpoints_vit')
output_dir = '/remote-home/bxy/kaggle/Image2Text/checkpoints_vit'


def distributed_concat(tensor, num_total_examples):
    output_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


class ImageDataset_csv(Dataset):
    def __init__(self, csv_path, img_trans=None, prompt_trans=None):
        self.csv_path = csv_path
        self.img_trans = img_trans
        self.prompt_trans = prompt_trans
        self.data_pair = self._get_items()

    def __getitem__(self, index):
        chosen = self.data_pair[index]
        prompt = chosen.prompt
        image_path = chosen.image_path
        raw_image = Image.open(image_path).convert('RGB')
        if self.img_trans is not None:
            raw_image = self.img_trans(raw_image)
        if self.prompt_trans is not None:
            prompt = self.prompt_trans.encode(prompt, convert_to_tensor=True)
        return raw_image, prompt

    def _get_items(self):
        samples = []
        ImgTxt = pd.read_csv(self.csv_path)
        for i in range(len(ImgTxt)):
            sample = Sample(prompt=ImgTxt['prompt'][i], image_path=ImgTxt['image_path'][i])
            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.data_pair)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.vision = model
        self.fc = nn.Linear(1024, 384)

    def forward(self, x):
        out = self.vision(x)
        return self.fc(out)


def cosine_similarity_loss(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    output = cos(pred, target).mean()
    return output


torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if local_rank == 0:
    print('Preparing models...')
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=384).to(device)

if local_rank == 0:
    print('Preparing data...')
mydataset_all = ImageDataset_csv('data58_combined_prompts.csv', transform, st_model)
length = len(mydataset_all)
test_size = int(0.1 * length)
mydataset_train, mydataset_test = random_split(mydataset_all, lengths=[0.9, 0.1])

# 分布式采样器
train_sampler = DistributedSampler(mydataset_train, shuffle=True)
train_loader = DataLoader(mydataset_train, batch_size=6, sampler=train_sampler, num_workers=0)
test_sampler = SequentialDistributedSampler(mydataset_test, batch_size=64)
testloader = DataLoader(mydataset_test, batch_size=64, sampler=test_sampler, num_workers=0)
optimizer = AdamW(model.parameters(), lr=1e-4)
loss_fn = cosine_similarity_loss
num_epochs = 30
min_loss = 10000
num_training_steps = num_epochs * len(train_loader)
if local_rank == 0:
    print("Total Steps:", num_training_steps)
progress_bar = tqdm(range(num_training_steps), disable=not local_rank == 0)
step = 0
every_evalstep = 500
lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.01 * num_training_steps))

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for image, ids in train_loader:
        image, ids = image.to(device), ids.to(device)
        optimizer.zero_grad()
        out = model(image)
        loss = loss_fn(out, ids)
        loss.backward()
        optimizer.step()
        writer.add_scalar('train_loss', loss, step)
        step = step + 1
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)
        if step % every_evalstep == 0:
            if local_rank == 0:
                print(f'\nStart {step // every_evalstep} eval')
            model.eval()
            with torch.no_grad():
                # 1. 得到本进程的prediction
                losses = []
                for image, ids in testloader:
                    image, ids = image.to(device), ids.to(device)
                    out = model(image)
                    e_loss = loss_fn(out, ids)
                    losses.append(e_loss)
                # 进行gather
                loss_eval = distributed_concat(torch.stack(losses),
                                               len(test_sampler.dataset))
                current_loss = torch.mean(loss_eval)
            model.train()
            print(f'Current eval loss is {current_loss} eval')
            writer.add_scalar('eval_loss', current_loss, step)
            if local_rank == 0:
                if current_loss < min_loss:
                    torch.save(model.state_dict(), os.path.join(output_dir, f'model_{step}.pth'))
                    min_loss = current_loss
            torch.distributed.barrier()
        lr_scheduler.step()
torch.save(model.state_dict(), os.path.join(output_dir, f'model_final.pth'))

destroy_process_group()

