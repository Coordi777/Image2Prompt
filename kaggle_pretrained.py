import json
import os
from pathlib import Path

from torch.distributed import destroy_process_group
from transformers.trainer_pt_utils import SequentialDistributedSampler

from dataset import ImageDataset_csv
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
from transformers import WEIGHTS_NAME, CONFIG_NAME
from torch.utils.tensorboard import SummaryWriter


def distributed_concat(tensor, num_total_examples):
    output_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


writer = SummaryWriter(log_dir='logging')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
output_dir = '/remote-home/bxy/kaggle/Image2Text/checkpoints'
path_csv = '/remote-home/bxy/data/Image2Text_6G/train.csv'
path_csv_test = '/remote-home/bxy/data/Image2Text_6G/eval.csv'
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-large")


def collate_fn(data):
    batch = list(zip(*data))
    images = processor(batch[0], return_tensors="pt")
    prompt = tokenizer(batch[1], padding=True, truncation=True, return_tensors="pt")
    del prompt['token_type_ids']
    return images, prompt


if local_rank == 0:
    print('Preparing data...')
# 数据集准备
mydataset = ImageDataset_csv(path_csv)
mydataset_test = ImageDataset_csv(path_csv_test)
# 分布式采样器
train_sampler = DistributedSampler(mydataset, shuffle=True)
train_loader = DataLoader(mydataset, batch_size=6, sampler=train_sampler, collate_fn=collate_fn)
test_sampler = SequentialDistributedSampler(mydataset_test, batch_size=32)
testloader = DataLoader(mydataset_test, batch_size=32, sampler=test_sampler, collate_fn=collate_fn)

if local_rank == 0:
    print('Preparing data over!')
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 20
num_training_steps = num_epochs * len(train_loader)
if local_rank == 0:
    print("Total Steps:", num_training_steps)
progress_bar = tqdm(range(num_training_steps), disable=not local_rank == 0)
step = 0
every_evalstep = 500
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
if local_rank == 0:
    print("Start training")
model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
min_loss = 1000
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for image, ids in train_loader:
        image, ids = image.to(device), ids.to(device)
        optimizer.zero_grad()
        loss = model(**image, **ids)[0]
        loss.backward()
        optimizer.step()
        writer.add_scalar('train_loss', loss, step)
        step = step + 1
        progress_bar.update(1)
        if step % every_evalstep == 0:
            if local_rank == 0:
                print(f'Start {step // every_evalstep} eval')
            model.eval()
            with torch.no_grad():
                # 1. 得到本进程的prediction
                losses = []
                for image, ids in testloader:
                    image, ids = image.to(device), ids.to(device)
                    losses.append(model(**image, **ids)[0])
                # 进行gather
                loss_eval = distributed_concat(torch.stack(losses),
                                               len(test_sampler.dataset))
                current_loss = torch.mean(loss_eval)
            model.train()
            if local_rank == 0:
                print(f'Current eval loss is {current_loss} eval')
                writer.add_scalar('eval_loss', current_loss, step)
                if current_loss < min_loss:
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(output_dir)
                    min_loss = current_loss
            torch.distributed.barrier()
    lr_scheduler.step()

destroy_process_group()
