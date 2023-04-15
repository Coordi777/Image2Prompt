import json
import os
from pathlib import Path

from torch import nn
from torch.distributed import destroy_process_group
from transformers.trainer_pt_utils import SequentialDistributedSampler

from dataset import ImageDataset_csv, ImageDataset_json
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


def distributed_concat(tensor, num_total_examples):
    output_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


writer = SummaryWriter(log_dir='checkpoints_clean512')

output_dir = '/remote-home/bxy/kaggle/Image2Text/checkpoints_clean512'
# path_csv = '/remote-home/bxy/data/Image2Text_6G/train.csv'
# path_csv_test = '/remote-home/bxy/data/Image2Text_6G/eval.csv'
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-large")
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
criterion = nn.CosineEmbeddingLoss()


# def collate_fn(data):
#     batch = list(zip(*data))
#     images = processor(batch[0], return_tensors="pt")
#     prompt = tokenizer(batch[1], padding=True, truncation=True, return_tensors="pt")
#     del prompt['token_type_ids']
#     return images, prompt

def collate_fn(data):
    batch = list(zip(*data))
    images = processor(batch[0], return_tensors="pt")
    # prompt = tokenizer(batch[1], padding=True, truncation=True, return_tensors="pt")
    prompt_st = st_model.encode(batch[1], convert_to_tensor=True)
    # del prompt['token_type_ids']
    return images, prompt_st


if local_rank == 0:
    print('Preparing data...')
# 数据集准备
# mydataset_all = ImageDataset_json('SD_1M.json')
mydataset_all = ImageDataset_csv('diffusiondb512.csv')
length = len(mydataset_all)
test_size = int(0.1 * length)
mydataset_train, mydataset_test = random_split(mydataset_all, lengths=[0.9, 0.1])
# 分布式采样器
train_sampler = DistributedSampler(mydataset_train, shuffle=True)
train_loader = DataLoader(mydataset_train, batch_size=6, sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
test_sampler = SequentialDistributedSampler(mydataset_test, batch_size=64)
testloader = DataLoader(mydataset_test, batch_size=64, sampler=test_sampler, collate_fn=collate_fn, num_workers=0)

if local_rank == 0:
    print('Preparing data over!')
optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 20
num_training_steps = num_epochs * len(train_loader)
if local_rank == 0:
    print("Total Steps:", num_training_steps)
progress_bar = tqdm(range(num_training_steps), disable=not local_rank == 0)
step = 0
every_evalstep = 500
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_training_steps=num_training_steps,
)
if local_rank == 0:
    print("Start training")
model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
output_final_model = os.path.join('/remote-home/bxy/kaggle/Image2Text/final', WEIGHTS_NAME)
output_final_config = os.path.join('/remote-home/bxy/kaggle/Image2Text/final', CONFIG_NAME)
min_loss = 1000
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for image, ids in train_loader:
        image, ids = image.to(device), ids.to(device)
        optimizer.zero_grad()
        generated = model.generate(**image)
        out = processor.batch_decode(generated, skip_special_tokens=True)
        prd = st_model.encode(out, convert_to_tensor=True)
        target = torch.ones(ids.size(0)).to(device)
        loss = criterion(prd, ids, target).requires_grad_(True)
        # loss = model(**image, **ids)[0]

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
                    generated = model.generate(**image)
                    out = processor.batch_decode(generated, skip_special_tokens=True)
                    prd = st_model.encode(out, convert_to_tensor=True)
                    target = torch.ones(ids.size(0)).to(device)
                    e_loss = criterion(prd, ids, target)
                    # e_loss = model(**image, **ids)[0]
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
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(output_dir)
                    min_loss = current_loss
            torch.distributed.barrier()
        lr_scheduler.step()
torch.save(model_to_save.state_dict(), output_final_model)
model_to_save.config.to_json_file(output_final_config)
tokenizer.save_vocabulary('/remote-home/bxy/kaggle/Image2Text/final')

destroy_process_group()
