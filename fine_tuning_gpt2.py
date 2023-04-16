import os

import pandas as pd
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import WEIGHTS_NAME, CONFIG_NAME

num_epochs = 25
UNFREEZE_START = 9
SEED = 3407
output_dir = 'gpt2_fintune'
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
writer = SummaryWriter(log_dir=output_dir)
torch.distributed.init_process_group(backend="nccl")


class ImageDataset(Dataset):
    def __init__(self, max_length=70, tokenizer_pre=None):
        path = 'metadata_withcmt.parquet'
        dataframe = pd.read_parquet(path)
        self.prompt_list = list(dataframe[dataframe['remove_idx_95']]['prompt'])
        self.max_length = max_length
        self.tokenizer_pre = tokenizer_pre
        self.prompt_list_clean = self.pre_process()

    def __len__(self):
        return len(self.prompt_list_clean)

    def pre_process(self):
        prompt_list_clean = []
        for sentence in self.prompt_list:
            if len(sentence.split()) > self.max_length:
                continue
            prompt_list_clean.append(sentence)
        return prompt_list_clean

    def __getitem__(self, idx):
        prompt = self.prompt_list_clean[idx]

        return prompt


def load_pretrained_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    trainable_model_weights = False
    for name, child in model.named_children():
        if name == 'transformer':
            for pn, p in child.named_parameters():
                if str(UNFREEZE_START) in pn:
                    """start unfreezing layer , the weights are trainable"""
                    trainable_model_weights = True
                p.requires_grad = trainable_model_weights
    return model


def cleanup():
    dist.destroy_process_group()


def run_demo():
    # 计算global_rank和world_size
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # 设置seed
    torch.manual_seed(SEED)

    # 创建模型, 并将其移动到local_rank对应的GPU上
    model = load_pretrained_model().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    dataset_all = ImageDataset(tokenizer_pre=tokenizer)
    train_dataset, test_dataset = train_test_split(dataset_all, test_size=0.2, random_state=42)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    train_sampler = DistributedSampler(train_dataset,
                                       rank=local_rank)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=32,
                              collate_fn=lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=71,
                                                             return_tensors="pt"))
    test_sampler = DistributedSampler(test_dataset,
                                      rank=local_rank)
    testloader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler, num_workers=32,
                            collate_fn=lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=71,
                                                           return_tensors="pt"))
    optimizer.zero_grad()
    step = 0
    step_eval = 1000
    min_loss = 10000
    num_training_steps = num_epochs * len(train_loader)
    if local_rank == 0:
        print("Total Steps:", num_training_steps)
    progress_bar = tqdm(range(num_training_steps), disable=not local_rank == 0)
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for data in train_loader:
            input_ids, attention_mask = data['input_ids'].to(local_rank), data['attention_mask'].to(local_rank)
            outputs = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step = step + 1
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            if local_rank == 0:
                writer.add_scalar('train_loss', loss, step)
            dist.barrier()
            if step % step_eval == 0:
                if local_rank == 0:
                    print(f'\nStart {step // step_eval} eval')
                ddp_model.eval()
                total_loss = 0
                with torch.no_grad():
                    for data in tqdm(testloader, disable=not local_rank == 0):
                        input_ids, attention_mask = data['input_ids'].to(local_rank), data['attention_mask'].to(
                            local_rank)
                        outputs = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                        loss = outputs[0]
                        total_loss += loss.item()
                    dist.barrier()
                    total_loss = torch.tensor(total_loss, requires_grad=False).to(local_rank)
                    dist.all_reduce(total_loss)
                    avg_loss = total_loss.item() / len(testloader)
                    ddp_model.train()

                    if local_rank == 0:
                        print(f"step {step}, avg_loss {avg_loss}")
                        writer.add_scalar('eval_loss', avg_loss, step // step_eval)
                        if avg_loss < min_loss:
                            min_loss = avg_loss
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(output_dir)
                    dist.barrier()
    cleanup()


if __name__ == "__main__":
    run_demo()
