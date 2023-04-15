import os
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    def __init__(self, max_length=60, tokenizer_pre=None):
        path = 'metadata_withcmt.parquet'
        dataframe = pd.read_parquet(path)
        self.prompt_list = list(dataframe[dataframe['remove_idx_95']]['prompt'])
        self.max_length = max_length
        self.tokenizer_pre = tokenizer_pre
        self.input_ids, self.attention_mask = self.pre_process()

    def __len__(self):
        return len(self.input_ids)

    def pre_process(self):
        prompt_list_clean = []
        for sentence in self.prompt_list:
            if len(sentence.split()) > self.max_length:
                continue
            prompt_list_clean.append(sentence)
        embeddings = tokenizer(prompt_list_clean, padding=True, truncation=True, return_tensors="pt",max_length='longest')
        input_ids = embeddings['input_ids']
        attention_mask = embeddings['attention_mask']
        return input_ids, attention_mask

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        return input_ids, attention_mask


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

dataset_all = ImageDataset(tokenizer_pre=tokenizer)
train_dataset, test_dataset = train_test_split(dataset_all, test_size=0.2, random_state=42)
print("Train dataset length: " + str(len(train_dataset)))
print("Test dataset length: " + str(len(test_dataset)))
training_args = TrainingArguments(
    output_dir="./gpt2_fintune",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=5,  # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps=400,  # Number of update steps between two evaluations.
    save_steps=800,  # after # steps model is saved
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    disable_tqdm=False,
    no_cuda=False,
    prediction_loss_only=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                  eval_dataset=test_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                         'attention_mask': torch.stack(
                                                                             [f[1] for f in data])}).train()
trainer.train()
trainer.save_model()
