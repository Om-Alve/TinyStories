import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torchmetrics
import random
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import GPT2

if __name__ == '__main__':
    dataset = load_dataset("roneneldan/TinyStories")


    tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories')
    tokenizer.pad_token = tokenizer.eos_token

    def transform(example):
        return tokenizer(text=example['text'],truncation=True,padding=True,max_length=512,return_tensors='pt')

    ds = dataset.with_transform(transform)

    batch_size = 32
    train_loader = DataLoader(ds['train'],batch_size=batch_size,num_workers=4,shuffle=True)
    val_loader = DataLoader(ds['validation'],batch_size=batch_size,num_workers=4)

    vocab_size = tokenizer.vocab_size
    block_size = 512
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_embed = 300
    n_heads = 12
    n_layer = 4
    dropout = 0.2
    num_epochs = 5
    model = GPT2(n_embed=n_embed,block_size=block_size,vocab_size=vocab_size,n_heads=n_heads,n_layers=n_layer,lr=lr,t_max=num_epochs*len(train_loader))

    callback = L.pytorch.callbacks.ModelCheckpoint(save_top_k=1,mode='max',monitor='validation_loss',save_last=True)
    trainer = L.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp',
        max_epochs=num_epochs,
        callbacks = [callback],
        precision='16-mixed',
    )

    trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
