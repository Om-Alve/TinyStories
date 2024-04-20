import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import GPT2,GenerateCallback,GPT2Config
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT2 model")
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories", help="Name of the dataset to load")
    parser.add_argument("--model_name", type=str, default="roneneldan/TinyStories", help="Name of the pretrained model to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--block_size',type=int,default=512,help="Block size")
    parser.add_argument("--n_embed", type=int, default=300, help="Embedding size")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--test_run",action="store_true")
    parser.add_argument("--dev_run",action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    dataset = load_dataset(args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"bos_token": "<bos>"})
    def transform(example):
        text = [tokenizer.bos_token + " " + text  + " " + tokenizer.eos_token for text in example['text']]
        return tokenizer(text=text, truncation=True, padding=True, max_length=args.block_size, return_tensors='pt')
    ds = dataset.with_transform(transform)

    train_loader = DataLoader(ds['train'],batch_size=args.batch_size,num_workers=4,shuffle=True)
    val_loader = DataLoader(ds['validation'],batch_size=args.batch_size,num_workers=4)
    num_epochs = args.num_epochs

    config = GPT2Config(
        vocab_size = tokenizer.vocab_size,
        block_size = args.block_size,
        lr = args.lr,
        n_embed = args.n_embed,
        n_heads = args.n_heads,
        n_layers = args.n_layers,
        dropout = args.dropout,
        t_max = len(train_loader) * num_epochs if not args.test_run else len(val_loader) * num_epochs,
    )
    model = GPT2(config)

    log_callback = L.pytorch.callbacks.ModelCheckpoint(save_top_k=1,mode='max',monitor='validation_loss',save_last=True)
    generate_callback = GenerateCallback(tokenizer=tokenizer)
    callbacks = [log_callback,generate_callback]
    trainer = L.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp',
        max_epochs=num_epochs,
        callbacks=callbacks,
        precision='16-mixed',
        fast_dev_run= True if args.dev_run else False,
        accumulate_grad_batches=6,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader if not args.test_run else val_loader,
        val_dataloaders=val_loader,
    )
    
    if args.generate:
        print(model.generate(tokenizer=tokenizer,max_tokens=256,temperature=1.0))
