import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import random
import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SwiGLU(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        hidden_size = int(4 * (2/3))
        self.w = nn.Linear(n_embed, n_embed * hidden_size, bias=False)
        self.v = nn.Linear(n_embed, n_embed * hidden_size, bias=False)
        self.w2 = nn.Linear(n_embed * hidden_size, n_embed, bias=False)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.w2.weight)
    def forward(self, x):
        out = self.dropout(self.w2(F.silu(self.w(x)) * self.v(x)))
        return out

class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.value.weight)
    def forward(self,x,training):
        # x : (B,T,n_embed)
        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x)
        v = self.value(x)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if training else 0.0, is_causal=True) 
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X,training):
        out = torch.cat([head(X,training) for head in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class Block(nn.Module):
    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_heads, n_embed // n_heads, n_embed, block_size, dropout)
        self.ffwd = SwiGLU(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x,training):
        x = self.ln1(x)
        x = x + self.sa_heads(x,training)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, n_embed, n_heads, n_layers, vocab_size, block_size, dropout=0.2):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positon_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x,training=False):
        # x : (B,T)
        B, T = x.shape
        tok_emb = self.embedding_table(x)  # B,T,n_embed
        pos_emb = self.positon_embedding(torch.arange(T, device=x.device))  # T,n_embed
        x = tok_emb + pos_emb  # B,T,n_embed
        for block in self.blocks:
            x = block(x,training)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # B,T,vocab_size
        return logits
    
    @torch.inference_mode()
    def generate(self,tokenizer,max_tokens,text="",temperature=0.0):
        self.eval()
        if text=="":
            text=tokenizer.bos_token
        idxs = tokenizer(text=text,return_tensors='pt')['input_ids']
        for _ in range(max_tokens):
            idxs = idxs[:,-512:]
            logits = self(idxs)[:,-1,:]
            if temperature == 0.0:
                _, next_token = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            idxs = torch.cat((idxs, next_token), dim=1)
        self.train()
        return tokenizer.decode(token_ids=idxs[0])
            
            
class GPT2(L.LightningModule):
    def __init__(self, n_embed, n_heads, n_layers, vocab_size, block_size, lr, t_max, dropout=0.2):
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.model = TransformerDecoder(n_embed, n_heads, n_layers, vocab_size, block_size, dropout)
        self.lr = lr
        self.t_max = t_max
    def forward(self, x):
        return self.model(x)
    
    @torch.inference_mode()
    def generate(self,tokenizer,max_tokens,text="",temperature=0.0):
        return self.model.generate(tokenizer=tokenizer,text=text,max_tokens=max_tokens,temperature=temperature)
    
    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'])
        targets = batch['input_ids'][..., 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),ignore_index=50256)
        self.log('training_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'])
        targets = batch['input_ids'][..., 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),ignore_index=50256)
        self.log('validation_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=self.t_max)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                "scheduler": scheduler,
                "monitor": "training_loss",
                "interval": "step",
                "frequency": 1,
            }
        }

class GenerateCallback(L.pytorch.callbacks.Callback):
    def __init__(self,tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    def on_epoch_end(self, trainer, pl_module):
        generated_text = pl_module.generate(tokenizer=self.tokenizer,max_tokens=256,temperature=1.0)
        print("Generated text:", generated_text)