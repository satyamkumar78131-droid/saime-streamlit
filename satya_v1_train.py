#!/usr/bin/env python3
"""
SATYA V1 - Training Script
==========================
Apna khud ka AI model, from scratch. (Hindi + English + Hinglish)

Ye kya hai?
  Ek chhota GPT-style Transformer (decoder-only). Ye text padh kar seekhta hai
  ki agla character kya aayega, aur phir khud text likh sakta hai.

Kaise chalaye (Google Colab ya PC):
  Train:    python satya_v1_train.py --data sample_data.txt --steps 500
  Generate: python satya_v1_train.py --mode generate \
                --checkpoint satya-v1-checkpoints/step-500.pt \
                --prompt "Namaste"

Tokenizer: character-level (koi download nahi chahiye).
  Hindi (Devanagari), English aur Hinglish (Roman Hindi) - teeno alag scripts
  hain. Character tokenizer har script ko handle kar leta hai, kyunki wo bas
  data me aaye hue characters ko seekhta hai. Isliye ye sabse simple tareeka
  hai - koi extra library ya vocab file nahi chahiye.

Model config (target ~50M parameters):
  d_model=640, n_layer=8, n_head=10, d_ff=2560, block_size=256.
  Script start hote hi EXACT parameter count print karta hai - kyunki
  character-level vocab chhoti hoti hai (~200 characters), embedding chhoti
  banti hai, isliye real count ~40M ke aas-paas aata hai. Exactly 50M chahiye
  to --n-layer 10 kar do, ya word-level vocab 8000 use karo.
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer (character-level)
# ---------------------------------------------------------------------------
class CharTokenizer:
    """Character-level tokenizer. Data me jitne unique characters hain,
    utni badi vocab banti hai. Koi download nahi, koi extra library nahi."""

    def __init__(self, text=None, chars=None):
        if chars is None:
            chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        # unknown character -> 0 (pehla character)
        return [self.stoi.get(ch, 0) for ch in s]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"chars": [self.itos[i] for i in range(self.vocab_size)]},
                f, ensure_ascii=False,
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            chars = json.load(f)["chars"]
        return cls(chars=chars)


# ---------------------------------------------------------------------------
# Model - GPT-style decoder Transformer
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention jo sirf pehle wale words ko dekh sakta hai
    (causal mask) - isiliye model aage ka text predict karna seekhta hai."""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0, "d_model, n_head se divide hona chahiye"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out(y))


class Block(nn.Module):
    """Ek Transformer block: attention + MLP, dono ke aage LayerNorm."""

    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SatyaV1(nn.Module):
    """SATYA V1 - decoder-only Transformer (GPT style)."""

    def __init__(self, vocab_size, d_model=640, n_layer=8, n_head=10,
                 d_ff=2560, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        """Prompt se aage ka text generate karo (temperature + top-k sampling)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(data, batch_size, seq_len, device):
    """Data se random contiguous chunks nikalo: x = input, y = next char."""
    max_start = max(1, len(data) - seq_len - 1)
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Learning-rate schedule: warmup + cosine decay
# ---------------------------------------------------------------------------
def get_lr(step, warmup_steps, total_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    text = load_text(args.data)
    print("Data file:", args.data, "| characters:", len(text))
    tok = CharTokenizer(text=text)
    print("Vocab size (unique characters):", tok.vocab_size)

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print("Total tokens:", len(data))

    seq_len = min(args.seq_len, max(16, len(data) // 4))
    if seq_len != args.seq_len:
        print("Data chhota hai, seq_len", args.seq_len, "->", seq_len, "kar diya.")

    model = SatyaV1(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
        block_size=args.block_size,
        dropout=args.dropout,
    ).to(device)

    n_params = count_params(model)
    print("=" * 50)
    print("SATYA V1 parameters: {:,}  ({:.1f} Million)".format(
        n_params, n_params / 1e6))
    print("=" * 50)

    os.makedirs(args.out, exist_ok=True)
    tok.save(os.path.join(args.out, "tokenizer.json"))

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    # Config bhi save karo, taaki generate karte time wahi config mile.
    config = {
        "vocab_size": tok.vocab_size,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "d_ff": args.d_ff,
        "block_size": args.block_size,
        "dropout": args.dropout,
        "params": n_params,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(data, args.batch_size, seq_len, device)
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping - training stable rehta hai.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            print("step {:>5}/{:>5} | loss {:.4f} | lr {:.2e} | {:.1f}s".format(
                step, args.steps, loss.item(), lr, dt))

        if step % args.save_every == 0 or step == args.steps:
            ckpt = os.path.join(args.out, "step-{}.pt".format(step))
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": config,
            }, ckpt)
            print("checkpoint saved:", ckpt)

    print("Training done! Checkpoints:", args.out)


# ---------------------------------------------------------------------------
# Generate (test the trained model)
# ---------------------------------------------------------------------------
def generate(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    tok_path = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = CharTokenizer.load(tok_path)
    else:
        raise SystemExit("tokenizer.json nahi mila: " + tok_path)

    model = SatyaV1(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        d_ff=config["d_ff"],
        block_size=config["block_size"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Loaded checkpoint:", args.checkpoint, "| step:", ckpt.get("step"))
    print("Parameters: {:,} ({:.1f}M)".format(
        count_params(model), count_params(model) / 1e6))

    prompt = args.prompt
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k,
    )
    print("-" * 50)
    print(tok.decode(out[0].tolist()))
    print("-" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="SATYA V1 - apna AI model, from scratch")
    p.add_argument("--mode", choices=["train", "generate"], default="train")
    p.add_argument("--data", default="sample_data.txt",
                   help="training text file (default: sample_data.txt)")
    p.add_argument("--out", default="satya-v1-checkpoints",
                   help="checkpoint folder (default: satya-v1-checkpoints)")
    p.add_argument("--steps", type=int, default=1000, help="training steps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=50, help="warmup steps")
    p.add_argument("--d-model", type=int, default=640)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=10)
    p.add_argument("--d-ff", type=int, default=2560)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="auto", help="auto / cuda / cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=250)
    # generate mode
    p.add_argument("--checkpoint", default="",
                   help="checkpoint .pt file (generate mode)")
    p.add_argument("--prompt", default="Namaste",
                   help="starting text for generation")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if not args.checkpoint:
            raise SystemExit("--mode generate ke liye --checkpoint chahiye.")
        generate(args)


if __name__ == "__main__":
    main()
