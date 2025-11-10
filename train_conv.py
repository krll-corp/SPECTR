import torch

#torch.set_default_dtype(torch.bfloat16)
torch.set_float32_matmul_precision('high') #because Jetson Orin Nano has Ampere GPU. TF32

import os
import math
import json
import wandb
import argparse
import time
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch._dynamo
torch._dynamo.config.suppress_errors = True

p = argparse.ArgumentParser()
p.add_argument('--resume', action='store_true', help='Resume wandb run and training state if possible')
p.add_argument('--checkpoint', type=str, default='checkpoint_last.pt', help='Path to training checkpoint file')
p.add_argument('--save-every', type=int, default=500, help='Save checkpoint every N optimizer steps')
args = p.parse_args()

RUN_ID_FILE = "wandb_run_id.txt"
wb_settings = wandb.Settings(_disable_stats=True, code_dir='.', start_method='thread', init_timeout=600)
if args.resume and os.path.exists(RUN_ID_FILE):
    run_id = open(RUN_ID_FILE).read().strip()
    run = wandb.init(project='SPECTR', id=run_id, resume='allow', settings=wb_settings)
else:
    run = wandb.init(project='SPECTR', name='conv_v1.1', settings=wb_settings)
    with open(RUN_ID_FILE, 'w') as f: f.write(run.id)


SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
CHEM_ELEMENTS = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si",
                 "P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co",
                 "Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y",
                 "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb",
                 "Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu",
                 "Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re",
                 "Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr",
                 "Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es",
                 "Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg",
                 "Cn","Nh","Fl","Mc","Lv","Ts","Og"]

DIGITS = [str(i) for i in range(10)]
VOCAB = SPECIAL_TOKENS + CHEM_ELEMENTS + DIGITS

token_to_id = {token: idx for idx, token in enumerate(VOCAB)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

PAD_IDX = token_to_id["<PAD>"]
SOS_IDX = token_to_id["<SOS>"]
EOS_IDX = token_to_id["<EOS>"]

def tokenize_formula(formula_str):
    tokens = []
    i = 0
    while i < len(formula_str):
        if i+1 < len(formula_str) and formula_str[i:i+2] in CHEM_ELEMENTS:
            tokens.append(formula_str[i:i+2])
            i += 2
            num_str = ""
            while i < len(formula_str) and formula_str[i].isdigit():
                num_str += formula_str[i]
                i += 1
            for d in num_str:
                tokens.append(d)
            continue
        elif formula_str[i] in [el for el in CHEM_ELEMENTS if len(el) == 1]:
            tokens.append(formula_str[i])
            i += 1
            num_str = ""
            while i < len(formula_str) and formula_str[i].isdigit():
                num_str += formula_str[i]
                i += 1
            for d in num_str:
                tokens.append(d)
            continue
        else:
            i += 1
    return ["<SOS>"] + tokens + ["<EOS>"]

def tokens_to_ids(tokens):
    return [token_to_id[tok] for tok in tokens if tok in token_to_id]

def ids_to_tokens(ids):
    return [id_to_token[i] for i in ids]


class SpectraFormulaDataset(Dataset):
    def __init__(self, df, max_peaks=300, max_formula_len=50):
        self.df = df.reset_index(drop=True)
        self.max_peaks = max_peaks
        self.max_formula_len = max_formula_len

        self.enc_seqs = []
        self.dec_seqs = []

        for _, row in self.df.iterrows():
            peaks = sorted(row["peaks"], key=lambda x: x[0])
            if len(peaks) > max_peaks:
                peaks = peaks[:max_peaks]
            else:
                peaks += [(0.0, 0.0)] * (max_peaks - len(peaks))
            enc_seq = np.array(peaks, dtype=np.float32)
            formula_str = row["formula"]
            tokens = tokenize_formula(formula_str)
            token_ids = tokens_to_ids(tokens)
            if len(token_ids) > max_formula_len:
                token_ids = token_ids[:max_formula_len]
            else:
                token_ids += [PAD_IDX] * (max_formula_len - len(token_ids))
            self.enc_seqs.append(enc_seq)
            self.dec_seqs.append(np.array(token_ids, dtype=np.int64))

        self.enc_seqs = np.stack(self.enc_seqs, axis=0)
        self.dec_seqs = np.stack(self.dec_seqs, axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.enc_seqs[idx], self.dec_seqs[idx]



class Encoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, max_peaks=300):
        super().__init__()
        self.d_model = d_model
        self.max_peaks = max_peaks
        
        # CNN layers for processing peaks
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, d_model, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2, stride=2)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(256)
        self.norm4 = nn.BatchNorm1d(d_model)
        
        # Global average pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool1d(max_peaks // 8)  # Reduce sequence length
        
        # Final projection to match transformer decoder expectations
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch_size, max_peaks, 2)
        """
        # Scale the inputs to prevent large values
        x_scaled = x.clone()
        x_scaled[..., 0] = x_scaled[..., 0] / 1000.0
        x_scaled[..., 1] = x_scaled[..., 1] / 100.0
        
        # Transpose for Conv1d: (batch_size, 2, max_peaks)
        x_scaled = x_scaled.transpose(1, 2)
        
        # CNN forward pass
        x = torch.relu(self.norm1(self.conv1(x_scaled)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.norm3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.norm4(self.conv4(x)))
        
        # Global pooling to get consistent output size
        x = self.global_pool(x)  # (batch_size, d_model, seq_len)
        
        # Transpose back: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        # Permute for Transformer decoder: (seq_len, batch_size, d_model)
        enc_out = x.permute(1, 0, 2)
        
        return enc_out

class Decoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, vocab_size=len(VOCAB), max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        """
        tgt: (batch_size, max_seq_len)
        memory: (src_seq_len, batch_size, d_model)
        """
        bsz, tgt_len = tgt.shape
        tgt_embed = self.embedding(tgt)  # (bsz, tgt_len, d_model)
        pos_embed_slice = self.pos_embedding[:, :tgt_len, :]
        tgt_embed = tgt_embed + pos_embed_slice
        # Permute to (tgt_len, bsz, d_model)
        tgt_embed = tgt_embed.permute(1, 0, 2)
        # Create causal mask so that each token can only attend to previous tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_embed.device)
        dec_out = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        dec_out = dec_out.permute(1, 0, 2)  # (bsz, tgt_len, d_model)
        logits = self.fc_out(dec_out)
        return logits

class SpectrumToFormulaModel(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, 
                 vocab_size=len(VOCAB), max_peaks=300, max_seq_len=50):
        super().__init__()
        self.encoder = Encoder(d_model=d_model, nhead=nhead, num_layers=num_layers, max_peaks=max_peaks)
        self.decoder = Decoder(d_model=d_model, nhead=nhead, num_layers=num_layers, 
                               vocab_size=vocab_size, max_seq_len=max_seq_len)

    def forward(self, enc_inp, dec_inp):
        memory = self.encoder(enc_inp)
        logits = self.decoder(dec_inp, memory)
        return logits

##############################################################################
# 4) Training loop с выводом шагов (steps)
##############################################################################
def save_checkpoint(model, optimizer, scheduler, global_step, path, run_id=None):
    try:
        payload = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'global_step': int(global_step),
            'run_id': run_id,
        }
        torch.save(payload, path)
        print(f"[CKPT] Saved checkpoint to {path} at step {global_step}.")
    except Exception as e:
        print(f"[CKPT] Failed to save checkpoint to {path}: {e}")

def _masked_token_accuracy(logits, tgt_out, pad_idx: int):
    """Compute token accuracy ignoring PAD tokens."""
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = (tgt_out != pad_idx)
        correct = ((pred == tgt_out) & mask).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0
        return acc, correct, total


def evaluate(model, loader, device):
    """Validation over a DataLoader: CE and token-accuracy ignoring PAD."""
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='sum')
    tot_loss = 0.0
    tot_tok = 0
    correct = 0
    with torch.no_grad():
        for enc, dec in loader:
            enc, dec = enc.to(device), dec.to(device)
            logits = model(enc, dec[:, :-1])
            tgt = dec[:, 1:]
            loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            tot_loss += loss.item()
            acc, corr, nnz = _masked_token_accuracy(logits, tgt, PAD_IDX)
            correct += corr
            tot_tok += nnz
    avg_loss = tot_loss / max(tot_tok, 1)
    acc = correct / max(tot_tok, 1)
    return avg_loss, acc


def _build_warmup_cosine(optimizer, warmup_steps: int, max_steps: int, min_lr_frac: float = 0.1):
    """Linear warmup then cosine decay to min_lr_frac of base LR."""
    min_lr_frac = float(min_lr_frac)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_model_steps(
    model,
    train_loader,
    val_loader,
    device,
    max_steps,
    lr=1e-4,
    log_every=1, #100
    grad_accum_steps=8,
    val_every=1000,
    warmup_steps=1000,
    min_lr_frac=0.1,
    checkpoint_path=None,
    save_every=None,
    resume=False,
    run_id=None,
):
    model.train()
    # Sum reduction to control averaging denominator explicitly (exclude PAD via ignore_index)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = _build_warmup_cosine(optimizer, warmup_steps=warmup_steps, max_steps=max_steps, min_lr_frac=min_lr_frac)

    global_step = 0
    micro_step = 0

    # Optionally load from checkpoint
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
                scheduler.load_state_dict(ckpt['scheduler'])
            global_step = int(ckpt.get('global_step', 0))
            print(f"[CKPT] Resumed from {checkpoint_path} at step {global_step}.")
        except Exception as e:
            print(f"[CKPT] Failed to load checkpoint from {checkpoint_path}: {e}")

    optimizer.zero_grad(set_to_none=True)
    last_step_time = time.perf_counter()

    if global_step >= max_steps:
        print(f"[CKPT] global_step {global_step} >= max_steps {max_steps}; nothing to do.")
        return

    try:
        while global_step < max_steps:
            for enc_inp, dec_inp in train_loader:
                enc_inp = enc_inp.to(device)
                dec_inp = dec_inp.to(device)

                tgt_in = dec_inp[:, :-1]
                tgt_out = dec_inp[:, 1:].contiguous()

                logits = model(enc_inp, tgt_in)

                # CE mean over non-PAD tokens
                tok_mask = (tgt_out != PAD_IDX)
                tok_count = tok_mask.sum().clamp_min(1).item()
                loss_sum = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                loss_mean = loss_sum / tok_count

                # Backprop with grad accumulation
                (loss_mean / grad_accum_steps).backward()
                micro_step += 1

                if micro_step % grad_accum_steps == 0:
                    # Gradient norm + clip
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # Train metrics (on last micro-batch)
                    acc_batch, corr, nnz = _masked_token_accuracy(logits, tgt_out, PAD_IDX)
                    now = time.perf_counter()
                    step_time = now - last_step_time
                    last_step_time = now
                    lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                    tokens_per_sec = nnz / max(step_time, 1e-6)

                    if global_step % log_every == 0:
                        loss_float = float(loss_mean.detach().cpu())
                        wandb.log({
                            'train/step': global_step,
                            'train/loss': loss_float,
                            'train/perplexity': float(math.exp(min(loss_float, 20.0))),
                            'train/acc_token': float(acc_batch),
                            'train/grad_norm': float(getattr(grad_norm, 'item', lambda: grad_norm)()),
                            'train/lr': float(lr_now),
                            'train/tokens_per_sec': float(tokens_per_sec),
                        })
                        print(f"Step {global_step}/{max_steps} | loss {loss_mean:.4f} | acc {acc_batch:.3f} | lr {lr_now:.2e}")

                    # Periodic checkpoint save
                    if checkpoint_path and save_every and (global_step % save_every == 0):
                        save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_path, run_id=run_id)

                    # Validation
                    if global_step % val_every == 0 and val_loader is not None:
                        val_loss, val_acc = evaluate(model, val_loader, device)
                        wandb.log({
                            'val/step': global_step,
                            'val/loss': float(val_loss),
                            'val/acc_token': float(val_acc),
                            'val/perplexity': float(math.exp(min(val_loss, 20.0))),
                        })
                        print(f"[VAL] step {global_step} | loss {val_loss:.4f} | acc {val_acc:.3f}")

                    if global_step >= max_steps:
                        break
            if global_step >= max_steps:
                break
    except KeyboardInterrupt:
        print("Training interrupted (KeyboardInterrupt), saving checkpoint...")
        if checkpoint_path:
            save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_path, run_id=run_id)
        raise
    except Exception as e:
        print(f"Training error: {e}. Saving checkpoint before raising...")
        if checkpoint_path:
            save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_path, run_id=run_id)
        raise
    finally:
        # Save a final checkpoint at the end of training as well
        if checkpoint_path:
            save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_path, run_id=run_id)
            
            
            

##############################################################################
# 5) Пример main() c train/val split
##############################################################################
if __name__ == "__main__":
    device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    data_file = "../mona_massbank_dataset.jsonl" #massbank_dataset.jsonl
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"File {data_file} not found!")
    records = []
    with open(data_file, "r") as f:
        for line in f:
            line_data = json.loads(line)
            if "formula" in line_data and "peaks" in line_data:
                if len(line_data["peaks"]) > 0 and len(line_data["formula"]) > 0:
                    peaks_list = []
                    for p in line_data["peaks"]:
                        mz = p.get("m/z", 0.0)
                        intensity = p.get("intensity", 0.0)
                        peaks_list.append((mz, intensity))
                    records.append({
                        "formula": line_data["formula"], 
                        "peaks": peaks_list
                    })
    import pandas as pd
    df = pd.DataFrame(records)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = SpectraFormulaDataset(train_df, max_peaks=300, max_formula_len=50)
    val_dataset   = SpectraFormulaDataset(val_df, max_peaks=300, max_formula_len=50)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0) #256
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=0)
    # Using a smaller learning rate for stability
    model = SpectrumToFormulaModel(d_model=256, nhead=8, num_layers=8, vocab_size=len(VOCAB), max_peaks=300, max_seq_len=50) # base size
    #model = SpectrumToFormulaModel(d_model=512, nhead=16, num_layers=16, vocab_size=len(VOCAB), max_peaks=300, max_seq_len=50) #medium size
    #model = SpectrumToFormulaModel(d_model=128, nhead=8, num_layers=4, vocab_size=len(VOCAB), max_peaks=300, max_seq_len=50) #small size

    # Move to target device before optional compilation
    model.to(device)

    # Count params before potential wrapping by torch.compile
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # torch.compile with Inductor is unstable on MPS for some dynamic reductions.
    # Avoid Inductor on MPS by using the 'eager' backend (or skipping compile entirely).
    #model = torch.compile(model, backend="aot_eager")
    print("model has", param_count, "parameters")
    max_steps = 10000 #100000 #3000
    warmup_steps = max(1, int(0.03 * max_steps))
    print("Loaded dataset with", len(train_dataset), "records.")

    # Log run config to W&B
    wandb.config.update({
        'model/d_model': 256,
        'model/nhead': 8,
        'model/num_layers': 8,
        'data/max_peaks': 300,
        'data/max_seq_len': 50,
        'train/batch_size': 64,
        'train/grad_accum_steps': 16,
        'train/lr': 1e-4,
        'train/max_steps': max_steps,
        'train/warmup_steps': warmup_steps,
        'train/scheduler': 'warmup+cosine',
        'optimizer': 'AdamW',
    }, allow_val_change=True)
    try:
        train_model_steps(
            model,
            train_loader,
            val_loader,
            device,
            max_steps,
            lr=1e-4,
            log_every=1,
            grad_accum_steps=16,
            val_every=100, #500
            warmup_steps=warmup_steps,
            min_lr_frac=0.1,
            checkpoint_path=args.checkpoint,
            save_every=args.save_every,
            resume=args.resume,
            run_id=run.id,
        )
    except Exception as e:
        print("Training interrupted, saving model...")
        torch.save(model.state_dict(), "spectrum2formula_max_10k_conv3-base.pth") #spectrum2formula_max_10k_conv.pth
    torch.save(model.state_dict(), "spectrum2formula_max_10k_conv3-base.pth") #spectrum2formula_max_10k_conv.pth

    print("Model saved")
