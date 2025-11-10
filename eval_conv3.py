import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore
    _HAS_TQDM = False

# ----- Vocabulary and Tokenization -----
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
CHEM_ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si",
    "P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co",
    "Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y",
    "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb",
    "Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu",
    "Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re",
    "Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr",
    "Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es",
    "Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg",
    "Cn","Nh","Fl","Mc","Lv","Ts","Og"
]
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
    return [token_to_id.get(tok, token_to_id["<UNK>"]) for tok in tokens]

def ids_to_tokens(ids):
    return [id_to_token.get(i, "<UNK>") for i in ids]

# ----- Dataset -----
class SpectraFormulaDataset(Dataset):
    def __init__(self, df, max_peaks=300, max_formula_len=50):
        self.enc_seqs = []
        self.dec_seqs = []
        for _, row in df.iterrows():
            peaks = sorted(row["peaks"], key=lambda x: x[0])
            peaks = peaks[:max_peaks] + [(0.0, 0.0)] * (max_peaks - len(peaks))
            self.enc_seqs.append(np.array(peaks, dtype=np.float32))
            token_ids = tokens_to_ids(tokenize_formula(row["formula"]))
            token_ids = token_ids[:max_formula_len] + [PAD_IDX] * (max_formula_len - len(token_ids))
            self.dec_seqs.append(np.array(token_ids, dtype=np.int64))
        self.enc_seqs = np.stack(self.enc_seqs, axis=0)
        self.dec_seqs = np.stack(self.dec_seqs, axis=0)

    def __len__(self):
        return len(self.enc_seqs)

    def __getitem__(self, idx):
        return self.enc_seqs[idx], self.dec_seqs[idx]

# ----- Model -----
class Encoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=8, max_peaks=300):
        super().__init__()
        self.d_model = d_model
        self.max_peaks = max_peaks
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
        self.global_pool = nn.AdaptiveAvgPool1d(max_peaks // 8)
        self.final_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x_scaled = x.clone()
        x_scaled[..., 0] = x_scaled[..., 0] / 1000.0
        x_scaled[..., 1] = x_scaled[..., 1] / 100.0
        x_scaled = x_scaled.transpose(1, 2)
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
        x = self.global_pool(x)
        x = x.transpose(1, 2)
        x = self.final_norm(x)
        x = x.permute(1, 0, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=8, vocab_size=len(VOCAB), max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, tgt, memory):
        bsz, tgt_len = tgt.shape
        tgt_embed = self.embedding(tgt) + self.pos_embedding[:, :tgt_len, :]
        tgt_embed = tgt_embed.permute(1, 0, 2)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_embed.device)
        dec_out = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        dec_out = dec_out.permute(1, 0, 2)
        logits = self.fc_out(dec_out)
        return logits

class SpectrumToFormulaModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=8, vocab_size=len(VOCAB), max_peaks=300, max_seq_len=50):
        super().__init__()
        self.encoder = Encoder(d_model=d_model, nhead=nhead, num_layers=num_layers, max_peaks=max_peaks)
        self.decoder = Decoder(d_model=d_model, nhead=nhead, num_layers=num_layers,
                               vocab_size=vocab_size, max_seq_len=max_seq_len)
    def forward(self, enc_inp, dec_inp):
        memory = self.encoder(enc_inp)
        logits = self.decoder(dec_inp, memory)
        return logits

# ----- Robust checkpoint loader -----
def load_checkpoint(model: nn.Module, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    # handle torch.compile _orig_mod.
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k[len("_orig_mod."):]: v for k, v in state.items() if k.startswith("_orig_mod.")}
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys after load ({len(missing)}):", missing)
    if unexpected:
        print(f"[WARN] Unexpected keys after load ({len(unexpected)}):", unexpected)

# ----- Decoding strategies -----
def generate_formula(
    model, 
    enc_input, 
    max_len=50, 
    strategy="greedy", 
    beam_width=3, 
    top_k=5, 
    top_p=0.9,
    device="cpu"
):
    model.eval()
    enc_input = torch.tensor(enc_input, dtype=torch.float32).unsqueeze(0).to(device)
    memory = model.encoder(enc_input)
    if strategy == "greedy":
        tokens = [SOS_IDX]
        for _ in range(max_len-1):
            inp = torch.tensor([tokens], device=device)
            logits = model.decoder(inp, memory)
            next_token = logits[0, -1].argmax(-1).item()
            tokens.append(next_token)
            if next_token == EOS_IDX:
                break
        return tokens
    elif strategy == "beam":
        beams = [([SOS_IDX], 0.0)]
        for _ in range(max_len-1):
            new_beams = []
            for seq, score in beams:
                inp = torch.tensor([seq], device=device)
                logits = model.decoder(inp, memory)
                probs = F.log_softmax(logits[0, -1], dim=-1)
                topk_probs, topk_idxs = torch.topk(probs, beam_width)
                for log_prob, idx in zip(topk_probs.tolist(), topk_idxs.tolist()):
                    new_seq = seq + [idx]
                    new_score = score + log_prob
                    new_beams.append((new_seq, new_score))
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq[-1] == EOS_IDX for seq, _ in new_beams):
                beams = new_beams
                break
            beams = new_beams
        return beams[0][0]
    elif strategy == "topk":
        tokens = [SOS_IDX]
        for _ in range(max_len-1):
            inp = torch.tensor([tokens], device=device)
            logits = model.decoder(inp, memory)
            probs = F.softmax(logits[0, -1], dim=-1)
            topk_probs, topk_idxs = torch.topk(probs, k=top_k)
            idx = torch.multinomial(topk_probs, 1).item()
            next_token = topk_idxs[idx].item()
            tokens.append(next_token)
            if next_token == EOS_IDX:
                break
        return tokens
    elif strategy == "topp":
        tokens = [SOS_IDX]
        for _ in range(max_len-1):
            inp = torch.tensor([tokens], device=device)
            logits = model.decoder(inp, memory)
            probs = F.softmax(logits[0, -1], dim=-1)
            sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            idx_mask = cumulative <= top_p
            idx_mask[0] = True
            filtered_idxs = sorted_idxs[idx_mask]
            filtered_probs = sorted_probs[idx_mask]
            filtered_probs = filtered_probs / filtered_probs.sum()
            idx = torch.multinomial(filtered_probs, 1).item()
            next_token = filtered_idxs[idx].item()
            tokens.append(next_token)
            if next_token == EOS_IDX:
                break
        return tokens
    else:
        raise ValueError("Unknown decoding strategy: " + strategy)

def decode_tokens(tokens):
    """Convert a sequence of token IDs to a clean formula string.
    - Trims at the first EOS if present
    - Removes SOS/EOS/PAD tokens from output
    """
    # Accept list or tensor
    if hasattr(tokens, 'detach'):
        tokens = tokens.detach().cpu().tolist()
    # Trim at EOS (do not include EOS)
    if EOS_IDX in tokens:
        tokens = tokens[:tokens.index(EOS_IDX)]
    # Map to tokens and filter control tokens
    toks = [t for t in ids_to_tokens(tokens) if t not in ("<SOS>", "<EOS>", "<PAD>")]
    return "".join(toks)

def sample_decode_and_compare(model, val_ds, n=5, device="cpu"):
    np.random.seed(42)
    indices = np.random.choice(len(val_ds), n, replace=False)
    for i in indices:
        enc, dec = val_ds[i]
        true_formula = decode_tokens(list(dec))
        print(f"\n[{i}] TRUE:  {true_formula}")
        for strat in ["greedy", "beam", "topk", "topp"]:
            out_ids = generate_formula(
                model, enc,
                strategy=strat,
                beam_width=5,
                top_k=5,
                top_p=0.92,
                device=device,
            )
            out_formula = decode_tokens(out_ids)
            print(f"  {strat:>6}: {out_formula}")

# ----- Standard evaluation (loss/accuracy) -----
def evaluate(model, loader, device):
    """Compute loss, token-level accuracy, and exact-sequence accuracy.
    - Token accuracy ignores PAD tokens.
    - Exact-sequence accuracy counts a sample correct if all non-PAD
      target tokens are predicted correctly (teacher forcing setting).
    """
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='sum')
    tot_loss = 0.0
    tot_tok = 0
    correct = 0
    seq_exact = 0
    seq_total = 0

    total_batches = None
    try:
        total_batches = len(loader)
    except Exception:
        pass

    iterator = loader
    pbar = None
    if _HAS_TQDM:
        pbar = tqdm(iterator, total=total_batches, desc="Evaluating", unit="batch")
        iterator = pbar

    with torch.no_grad():
        for idx, (enc, dec) in enumerate(iterator):
            enc, dec = enc.to(device), dec.to(device)
            logits = model(enc, dec[:, :-1])                 # (B, T-1, V)
            tgt = dec[:, 1:]                                  # (B, T-1)
            # Loss over non-PAD tokens
            loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            tot_loss += loss.item()
            # Token-level accuracy ignoring PAD
            mask = (tgt != PAD_IDX)
            pred = logits.argmax(-1)
            correct += ((pred == tgt) & mask).sum().item()
            tot_tok += mask.sum().item()

            # Exact-sequence accuracy (all non-PAD tokens must match)
            eq = ((pred == tgt) | (~mask))  # True for PAD positions
            batch_exact = eq.all(dim=1).sum().item()
            seq_exact += batch_exact
            seq_total += enc.size(0)

            # Live progress metrics
            if pbar is not None:
                avg_loss = (tot_loss / max(tot_tok, 1))
                acc = (correct / max(tot_tok, 1))
                seq_acc = (seq_exact / max(seq_total, 1))
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2%}", exact=f"{seq_acc:.2%}")
            elif total_batches is not None and (idx + 1) % 10 == 0:
                pct = 100.0 * (idx + 1) / total_batches
                print(f"Evaluating: {idx + 1}/{total_batches} ({pct:.1f}%)")

    if pbar is not None:
        pbar.close()

    # Avoid division by zero
    avg_loss = (tot_loss / max(tot_tok, 1))
    acc = (correct / max(tot_tok, 1))
    seq_acc = (seq_exact / max(seq_total, 1))
    return avg_loss, acc, seq_acc

# ----- Main -----
def main():
    p = argparse.ArgumentParser(description="Evaluate Spectrum→Formula model (with decoding strategies)")
    p.add_argument("--data_file", default="massbank_dataset.jsonl")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--encoding", default="utf-8", help="File encoding (try 'latin-1' if utf-8 fails)")
    args = p.parse_args()

    # Load dataset
    records = []
    with open(args.data_file, "rb") as f:
        file_iter = f
        if _HAS_TQDM:
            # Show line-level progress while reading
            file_iter = tqdm(f, desc="Reading dataset", unit="lines")
        for raw in file_iter:
            try:
                line = raw.decode(args.encoding)
            except UnicodeDecodeError:
                line = raw.decode("latin-1")
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "formula" in j and "peaks" in j and j["peaks"]:
                peaks = [(p.get("m/z", 0.0), p.get("intensity", 0.0)) for p in j["peaks"]]
                records.append({"formula": j["formula"], "peaks": peaks})
    if _HAS_TQDM and hasattr(file_iter, 'close'):
        try:
            file_iter.close()
        except Exception:
            pass
    if not records:
        raise ValueError("No valid records found – check encoding.")
    df = pd.DataFrame(records)
    _, val_df = train_test_split(df, test_size=0.99, random_state=1337) #test_size=0.1 #1.0 throws an error #42
    val_ds = SpectraFormulaDataset(val_df)
    loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = SpectrumToFormulaModel()
    load_checkpoint(model, args.checkpoint, args.device)
    model.to(args.device)

    loss, tok_acc, seq_acc = evaluate(model, loader, args.device)
    print(f"Validation cross-entropy loss: {loss:.4f}")
    print(f"Token-level accuracy: {tok_acc:.2%}")
    print(f"Exact formula accuracy: {seq_acc:.2%}")

    print("\n----- Decoding samples for validation set -----")
    sample_decode_and_compare(model, val_ds, n=25, device=args.device) #n=5

if __name__ == "__main__":
    main()
