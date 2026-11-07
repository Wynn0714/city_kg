# -*- coding: utf-8 -*-
"""
BiLSTM-CRF 实体识别（基线模型 · GPU 版）
运行环境：Python 3.10+，建议 CUDA GPU
依赖：pip install torch pytorch-crf seqeval
      torch 需装 CUDA 版：pip install torch --index-url https://download.pytorch.org/whl/cu121

设备策略：自动检测，有 CUDA 用 GPU（cuda:0），否则回退 CPU。
用法：
    python train_bilstm.py            # 默认训练+评测，输出 ner/metrics_bilstm.json
    python train_bilstm.py --epochs 30 --batch 64
    python train_bilstm.py --no-cuda  # 强制 CPU（调试用）
"""
import os, json, argparse, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from seqeval.metrics import classification_report, f1_score as seq_f1

DEVICE = None  # 延迟初始化，在 main 里根据 args 确定

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NER  = os.path.join(BASE, "ner")

def load_bio(path):
    sents, chars, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if chars:
                    sents.append((chars, tags)); chars, tags = [], []
                continue
            c, t = line.split("\t")[:2]
            chars.append(c); tags.append(t)
    if chars:
        sents.append((chars, tags))
    return sents

def build_vocab(sents):
    cs = set(c for ch, _ in sents for c in ch)
    chars = ["<PAD>", "<UNK>"] + sorted(cs)
    ts = set(t for _, tg in sents for t in tg)
    tags = ["O"] + sorted([t for t in ts if t != "O"])
    return chars, tags

class NERDataset(Dataset):
    def __init__(self, sents, c2i, t2i):
        self.data = [(ch, tg) for ch, tg in sents if len(ch) <= 256]
        self.c2i, self.t2i = c2i, t2i
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        ch, tg = self.data[i]
        return (torch.tensor([self.c2i.get(c, 1) for c in ch], dtype=torch.long),
                torch.tensor([self.t2i[t] for t in tg], dtype=torch.long),
                len(ch))

def collate(batch):
    xs, ys, ls = zip(*batch)
    maxl = max(ls)
    xpad = torch.zeros(len(xs), maxl, dtype=torch.long)
    ypad = torch.zeros(len(ys), maxl, dtype=torch.long)
    mask = torch.zeros(len(xs), maxl, dtype=torch.bool)
    for i, (x, y, l) in enumerate(zip(xs, ys, ls)):
        xpad[i, :l] = x[:l]; ypad[i, :l] = y[:l]; mask[i, :l] = True
    return xpad, ypad, mask, list(ls)

class BiLSTMCRF(nn.Module):
    def __init__(self, nchar, ntag, emb=100, hid=128, drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(nchar, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, bidirectional=True, batch_first=True, num_layers=1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hid * 2, ntag)
        self.crf = CRF(ntag, batch_first=True)
    def forward(self, x, mask):
        e = self.drop(self.emb(x))
        h, _ = self.lstm(e)
        return self.fc(h), mask
    def loss(self, logits, y, mask):
        return -self.crf(logits, y, mask=mask, reduction="mean")
    def decode(self, logits, mask):
        return self.crf.decode(logits, mask=mask)

def evaluate(model, loader, i2t, device):
    model.eval()
    golds, preds = [], []
    with torch.no_grad():
        for x, y, mask, ls in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits, mask = model(x, mask)
            out = model.decode(logits, mask)
            for i, l in enumerate(ls):
                golds.append([i2t[y[i, j].item()] for j in range(l)])
                preds.append([i2t[out[i][j]] for j in range(l)])
    return golds, preds

def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--emb", type=int, default=100)
    ap.add_argument("--no-cuda", action="store_true", help="强制使用 CPU")
    args = ap.parse_args()

    # 设备检测：有 CUDA 用 GPU，否则 CPU
    if args.no_cuda or not torch.cuda.is_available():
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda:0")
    print(f"[device] 使用设备: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    train_s = load_bio(os.path.join(NER, "train.bio"))
    dev_s   = load_bio(os.path.join(NER, "dev.bio"))
    test_s  = load_bio(os.path.join(NER, "test.bio"))
    chars, tags = build_vocab(train_s)
    c2i = {c: i for i, c in enumerate(chars)}
    t2i = {t: i for i, t in enumerate(tags)}
    i2t = {i: t for t, i in t2i.items()}
    nchar, ntag = len(chars), len(tags)
    print(f"[vocab] chars={nchar} tags={ntag}")

    train_ds = NERDataset(train_s, c2i, t2i)
    dev_ds   = NERDataset(dev_s, c2i, t2i)
    test_ds  = NERDataset(test_s, c2i, t2i)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dev_dl   = DataLoader(dev_ds, batch_size=args.batch, collate_fn=collate)
    test_dl  = DataLoader(test_ds, batch_size=args.batch, collate_fn=collate)

    model = BiLSTMCRF(nchar, ntag, emb=args.emb, hid=args.hid).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1, wait = 0.0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y, mask, _ in train_dl:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            opt.zero_grad()
            logits, mask = model(x, mask)
            loss = model.loss(logits, y, mask)
            loss.backward(); opt.step()
            tot += loss.item()
        g, p = evaluate(model, dev_dl, i2t, DEVICE)
        f1 = seq_f1(p, g)
        print(f"epoch {ep:02d}  loss={tot/len(train_dl):.3f}  dev_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; wait = 0
            torch.save(model.state_dict(), os.path.join(NER, "bilstm_crf.pt"))
        else:
            wait += 1
            if wait >= 5:
                print("early stop"); break

    model.load_state_dict(torch.load(os.path.join(NER, "bilstm_crf.pt"), map_location=DEVICE))
    g, p = evaluate(model, test_dl, i2t, DEVICE)
    rep = classification_report(g, p, output_dict=True, zero_division=0)
    overall = seq_f1(p, g)
    metrics = {
        "model": "BiLSTM-CRF",
        "test_macro_f1": overall,
        "per_type": {k: {"precision": v.get("precision"), "recall": v.get("recall"), "f1": v.get("f1-score")}
                     for k, v in rep.items() if k not in ("macro avg", "weighted avg", "micro avg")},
        "hyperparams": vars(args),
    }
    with open(os.path.join(NER, "metrics_bilstm.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n===== BiLSTM-CRF 测试结果 =====")
    print(classification_report(g, p, zero_division=0))
    print("已保存 ner/metrics_bilstm.json")

if __name__ == "__main__":
    main()
