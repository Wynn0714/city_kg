# -*- coding: utf-8 -*-
"""
BERT-CRF 实体识别（主模型，需 GPU）
运行环境：Python 3.10+，建议 CUDA GPU
依赖：pip install torch transformers pytorch-crf seqeval

用法：
    python train_bert.py                       # 默认 bert-base-chinese
    python train_bert.py --model bert-base-chinese --epochs 5 --batch 16
首次运行会自动下载 bert-base-chinese（约 400MB）。
"""
import os, json, argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from transformers import BertTokenizer, BertModel
from seqeval.metrics import classification_report, f1_score as seq_f1

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NER  = os.path.join(BASE, "ner")
MAXLEN = 128

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

def build_tags(sents):
    ts = set(t for _, tg in sents for t in tg)
    return ["O"] + sorted([t for t in ts if t != "O"])

def sub_tag(t):
    """词的后续子词（##xx）继承标签：B-X -> I-X，其余（O / I-X）保持不变，
    保证 BIO 链在子词切分后依然合法。"""
    if t.startswith("B-"):
        return "I-" + t[2:]
    return t

class BERTNERDataset(Dataset):
    def __init__(self, sents, tokenizer, t2i):
        self.data, self.tokenizer, self.t2i = [], tokenizer, t2i
        for ch, tg in sents:
            if len(ch) > MAXLEN - 2:
                ch, tg = ch[:MAXLEN - 2], tg[:MAXLEN - 2]
            enc = tokenizer(ch, is_split_into_words=True, max_length=MAXLEN,
                           truncation=True, padding="max_length")
            wids = enc.word_ids()
            labels = []
            prev = None
            for wi in wids:
                if wi is None:
                    # [CLS] / [SEP] / [PAD]：必须填合法标签 id（0 = "O"）。
                    # 注意：CRF 不接受 -100（那是 CrossEntropyLoss 的忽略值），
                    # torchcrf 第 0 个位置的 tag 会直接用于索引，越界即触发 CUDA assert。
                    labels.append(0)
                elif wi != prev:
                    labels.append(t2i[tg[wi]])          # 词的首个子词：真实标签
                else:
                    labels.append(t2i[sub_tag(tg[wi])])  # 后续子词：继承 B->I
                prev = wi
            self.data.append((enc["input_ids"], enc["attention_mask"], labels))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        x, m, y = self.data[i]
        return torch.tensor(x), torch.tensor(m), torch.tensor(y)

def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ms = torch.stack([b[1] for b in batch])
    ys = torch.stack([b[2] for b in batch])
    return xs, ms, ys

class BERTCRF(nn.Module):
    def __init__(self, ntag, name="bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(name)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, ntag)
        self.crf = CRF(ntag, batch_first=True)
    def forward(self, x, m):
        h = self.bert(x, attention_mask=m).last_hidden_state
        return self.fc(self.drop(h)), m.bool()
    def loss(self, logits, y, mask):
        return -self.crf(logits, y, mask=mask, reduction="mean")
    def decode(self, logits, mask):
        return self.crf.decode(logits, mask=mask)

def evaluate(model, loader, i2t, device):
    model.eval()
    golds, preds = [], []
    with torch.no_grad():
        for x, m, y in loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            logits, mask = model(x, m)
            out = model.decode(logits, mask)
            for i in range(x.size(0)):
                # mask 是连续前缀：[CLS] + 词... + [SEP] + [PAD]*，长度 L = mask.sum()
                # torchcrf 的 viterbi 输出长度同为 L，故下标与原序列对齐
                L = int(m[i].sum().item())
                idx = range(1, L - 1)          # 去掉首 [CLS] 与末 [SEP]
                golds.append([i2t[y[i, j].item()] for j in idx])
                preds.append([i2t[out[i][j]] for j in idx])
    return golds, preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    train_s = load_bio(os.path.join(NER, "train.bio"))
    dev_s   = load_bio(os.path.join(NER, "dev.bio"))
    test_s  = load_bio(os.path.join(NER, "test.bio"))
    tags = build_tags(train_s)
    t2i = {t: i for i, t in enumerate(tags)}
    i2t = {i: t for t, i in t2i.items()}
    ntag = len(tags)
    print(f"[tags] {ntag}: {tags}")

    tok = BertTokenizer.from_pretrained(args.model)
    train_ds = BERTNERDataset(train_s, tok, t2i)
    dev_ds   = BERTNERDataset(dev_s, tok, t2i)
    test_ds  = BERTNERDataset(test_s, tok, t2i)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dev_dl   = DataLoader(dev_ds, batch_size=args.batch, collate_fn=collate)
    test_dl  = DataLoader(test_ds, batch_size=args.batch, collate_fn=collate)

    # 自检：所有标签 id 必须落在 [0, ntag)，否则 CRF 会 CUDA assert 越界
    lab_max = max(max(d[2]) for d in train_ds.data)
    lab_min = min(min(d[2]) for d in train_ds.data)
    assert 0 <= lab_min and lab_max < ntag, \
        f"标签越界: min={lab_min}, max={lab_max}, ntag={ntag}（CRF 不接受 -100 等非法 id）"
    print(f"[check] 标签 id 范围 [{lab_min}, {lab_max}]，ntag={ntag} OK")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTCRF(ntag, args.model).to(device)
    print("device =", device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_f1, wait = 0.0, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, m, y in train_dl:
            x, m, y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            logits, mask = model(x, m)
            loss = model.loss(logits, y, mask)
            loss.backward(); opt.step(); tot += loss.item()
        g, p = evaluate(model, dev_dl, i2t, device)
        f1 = seq_f1(p, g)
        print(f"epoch {ep:02d}  loss={tot/len(train_dl):.3f}  dev_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; wait = 0
            torch.save(model.state_dict(), os.path.join(NER, "bert_crf.pt"))
        else:
            wait += 1
            if wait >= 3:
                print("early stop"); break

    model.load_state_dict(torch.load(os.path.join(NER, "bert_crf.pt"), map_location=device))
    g, p = evaluate(model, test_dl, i2t, device)
    rep = classification_report(g, p, output_dict=True, zero_division=0)
    overall = seq_f1(p, g)
    metrics = {
        "model": "BERT-CRF",
        "pretrained": args.model,
        "test_macro_f1": overall,
        "per_type": {k: {"precision": v.get("precision"), "recall": v.get("recall"), "f1": v.get("f1-score")}
                     for k, v in rep.items() if k not in ("macro avg", "weighted avg", "micro avg")},
        "hyperparams": vars(args),
    }
    with open(os.path.join(NER, "metrics_bert.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n===== BERT-CRF 测试结果 =====")
    print(classification_report(g, p, zero_division=0))
    print("已保存 ner/metrics_bert.json")

if __name__ == "__main__":
    main()
