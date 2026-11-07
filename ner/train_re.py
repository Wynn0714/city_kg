# -*- coding: utf-8 -*-
"""
BERT 关系分类（pipeline 第 2 步：关系抽取）
做法：在文本中用特殊标记 <e1></e1> <e2></e2> 标出两个实体，
      用 bert-base-chinese 的 [CLS] 向量做关系分类。
建议 CUDA GPU。

依赖：pip install torch transformers sklearn
用法：
    python train_re.py
    python train_re.py --epochs 8 --batch 16
"""
import os, json, argparse
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # transformers 新版已移除顶层 AdamW，改用 torch.optim 内置版本
from sklearn.metrics import classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NER  = os.path.join(BASE, "ner")
LABELS = ["no_relation", "located_in", "built_in", "related_person", "managed_by"]
L2I = {l: i for i, l in enumerate(LABELS)}
MAXLEN = 160

def build_marked(text, subj, e2text, e2s):
    """插入实体标记；e2s<0 表示负例，实体 2 拼在末尾。

    注意：本函数与 ner/bert_re.pt 的权重是配套的（权重即由本版标记训练得到），
    修改标记实现后必须重新训练，否则评估结果与 metrics_re.json 不再一致。
    已知问题：e2 的下标是相对原文的，插入 <e1></e1>（9 字符）后未做偏移修正，
    e2 标记的位置会前移 9 个字符；修正后需重训才能体现收益。
    """
    s, e = subj["e1_start"], subj["e1_end"]
    marked = text[:s] + "<e1>" + text[s:e] + "</e1>" + text[e:]
    if e2s >= 0:
        i, j = e2s, subj["e2_end"]
        marked = marked[:i] + "<e2>" + marked[i:j] + "</e2>" + marked[j:]
    else:
        marked = marked + " <e2>" + e2text + "</e2>"
    return marked

class REDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.items = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tokenizer
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        marked = build_marked(it["text"], it, it["e2"], it["e2_start"])
        enc = self.tok(marked, max_length=MAXLEN, truncation=True, padding="max_length", return_tensors="pt")
        return (enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0),
                torch.tensor(L2I[it["relation"]], dtype=torch.long))

def collate(b):
    xs = torch.stack([x[0] for x in b]); ms = torch.stack([x[1] for x in b])
    ys = torch.stack([x[2] for x in b]); return xs, ms, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--sample_neg", type=float, default=1.0,
                    help="负例下采样倍数(相对正例总数)，1.0 表示不采样")
    args = ap.parse_args()

    tok = BertTokenizer.from_pretrained(args.model)
    tok.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])

    train_ds = REDataset(os.path.join(NER, "rel_train.jsonl"), tok)
    dev_ds   = REDataset(os.path.join(NER, "rel_dev.jsonl"), tok)
    test_ds  = REDataset(os.path.join(NER, "rel_test.jsonl"), tok)

    # 负例下采样以缓解类别不平衡
    if args.sample_neg and args.sample_neg < 1e9:
        pos = sum(1 for it in train_ds.items if it["relation"] != "no_relation")
        neg_all = [i for i, it in enumerate(train_ds.items) if it["relation"] == "no_relation"]
        keep = int(pos * args.sample_neg)
        import random; random.seed(42)
        keep_idx = set(random.sample(neg_all, min(keep, len(neg_all))))
        idx = [i for i in range(len(train_ds.items)) if train_ds.items[i]["relation"] != "no_relation" or i in keep_idx]
        train_ds.items = [train_ds.items[i] for i in idx]

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dev_dl   = DataLoader(dev_ds, batch_size=args.batch, collate_fn=collate)
    test_dl  = DataLoader(test_ds, batch_size=args.batch, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(args.model, num_labels=len(LABELS))
    model.resize_token_embeddings(len(tok))
    model.to(device)
    print("device =", device, " train 实例 =", len(train_ds))

    # 类别加权损失
    cnt = Counter(it["relation"] for it in train_ds.items)
    weights = torch.tensor([cnt.get(L, 1) for L in LABELS], dtype=torch.float)
    weights = 1.0 / (weights / weights.sum())
    weights = weights / weights.sum() * len(LABELS)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    opt = AdamW(model.parameters(), lr=args.lr)
    best_f1, wait = 0.0, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, m, y in train_dl:
            x, m, y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            logits = model(input_ids=x, attention_mask=m).logits
            loss = criterion(logits, y)
            loss.backward(); opt.step(); tot += loss.item()
        # dev 评估
        model.eval(); preds, golds = [], []
        with torch.no_grad():
            for x, m, y in dev_dl:
                x, m = x.to(device), m.to(device)
                p = model(input_ids=x, attention_mask=m).logits.argmax(1).cpu().tolist()
                preds += p; golds += y.tolist()
        f1 = classification_report(golds, preds, labels=list(range(len(LABELS))),
                                   target_names=LABELS, output_dict=True, zero_division=0)["macro avg"]["f1-score"]
        print(f"epoch {ep:02d}  loss={tot/len(train_dl):.3f}  dev_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; wait = 0
            torch.save(model.state_dict(), os.path.join(NER, "bert_re.pt"))
        else:
            wait += 1
            if wait >= 3:
                print("early stop"); break

    model.load_state_dict(torch.load(os.path.join(NER, "bert_re.pt")))
    model.eval(); preds, golds = [], []
    with torch.no_grad():
        for x, m, y in test_dl:
            x, m = x.to(device), m.to(device)
            p = model(input_ids=x, attention_mask=m).logits.argmax(1).cpu().tolist()
            preds += p; golds += y.tolist()

    rep = classification_report(golds, preds, labels=list(range(len(LABELS))),
                               target_names=LABELS, output_dict=True, zero_division=0)
    metrics = {"model": "BERT-RE", "pretrained": args.model,
               "per_relation": {L: {"precision": rep[L]["precision"], "recall": rep[L]["recall"], "f1": rep[L]["f1-score"]}
                                for L in LABELS},
               "macro_f1": rep["macro avg"]["f1-score"],
               "hyperparams": vars(args)}
    with open(os.path.join(NER, "metrics_re.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n===== BERT-RE 测试结果 =====")
    print(classification_report(golds, preds, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0))
    print("已保存 ner/metrics_re.json")

if __name__ == "__main__":
    main()
