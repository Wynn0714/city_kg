# -*- coding: utf-8 -*-
"""BERT-CRF 实体识别的“评估 + 预测演示”，用于实验报告截图。

跑法：python eval_demo.py
输出：测试集逐类 P/R/F1（seqeval）+ 若干句的预测与标注对照。
"""
import os
import sys
# 离线加载本地缓存（本机走代理时 huggingface 会 502，直接禁用联网）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)
import warnings
warnings.filterwarnings("ignore")
try:
    from transformers.utils import logging as _hflog
    _hflog.disable_progress_bar()
    _hflog.set_verbosity_error()
except Exception:
    pass
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from seqeval.metrics import classification_report
from seqeval.scheme import Entities

NER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NER)
from train_bert import (load_bio, build_tags, BERTNERDataset, BERTCRF,  # noqa: E402
                        collate, evaluate, MAXLEN)


def spans2(chars, tags):
    """按字符还原实体（保留原词）。"""
    out, cur = [], None
    for ch, t in zip(chars, tags):
        if t.startswith("B-"):
            if cur:
                out.append(cur)
            cur = [t[2:], ch]
        elif t.startswith("I-") and cur:
            cur[1] += ch
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return [(a, b) for a, b in out]


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    train_s = load_bio(os.path.join(NER, "train.bio"))
    test_s = load_bio(os.path.join(NER, "test.bio"))
    tags = build_tags(train_s)
    t2i = {t: i for i, t in enumerate(tags)}
    i2t = {i: t for t, i in t2i.items()}

    tok = BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
    test_ds = BERTNERDataset(test_s, tok, t2i)
    test_dl = DataLoader(test_ds, batch_size=32, collate_fn=collate)

    device = torch.device("cpu")
    model = BERTCRF(len(tags)).to(device)
    model.load_state_dict(torch.load(os.path.join(NER, "bert_crf.pt"), map_location=device))
    model.eval()
    print("加载模型 ner/bert_crf.pt  标签数 %d  测试样本 %d" % (len(tags), len(test_s)))
    print("标签集: %s" % " ".join(tags))

    golds, preds = evaluate(model, test_dl, i2t, device)
    print("\n===== 测试集评估（seqeval 实体级）=====")
    print(classification_report(golds, preds, digits=4, zero_division=0))

    # 预测演示：取 3 个标注干净（实体长度≥2）的测试句，便于对照阅读
    print("\n===== 预测演示 =====")
    shown = 0
    for chars, tg in test_s:
        ents = spans2(chars, tg)
        if not (2 <= len(ents) <= 6) or any(len(b) < 2 for _, b in ents):
            continue
        if not (20 <= len(chars) <= 92):
            continue
        types = set(a for a, _ in ents)
        if len(types) < 2:
            continue
        enc = tok(chars[:MAXLEN - 2], is_split_into_words=True, max_length=MAXLEN,
                  truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            logits, mask = model(enc["input_ids"], enc["attention_mask"])
            out = model.decode(logits, mask)[0]
        L = int(enc["attention_mask"][0].sum().item())
        ptags = [i2t[out[j]] for j in range(1, L - 1)]
        shown += 1
        sent = "".join(chars[:L - 2])
        if len(sent) > 96:
            sent = sent[:96] + "…"
        print("\n【句 %d】%s" % (shown, sent))
        print("  标注: %s" % "  ".join("%s[%s]" % (b, a) for a, b in ents))
        print("  预测: %s" % "  ".join("%s[%s]" % (b, a) for a, b in spans2(chars[:L - 2], ptags)))
        if shown == 1:
            n = min(len(chars), 34)
            print("  逐字对照（前 %d 字）:" % n)
            print("    字: %s" % " ".join(chars[:n]))
            print("    真: %s" % " ".join(tg[:n]))
            print("    预: %s" % " ".join(ptags[:n]))
        if shown >= 3:
            break


if __name__ == "__main__":
    main()
