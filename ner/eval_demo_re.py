# -*- coding: utf-8 -*-
"""BERT-RE 关系抽取的“评估 + 预测演示”，用于实验报告截图。

跑法：python eval_demo_re.py
输出：测试集逐类 P/R/F1 + 若干实例的实体标记与预测对照。
"""
import os
import json
import re
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
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report

NER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NER)
from train_re import build_marked  # noqa: E402  训练/推理共用同一份标记实现

LABELS = ["no_relation", "located_in", "built_in", "related_person", "managed_by"]
L2I = {l: i for i, l in enumerate(LABELS)}


def mark(it):
    return build_marked(it["text"], it, it["e2"], it["e2_start"])


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备 = %s" % dev)
    tok = BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
    tok.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
    items = [json.loads(l) for l in open(os.path.join(NER, "rel_test.jsonl"), encoding="utf-8")]
    print("测试实例: %d 条" % len(items))

    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=len(LABELS), local_files_only=True)
    model.resize_token_embeddings(len(tok))
    model.load_state_dict(torch.load(os.path.join(NER, "bert_re.pt"), map_location=dev))
    model.to(dev)
    model.eval()
    print("加载模型 ner/bert_re.pt   关系类别: %s" % ", ".join(LABELS))

    # 批量评估
    preds, golds, confs = [], [], []
    B = 64
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        enc = tok([mark(it) for it in chunk], max_length=160,
                  truncation=True, padding="max_length", return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**enc).logits
        pr = torch.softmax(logits, -1).cpu()
        mx, idx = pr.max(-1)
        preds += idx.tolist()
        confs += mx.tolist()
        golds += [L2I[it["relation"]] for it in chunk]
        if (i // B) % 40 == 0:
            print("  已推理 %d / %d" % (min(i + B, len(items)), len(items)))

    print("\n===== 测试集评估 =====")
    print(classification_report(golds, preds, labels=list(range(len(LABELS))),
                                target_names=LABELS, digits=4, zero_division=0))

    print("\n===== 预测演示 =====")
    want = ["located_in", "built_in", "related_person", "managed_by"]
    shown = set()
    for it, p, c in zip(items, preds, confs):
        if it["relation"] not in want or it["relation"] in shown:
            continue
        t = re.sub(r"\s+", "", it["text"])
        if len(it["e1"]) < 2 or len(it["e2"]) < 2 or not (20 <= len(t) <= 90):
            continue
        shown.add(it["relation"])
        print("\n【%s】句子：%s" % (it["relation"], t[:78] + ("…" if len(t) > 78 else "")))
        print("  实体对：%s（头，SCENIC） → %s（尾）" % (it["e1"], it["e2"]))
        print("  标注关系：%-14s 预测关系：%-14s 置信度 %.3f"
              % (it["relation"], LABELS[p], c))
        if len(shown) == len(want):
            break


if __name__ == "__main__":
    main()
