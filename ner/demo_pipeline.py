# -*- coding: utf-8 -*-
"""
端到端样例：NER -> RE -> Neo4j
================================
把实验二（BERT-CRF 实体抽取）与实验三（BERT-RE 关系抽取）串成一条真实链路：
  1. 取一段景点介绍文本；
  2. 用 bert_crf.pt 抽取实体（SCENIC/LOC/TIME/PER/ORG）；
  3. 对 (SCENIC, 其他实体) 实体对用 bert_re.pt 判定关系；
  4. 把抽取出的"实体—关系—实体"三元组写入 Neo4j（独立子图，不污染主图谱）。

作用（同时满足两处要求）：
  - 实验三 要求#2「基于实体抽取结果实现关系抽取」——这里 RE 的输入正是 NER 的预测输出；
  - 实验四 要求#1「将模型中抽取出的实体和关系保存到 neo4j 数据库」。

运行（在本地 ner 环境、Neo4j 已启动的前提下）：
    python demo_pipeline.py                 # 抽取 + 入库
    python demo_pipeline.py --no-db         # 只抽取并打印 Cypher，不连库
    python demo_pipeline.py --text "自定义介绍文本"

说明：脚本会优先从 cleaned/scenic.csv 读取一条真实简介作为样例；也可 --text 指定。
      DB 不可达时自动跳过入库并打印提示，不影响抽取演示。
"""
import os, sys, json, argparse, csv, hashlib
# 本地离线加载权重，并关掉 transformers 的加载进度条/警告，保持运行输出干净
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import BertTokenizer, BertForSequenceClassification
try:
    from transformers.utils import logging as _hflog
    _hflog.disable_progress_bar()
    _hflog.set_verbosity_error()
except Exception:
    pass

# 复用实验二 / 实验三的模型定义与标签
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bert import BERTCRF, load_bio, build_tags, MAXLEN as NER_MAXLEN
from train_re import LABELS, MAXLEN as RE_MAXLEN

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NER  = os.path.join(BASE, "ner")
CLEANED = os.path.join(BASE, "cleaned")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")

REL2TYPE = {  # 关系 -> Neo4j 关系类型
    "located_in": "LOCATED_IN",
    "built_in": "BUILT_IN",
    "related_person": "RELATED_PERSON",
    "managed_by": "MANAGED_BY",
}


def load_sample_intro(ner_model, tok_n, t2i, i2t, device, fallback):
    """从真实数据挑一条"简介里包含景点名"的样本（这类才有 SCENIC 可抽），
    并用 NER 实测确认能抽出实体；否则回退到内置代表性示例。"""
    import random
    path = os.path.join(CLEANED, "scenic.csv")
    cands = []
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                intro = (r.get("intro") or "").strip()
                name = (r.get("name") or "").strip()
                # 只保留简介中出现景点名、且长度适中的样本
                if name and 40 <= len(intro) <= 200 and name in intro:
                    cands.append((name, intro))
    random.seed(0)
    random.shuffle(cands)
    primary = None          # 退而求其次：只有 SCENIC 的样本
    # 优先选"有 SCENIC 且另有 LOC/TIME/PER/ORG"的样本（这样 RE 才有关系可抽）
    for name, intro in cands[:250]:
        ents = ner_predict(intro, ner_model, tok_n, t2i, i2t, device)
        labels = {e["label"] for e in ents}
        if "SCENIC" in labels and (labels & {"LOC", "TIME", "PER", "ORG"}):
            return name, intro, ents
        if primary is None and "SCENIC" in labels:
            primary = (name, intro, ents)
    if primary is not None:
        return primary
    # 实在没有，用内置示例兜底（含 SCENIC/LOC/TIME/PER/ORG）
    fb = ("故宫又名紫禁城，位于北京市中心，是明清两代的皇家宫殿，"
          "由明成祖朱棣下令修建，现由故宫博物院管理。")
    return "（内置示例）", fb, None


# ---------- 实体文本清洗（剥掉 NER 误吞的前后噪声） ----------
# BERT-CRF 有时会把实体前面的方位/动词（如"内有""位于""在"）或后面的"的""等"
# 一起标进实体（例如把"内有陕西回族大寺"整段标成 ORG）。这些不是实体本身，
# 在入库/展示前剥掉。只剥"两端"，不碰实体中间的字符，避免误伤（如"上海"不会被剥）。
_LEAD_NOISE = ("内有", "位于", "坐落在", "坐落", "包括", "附近", "旁边",
               "有", "在", "于", "与", "和", "及", "的")
_TRAIL_NOISE = ("的", "等", "附近", "一带", "及", "和", "与")


def _clean_entity(raw):
    t = raw.strip()
    lead = 0
    changed = True
    while changed:                       # 反复剥，直到两端都不是噪声
        changed = False
        for p in sorted(_LEAD_NOISE, key=len, reverse=True):   # 优先剥更长的（"位于">"在"）
            if t.startswith(p) and len(t) >= len(p):
                t = t[len(p):]
                lead += len(p)
                changed = True
                break
    trail = 0
    for p in sorted(_TRAIL_NOISE, key=len, reverse=True):
        while t.endswith(p) and len(t) >= len(p):
            t = t[:len(t) - len(p)]
            trail += len(p)
    return t, lead, trail


# ---------- 实体抽取（复用 BERT-CRF） ----------
def ner_predict(text, model, tokenizer, t2i, i2t, device):
    chars = list(text)
    if len(chars) > NER_MAXLEN - 2:
        chars = chars[: NER_MAXLEN - 2]
    enc = tokenizer(chars, is_split_into_words=True, max_length=NER_MAXLEN,
                    truncation=True, padding="max_length", return_tensors="pt")
    wids = enc.word_ids()
    x = enc["input_ids"].to(device)
    m = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits, mask = model(x, m)
        out = model.decode(logits, mask)[0]          # 长度 L 的标签 id 序列
    L = int(m[0].sum().item())
    word_tags = {}
    for p in range(1, L - 1):                         # 去掉 [CLS] 与 [SEP]
        wi = wids[p]
        if wi is not None:
            word_tags[wi] = i2t[out[p]]
    seq = [word_tags.get(i, "O") for i in range(len(chars))]
    # BIO -> 实体 span
    entities, i = [], 0
    while i < len(seq):
        if seq[i].startswith("B-"):
            label = seq[i][2:]
            j = i
            while j + 1 < len(seq) and seq[j + 1] == "I-" + label:
                j += 1
            raw = "".join(chars[i:j + 1])
            # 清洗前后噪声（如"内有"），并相应平移起止偏移，保证后续 RE 标注的是干净实体
            clean, lead, trail = _clean_entity(raw)
            if clean:
                entities.append({"text": clean, "label": label,
                                 "start": i + lead, "end": j + 1 - trail})
            i = j + 1
        else:
            i += 1
    return entities


# ---------- 关系分类（复用 BERT-RE） ----------
def re_predict(text, subj, obj, model, tokenizer, device):
    s, e = subj["start"], subj["end"]
    marked = text[:s] + "<e1>" + text[s:e] + "</e1>" + text[e:]
    shift = len("<e1></e1>")
    o_s, o_e = obj["start"] + shift, obj["end"] + shift
    marked = marked[:o_s] + "<e2>" + marked[o_s:o_e] + "</e2>" + marked[o_e:]
    enc = tokenizer(marked, max_length=RE_MAXLEN, truncation=True,
                    padding="max_length", return_tensors="pt")
    x = enc["input_ids"].to(device)
    m = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=x, attention_mask=m).logits
        pred = int(logits.argmax(1).item())
    return LABELS[pred]


# ---------- 入库 ----------
def insert_triples(g, triples, source):
    """用显式事务写入并立即提交，避免 py2neo 在 Neo4j 5 下自动提交不可靠。"""
    tx = g.begin()
    for a, rel, b in triples:
        tx.run(
            "MERGE (a:NEREntity {text:$at, label:$al}) "
            "MERGE (b:NEREntity {text:$bt, label:$bl}) "
            "MERGE (a)-[:%s {source:$src}]->(b)" % REL2TYPE[rel],
            at=a["text"], al=a["label"], bt=b["text"], bl=b["label"], src=source,
        )
    # 锚定来源文档节点，便于在图谱中定位本子图。
    # 注意：source 属性挂在"关系"上而不是"实体"上，所以这里要经关系反查实体，
    # 不能写 MATCH (e:NEREntity {source:$src})（那样永远匹配不到）。
    if triples:
        tx.run(
            "MERGE (d:Doc {name:$src}) "
            "WITH d "
            "MATCH (a:NEREntity)-[r]->(b:NEREntity) WHERE r.source = $src "
            "WITH d, collect(DISTINCT a) + collect(DISTINCT b) AS ents "
            "UNWIND ents AS e "
            "MERGE (d)-[:CONTAINS]->(e)",
            src=source,
        )
    g.commit(tx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="自定义介绍文本；缺省从数据集取一条")
    ap.add_argument("--no-db", action="store_true", help="只抽取并打印 Cypher，不连库")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # ---- 载入 NER 模型 ----
    sents = load_bio(os.path.join(NER, "train.bio"))
    tags = build_tags(sents)
    t2i = {t: i for i, t in enumerate(tags)}
    i2t = {i: t for t, i in t2i.items()}
    tok_n = BertTokenizer.from_pretrained("bert-base-chinese")
    ner_model = BERTCRF(len(tags)).to(device)
    ner_model.load_state_dict(torch.load(os.path.join(NER, "bert_crf.pt"), map_location=device))
    ner_model.eval()
    print("NER 模型已加载：%d 类标签" % len(tags))

    # ---- 载入 RE 模型 ----
    tok_r = BertTokenizer.from_pretrained("bert-base-chinese")
    tok_r.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
    re_model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese", num_labels=len(LABELS))
    re_model.resize_token_embeddings(len(tok_r))
    re_model.load_state_dict(torch.load(os.path.join(NER, "bert_re.pt"), map_location=device))
    re_model.to(device)
    re_model.eval()
    print("RE 模型已加载：%d 类关系 %s" % (len(LABELS), LABELS))

    # ---- 取样例文本（优先挑能抽出实体的真实样本） ----
    name, text, _ = load_sample_intro(ner_model, tok_n, t2i, i2t, device, None)
    if args.text:
        text = args.text
        name = "（自定义文本）"
    print("\n===== 样例文本（来源：%s）=====\n%s\n" % (name, text))

    # ---- 1) 实体抽取 ----
    entities = ner_predict(text, ner_model, tok_n, t2i, i2t, device)
    print("【实体抽取结果】")
    for e in entities:
        print("   %-10s %s" % (e["label"], e["text"]))
    if not entities:
        print("   （未抽取到实体，样例文本可能与训练分布差异较大，可换 --text 重试）")
        return

    # ---- 2) 关系分类（SCENIC 作主语，与其余实体配对） ----
    scenics = [e for e in entities if e["label"] == "SCENIC"]
    subj = scenics[0] if scenics else entities[0]
    others = [e for e in entities if e is not subj]
    triples = []
    print("\n【关系抽取结果】（主语=%s）" % subj["text"])
    for o in others:
        rel = re_predict(text, subj, o, re_model, tok_r, device)
        print("   (%s) —[%s]—> (%s)" % (subj["text"], rel, o["text"]))
        if rel != "no_relation":
            triples.append((subj, rel, o))

    # ---- 3) 打印 Cypher ----
    print("\n【将写入 Neo4j 的 Cypher】")
    for a, rel, b in triples:
        print('MERGE (a:NEREntity {text:"%s", label:"%s"})'
              ' MERGE (b:NEREntity {text:"%s", label:"%s"})'
              ' MERGE (a)-[:%s]->(b);' % (a["text"], a["label"], b["text"], b["label"], REL2TYPE[rel]))

    # ---- 4) 入库 ----
    if args.no_db:
        print("\n（--no-db：已跳过入库）")
        return
    try:
        from py2neo import Graph
        g = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name=NEO4J_DB)
        g.run("RETURN 1").data()
        print("\n[Neo4j] 已连接，目标库 NEO4J_DB='%s'（数据即写入此库）" % NEO4J_DB)
        # 用 md5 生成确定性短 id：Python 的 hash() 每个进程随机，会导致同一段文本
        # 每次运行得到的 id 都不同，无法在 Browser 里稳定复现；md5 则恒定不变。
        src = "demo_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        insert_triples(g, triples, src)
        # 回读校验：确认数据真正落盘
        cnt_e = g.run("MATCH (e:NEREntity) RETURN count(e) AS c").data()[0]["c"]
        cnt_c = g.run("MATCH (:Doc)-[r:CONTAINS]->(:NEREntity) RETURN count(r) AS c").data()[0]["c"]
        print("\n已写入 Neo4j：Doc{name:'%s'}，本次 %d 条关系；库中 NEREntity=%d、CONTAINS=%d。"
              % (src, len(triples), cnt_e, cnt_c))
        print("\n在 Neo4j Browser 执行（推荐第一条，无需记 id）：")
        print("  MATCH (d:Doc)-[:CONTAINS]->(e) RETURN d, e;")
        if triples:
            print("  MATCH (d:Doc{name:'%s'})-[:CONTAINS]->(e) RETURN d, e;" % src)
    except Exception as ex:
        print("\n[警告] 未能写入 Neo4j：%s" % ex)
        print("请确保 Neo4j 已启动、并已设置环境变量 NEO4J_PASSWORD，再重跑本脚本（去掉 --no-db）。")


if __name__ == "__main__":
    main()
