# -*- coding: utf-8 -*-
"""
关系抽取远程监督标注（pipeline 第 2 步）
依赖：实验二 NER 的实体定义；从景点介绍文本中，以 SCENIC 为主语，
自动生成 (SCENIC, 关系, 对象) 三元组实例，供 BERT-RE 训练。

关系类型：
  located_in     SCENIC -> LOC       （景点位于某地，高精度）
  built_in       SCENIC -> TIME      （景点建于某朝代/年代，中精度）
  related_person SCENIC -> PER       （景点与某历史人物相关，中精度/噪声）
  managed_by     SCENIC -> ORG       （景点由某机构管理/所属，中精度）
  no_relation    SCENIC -> *         （负例，跨文档采样）

输出：
  ner/rel_train.jsonl  ner/rel_dev.jsonl  ner/rel_test.jsonl
  ner/rel_labeling_report.txt  ner/rel_stats.json
"""
import os, re, json, random
from collections import Counter
import pandas as pd

random.seed(42)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(BASE, "cleaned", "scenic.csv")
OUT  = os.path.join(BASE, "ner")
os.makedirs(OUT, exist_ok=True)

REL_MAP = {"LOC": "located_in", "TIME": "built_in", "PER": "related_person", "ORG": "managed_by"}
INV_REL = {v: k for k, v in REL_MAP.items()}

# ---- 复用实验二的正则与词典 ----
TIME_PATTERNS = [
    r'(公元)?\d{1,4}(余)?(年|世纪)', r'距今\d{1,4}年',
    r'侏罗纪|白垩纪|寒武纪|奥陶纪|志留纪|泥盆纪|石炭纪|二叠纪|三叠纪|第四纪|第三纪|新生代|中生代|古生代|太古代|元古代',
    r'新石器(时代)?|旧石器(时代)?|青铜器时代|铁器时代',
    r'(秦|汉|唐|宋|元|明|清|隋|晋|魏|周|商|春秋|战国|南北朝|五代|西夏|辽|金|民国|中华人民共和国|夏)(代|朝|国|时期)?',
    r'(乾隆|康熙|雍正|贞观|开元|天宝|咸丰|道光|光绪|宣统|永乐|洪武|景泰|成化|嘉庆|同治|顺治|建炎|绍兴|大观|政和|至和)',
]
TIME_RE = [re.compile(p) for p in TIME_PATTERNS]
ORG_RE = re.compile(r'([一-龥]{2,12})(寺|庙|宫|观|庵|书院|教堂|清真寺|博物馆|纪念馆|陵园|道观|府衙|衙门|驿站|军用机场|机场|火车站|港口|学堂|学院|博物院|禅寺|古寺|大教堂)')
PER_LIST = ['李白','杜甫','苏轼','苏辙','苏洵','王维','白居易','杜牧','李商隐','孟浩然','王勃','骆宾王','陈子昂','韩愈','柳宗元','刘禹锡','李贺','温庭筠','韦应物','元稹','岑参','高适','王昌龄','王之涣','贺知章','张九龄','王羲之','王献之','谢灵运','陶渊明','谢朓','曹操','曹植','刘备','孙权','诸葛亮','周瑜','司马懿','司马迁','司马光','屈原','宋玉','李清照','辛弃疾','陆游','文天祥','岳飞','卫青','霍去病','张骞','郑和','玄奘','鉴真','秦始皇','汉武帝','唐太宗','唐玄宗','乾隆','康熙','雍正','光绪','慈禧','项羽','刘邦','韩信','萧何','张良','班超','班固','蔡伦','祖冲之','张衡','李时珍','扁鹊','华佗','关羽','张飞','赵云','包拯','海瑞','于谦','袁崇焕','林则徐','刘伯温','纪晓岚']

MAX_POS, MAX_NEG = 6, 3


def extract_objects(text):
    """返回文本中所有非 SCENIC 实体 span: list of (start,end,type)"""
    spans = []
    for rx in TIME_RE:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), "TIME"))
    for m in ORG_RE.finditer(text):
        spans.append((m.start(), m.end(), "ORG"))
    for p in PER_LIST:
        for m in re.finditer(re.escape(p), text):
            spans.append((m.start(), m.end(), "PER"))
    return spans


def find_subject(text, name_zh, name):
    for s in [name_zh, name]:
        if s and len(str(s)) >= 2:
            i = text.find(str(s))
            if i >= 0:
                return i, i + len(str(s)), str(s)
    return None


def build_instances(df):
    rows = []
    global_pool = []  # (span_text, type) 用于负例
    for _, r in df.iterrows():
        text = str(r.get("intro", "")) if pd.notna(r.get("intro")) else ""
        if not text or len(text) < 4:
            continue
        name_zh = r.get("name_zh", "") or ""
        name = r.get("name", "") or ""
        city = r.get("city", "") or ""
        district = r.get("district", "") or ""
        subj = find_subject(text, name_zh, name)
        if not subj:
            continue
        s_start, s_end, s_text = subj
        # LOC 对象：城市/区县（与主语不同）
        locs = []
        for loc in [str(city), str(district)]:
            if loc and len(loc) >= 2 and loc != s_text:
                i = text.find(loc)
                if i >= 0:
                    locs.append((i, i + len(loc), "LOC", loc))
        objs = extract_objects(text)  # TIME/PER/ORG
        # 组装正例
        pos = []
        for (i, j, t, _txt) in locs:
            pos.append((i, j, t, text[i:j]))
        for (i, j, t) in objs:
            if i >= s_start and j <= s_end:
                continue  # 与主语重叠则跳过
            pos.append((i, j, t, text[i:j]))
        # 去重（按 span 文本+类型）
        seen, pos_u = set(), []
        for (i, j, t, txt) in pos:
            key = (txt, t)
            if key not in seen:
                seen.add(key); pos_u.append((i, j, t, txt))
        pos_u = pos_u[:MAX_POS]
        insts = []
        for (i, j, t, txt) in pos_u:
            insts.append({"text": text, "e1": s_text, "e1_start": s_start, "e1_end": s_end,
                          "e2": txt, "e2_start": i, "e2_end": j,
                          "e1_type": "SCENIC", "e2_type": t, "relation": REL_MAP[t]})
            global_pool.append((txt, t))
        # 负例：跨文档采样，对象不在本 intro
        neg_pool = [g for g in global_pool if g[0] not in text]
        random.shuffle(neg_pool)
        added = 0
        for (txt, t) in neg_pool:
            if added >= MAX_NEG:
                break
            k = text.find(txt)
            if k < 0:
                insts.append({"text": text, "e1": s_text, "e1_start": s_start, "e1_end": s_end,
                              "e2": txt, "e2_start": -1, "e2_end": -1,
                              "e1_type": "SCENIC", "e2_type": t, "relation": "no_relation"})
                added += 1
        rows.append(insts)
    return rows


def main():
    df = pd.read_csv(SRC)
    df = df[df["intro"].notna()].reset_index(drop=True)
    # 与实验二同种子、同文档级划分 70/15/15
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    n1 = int(n * 0.70); n2 = int(n * 0.15)
    splits = {"rel_train": df.iloc[:n1], "rel_dev": df.iloc[n1:n1 + n2], "rel_test": df.iloc[n1 + n2:]}

    rel_counter = Counter()
    stats = {}
    for name, part in splits.items():
        insts_all = build_instances(part)
        flat = [x for sub in insts_all for x in sub]
        with open(os.path.join(OUT, name + ".jsonl"), "w", encoding="utf-8") as f:
            for it in flat:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        for it in flat:
            rel_counter[it["relation"]] += 1
        stats[name] = len(flat)
        print(f"{name}: {len(flat)} 实例")

    stats["relation_counts"] = dict(rel_counter)
    stats["relations"] = ["located_in", "built_in", "related_person", "managed_by", "no_relation"]
    with open(os.path.join(OUT, "rel_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    lines = ["=== 关系抽取远程监督标注统计 ==="]
    lines.append(f"train/dev/test 实例数 = {stats['rel_train']} / {stats['rel_dev']} / {stats['rel_test']}")
    lines.append("关系类型分布：")
    for k, v in rel_counter.most_common():
        lines.append(f"  {k:14s} {v}")
    lines.append("")
    lines.append("说明：located_in 由 city/district 字段对齐，高精度；")
    lines.append("      built_in/related_person/managed_by 由正则/词典抽取，含噪声；")
    lines.append("      no_relation 为负例（跨文档采样）。建议人工抽检 PER/ORG 相关三元组。")
    with open(os.path.join(OUT, "rel_labeling_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
