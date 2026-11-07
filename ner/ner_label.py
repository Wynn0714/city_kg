# -*- coding: utf-8 -*-
"""
远程监督标注：从景点介绍文本中抽取 5 类实体（char 级 BIO）
- SCENIC 景点/子景点  : 由结构化字段 name_zh / name 自动对齐（高精度）
- LOC    地理位置      : 由 city / district 字段自动对齐（高精度）
- TIME   时间/朝代     : 朝代/年号/年代/地质年代正则（中精度）
- PER    人物          : 常见历史人物词典（低精度，需人工核对）
- ORG    机构          : 寺/庙/宫/观/博物馆... 后缀正则（中精度）

输出：
  ner/train.bio  ner/dev.bio  ner/test.bio   (CoNLL 格式: 字\\t标签)
  ner/label_stats.json
  ner/labeling_report.txt
"""
import os, re, json, random
from collections import Counter
import pandas as pd

random.seed(42)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(BASE, "cleaned", "scenic.csv")
OUT  = os.path.join(BASE, "ner")
os.makedirs(OUT, exist_ok=True)

# ---------------- 词典 / 正则 ----------------
TIME_PATTERNS = [
    r'(公元)?\d{1,4}(余)?(年|世纪)',
    r'距今\d{1,4}年',
    r'侏罗纪|白垩纪|寒武纪|奥陶纪|志留纪|泥盆纪|石炭纪|二叠纪|三叠纪|第四纪|第三纪|新生代|中生代|古生代|太古代|元古代',
    r'新石器(时代)?|旧石器(时代)?|青铜器时代|铁器时代',
    r'(秦|汉|唐|宋|元|明|清|隋|晋|魏|周|商|春秋|战国|南北朝|五代|西夏|辽|金|民国|中华人民共和国|夏)(代|朝|国|时期)?',
    r'(乾隆|康熙|雍正|贞观|开元|天宝|咸丰|道光|光绪|宣统|永乐|洪武|景泰|成化|嘉庆|同治|顺治|建炎|绍兴|大观|政和|至和)',
]
TIME_RE = [re.compile(p) for p in TIME_PATTERNS]

ORG_RE = re.compile(
    r'([一-龥]{2,12})(寺|庙|宫|观|庵|书院|教堂|清真寺|博物馆|纪念馆|陵园|道观|府衙|衙门|驿站|军用机场|机场|火车站|港口|学堂|学院|博物院|禅寺|古寺|大教堂)'
)

PER_LIST = ['李白','杜甫','苏轼','苏辙','苏洵','王维','白居易','杜牧','李商隐','孟浩然','王勃','骆宾王','陈子昂','韩愈','柳宗元','刘禹锡','李贺','温庭筠','韦应物','元稹','岑参','高适','王昌龄','王之涣','贺知章','张九龄','王羲之','王献之','谢灵运','陶渊明','谢朓','曹操','曹植','刘备','孙权','诸葛亮','周瑜','司马懿','司马迁','司马光','屈原','宋玉','李清照','辛弃疾','陆游','文天祥','岳飞','卫青','霍去病','张骞','郑和','玄奘','鉴真','秦始皇','汉武帝','唐太宗','唐玄宗','乾隆','康熙','雍正','光绪','慈禧','项羽','刘邦','韩信','萧何','张良','班超','班固','蔡伦','祖冲之','张衡','李时珍','扁鹊','华佗','关羽','张飞','赵云','包拯','海瑞','于谦','袁崇焕','林则徐','刘伯温','纪晓岚']

# 优先级：数值越小越优先（用于重叠消解）
PRIORITY = {"SCENIC": 0, "LOC": 1, "ORG": 2, "PER": 3, "TIME": 4}


def find_spans(text, substr, label):
    spans = []
    if substr is None or (isinstance(substr, float) and pd.isna(substr)):
        return spans
    substr = str(substr)
    if not substr or len(substr) < 2:
        return spans
    for m in re.finditer(re.escape(substr), text):
        spans.append((m.start(), m.end(), label))
    return spans


def label_intro(row):
    text = str(row.get("intro", "")) if pd.notna(row.get("intro")) else ""
    if not text or len(text) < 4:
        return None
    spans = []
    name_zh = row.get("name_zh", "")
    name    = row.get("name", "")
    city    = row.get("city", "")
    district= row.get("district", "")

    # 高精度：结构化字段对齐
    for s in [name_zh, name]:
        if s and s != city:
            spans += find_spans(text, s, "SCENIC")
    if city:
        spans += find_spans(text, city, "LOC")
    if district and district != city:
        spans += find_spans(text, district, "LOC")

    # 中精度：机构后缀
    for m in ORG_RE.finditer(text):
        spans.append((m.start(), m.end(), "ORG"))
    # 中精度：时间/朝代
    for rx in TIME_RE:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), "TIME"))
    # 低精度：人物词典
    for p in PER_LIST:
        spans += find_spans(text, p, "PER")

    if not spans:
        return None

    # 重叠消解：按 start 升序、priority 升序贪心
    spans.sort(key=lambda x: (x[0], PRIORITY[x[2]]))
    tags = ["O"] * len(text)
    last_end = -1
    used = []
    for s, e, lab in spans:
        if s >= last_end:
            tags[s] = "B-" + lab
            for i in range(s + 1, e):
                tags[i] = "I-" + lab
            last_end = e
            used.append(lab)
    return list(text), tags, used


def main():
    df = pd.read_csv(SRC)
    # 只处理有介绍文本的景点
    df = df[df["intro"].notna()]
    samples, type_counter, doc_with_type = [], Counter(), Counter()
    dropped = 0
    for _, row in df.iterrows():
        res = label_intro(row)
        if res is None:
            dropped += 1
            continue
        chars, tags, used = res
        samples.append((chars, tags))
        for t in used:
            type_counter[t] += 1
        for t in set(used):
            doc_with_type[t] += 1

    # 划分 70/15/15
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.70)
    n_dev   = int(n * 0.15)
    splits = {"train": samples[:n_train],
              "dev":   samples[n_train:n_train + n_dev],
              "test":  samples[n_train + n_dev:]}

    def write_bio(name, data):
        path = os.path.join(OUT, name + ".bio")
        with open(path, "w", encoding="utf-8") as f:
            for chars, tags in data:
                for c, t in zip(chars, tags):
                    f.write(f"{c}\t{t}\n")
                f.write("\n")
        return len(data)

    stats = {}
    for name, data in splits.items():
        cnt = write_bio(name, data)
        stats[name] = cnt

    stats["total_samples"] = n
    stats["dropped_no_text"] = dropped
    stats["entity_counts"] = dict(type_counter)
    stats["docs_with_entity"] = dict(doc_with_type)
    stats["labels"] = ["SCENIC", "LOC", "TIME", "PER", "ORG"]

    with open(os.path.join(OUT, "label_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 文字报告
    lines = []
    lines.append("=== 远程监督标注统计 ===")
    lines.append(f"有效标注样本数 : {n}")
    lines.append(f"无介绍文本跳过 : {dropped}")
    lines.append(f"train / dev / test = {stats['train']} / {stats['dev']} / {stats['test']}")
    lines.append("")
    lines.append("各类实体出现次数（含重叠计数）与覆盖文档数：")
    for lab in ["SCENIC", "LOC", "TIME", "PER", "ORG"]:
        lines.append(f"  {lab:8s} 计数={type_counter.get(lab,0):6d}  覆盖文档={doc_with_type.get(lab,0):6d}")
    lines.append("")
    lines.append("说明：SCENIC/LOC 由结构化字段自动对齐，精度高；")
    lines.append("      TIME/ORG 由正则抽取，中精度；PER 由人物词典，低精度，建议人工抽检。")
    with open(os.path.join(OUT, "labeling_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
