# -*- coding: utf-8 -*-
"""
实验一 · 数据预处理阶段
把原始数据目录（由环境变量 CITYDATA_DIR 指定）下的分城市爬虫 CSV 清洗成知识图谱可用的结构化数据。

输出（默认写到本脚本同级的 cleaned/ 目录）:
  - scenic.csv   景点实体表（一行一个景点）
  - city.csv     城市实体表
  - season.csv   季节实体表
  - category.csv 景点类别实体表
  - relation_scenic_season.csv  景点 -> 季节 关系表
  - relation_scenic_category.csv 景点 -> 类别 关系表
  - preprocess_report.txt       预处理统计报告（可直接粘进实验报告）
"""

import os
import re
import ast
import csv
import glob
import sys

# ============ 配置区（按需修改）============
SRC_DIR = os.environ.get(
    "CITYDATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw"))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned")
INTRO_MAX = 300          # 简介保留的最大长度
MIN_NAME_LEN = 2         # 景点名最短长度，过滤脏数据
# ==========================================

# 类别规则：关键词 -> 类别名（顺序敏感，从上往下匹配）
CATEGORY_RULES = [
    (("博物馆", "纪念馆", "故居", "展览馆", "美术馆", "陈列馆", "科技馆", "图书馆"), "人文场馆"),
    (("古镇", "古城", "古村", "村落", "老街", "古街", "历史街区"), "古村镇"),
    (("寺", "庙", "观", "庵", "教堂", "清真寺", "塔"), "宗教建筑"),
    (("山", "峰", "岭", "峡谷", "谷", "崖", "岩", "洞", "森林", "林场", "草原", "沙漠"), "自然山地"),
    (("湖", "海", "江", "河", "溪", "泉", "瀑", "潭", "湾", "水库", "湿地"), "水域景观"),
    (("公园", "园林", "植物园", "动物园", "花卉", "庄园", "农庄"), "公园园林"),
    (("岛", "半岛", "群岛"), "岛屿"),
    (("游乐园", "度假区", "欢乐世界", "水上乐园", "主题公园"), "主题乐园"),
    (("遗址", "陵园", "陵", "墓", "城墙", "关", "长城"), "历史遗迹"),
    (("步道", "绿道", "骑行", "露营", "温泉", "滑雪"), "户外体验"),
]


def norm_text(v):
    """把单元格里成段的 \\n 和多余空白压平"""
    if v is None:
        return ""
    s = str(v)
    if s.lower() == "nan":
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{2,}", "\n", s)
    s = re.sub(r"[ \t\u3000]+", " ", s)
    return s.strip()


def parse_address_block(raw):
    """从地址单元格中拆出 地址 / 电话 / 官网"""
    s = norm_text(raw)
    result = {"address": "", "phone": "", "website": ""}
    if not s:
        return result

    # 按 “键:” 切段，键可能是 地址 / 电话 / 官网
    parts = re.split(r"\n*(地址|电话|官网|网址|开放时间)\s*[:：]\s*", s)
    if len(parts) >= 3:
        # parts 形如 ['', '地址', 'xxx', '电话', 'yyy', '官网', 'zzz']
        for i in range(1, len(parts) - 1, 2):
            key, val = parts[i], norm_text(parts[i + 1])
            if key == "地址":
                result["address"] = val
            elif key == "电话":
                result["phone"] = val
            elif key in ("官网", "网址"):
                result["website"] = val
    else:
        result["address"] = s
    return result


def parse_score(raw):
    """评分 -> float，异常返回空"""
    s = norm_text(raw)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else ""


def parse_duration(raw):
    """建议游玩时间 -> 规范化文本"""
    s = norm_text(raw).replace("建议游览时间：", "").replace("建议游玩时间：", "")
    s = s.replace("建议游览时间:", "").replace("建议游玩时间:", "")
    return s.strip("；;。 ")


def parse_ticket(raw):
    """门票字段是形如 {'成人票\\n\\n': ['¥397起']} 的字典字符串
    返回 (最低价 float, 票种数量 int, 票种名称用 | 连接)
    """
    s = norm_text(raw)
    if not s or not s.startswith("{"):
        return "", 0, ""

    try:
        data = ast.literal_eval(s)
        if not isinstance(data, dict):
            return "", 0, ""
    except Exception:
        return "", 0, ""

    prices, names = [], []
    for name, val in data.items():
        n = norm_text(name)
        if n:
            names.append(n)
        if isinstance(val, (list, tuple)):
            text = " ".join(str(x) for x in val)
        else:
            text = str(val)
        for m in re.finditer(r"(\d+(?:\.\d+)?)", text):
            prices.append(float(m.group(1)))

    min_price = min(prices) if prices else ""
    return min_price, len(names), "|".join(names[:5])


def parse_season(raw):
    """建议季节 -> 规范化季节（可能多个，用 | 分隔）"""
    s = norm_text(raw)
    if not s:
        return []
    if "四季皆宜" in s or "全年" in s:
        return ["四季皆宜"]

    found = [w for w in ("春", "夏", "秋", "冬") if w in s]
    if len(found) >= 4:
        return ["四季皆宜"]
    # “春秋最佳” 这类写法
    return found if found else []


def guess_category(name):
    for keys, label in CATEGORY_RULES:
        for k in keys:
            if k in name:
                return label
    return "其他"


def extract_district(address, city):
    """从地址中抽取区/县/旗。
    逐个定位 “区/县/旗”，向前取 4/3/2 个字，若包含 市/省/自治 等上级行政区划用字则换更短的长度，
    直到取到干净的区县名（如 茄子河区、浦东新区）。
    """
    if not address:
        return ""
    for m in re.finditer(r"[区县旗]", address):
        i = m.start()
        for L in (4, 3, 2):
            if i - L < 0:
                continue
            cand = address[i - L:i + 1]
            body = cand[:-1]
            if any(ch in body for ch in ("市", "省", "自治", "盟", "地区")):
                continue
            if cand == city:
                continue
            return cand
    return ""


def clean_name(name):
    """去掉名字里的空白和引号"""
    return norm_text(name).replace('"', "").replace("'", "").strip()


def split_name(name):
    """把 “上海迪士尼度假区Shanghai Disney Resort” 拆成中文名 + 英文名。
    规则：从第一个英文字母处切开，前面是中文名，后面是英文名。
    纯中文名则英文名留空。
    """
    m = re.match(r"^([\u4e00-\u9fa5\u3000-\u303f·、（）()\-—\d\s]*?)\s*([A-Za-z].*)$", name)
    if not m:
        return name, ""
    zh, en = m.group(1).strip(), m.group(2).strip()
    if not zh:                       # 纯英文名，整体作为英文名
        return name, name
    # 英文部分若仍夹杂中文（如 “艾肯（I CAN）梦想城”），说明英文名嵌在中间，不做拆分
    if re.search(r"[\u4e00-\u9fa5]", en):
        return name, ""
    return zh, en


def main():
    src_dir = SRC_DIR
    if not os.path.isdir(src_dir):
        print("找不到数据目录: %s" % src_dir)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.csv")))
    print("发现城市文件 %d 个" % len(files))

    scenics = []
    seen = set()              # (城市, 景点名) 去重
    raw_rows = 0
    dropped = 0
    city_counter = {}

    for fp in files:
        city = os.path.splitext(os.path.basename(fp))[0].strip()
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows += 1
                name = clean_name(row.get("名字"))
                if len(name) < MIN_NAME_LEN:
                    dropped += 1
                    continue

                key = (city, name)
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)

                addr_info = parse_address_block(row.get("地址"))
                min_price, ticket_cnt, ticket_names = parse_ticket(row.get("门票"))
                seasons = parse_season(row.get("建议季节"))
                intro = norm_text(row.get("介绍")).replace("\n", " ")
                if len(intro) > INTRO_MAX:
                    intro = intro[:INTRO_MAX] + "…"

                name_zh, name_en = split_name(name)
                scenics.append({
                    "name": name,
                    "name_zh": name_zh,
                    "name_en": name_en,
                    "city": city,
                    "address": addr_info["address"],
                    "district": extract_district(addr_info["address"], city),
                    "phone": addr_info["phone"],
                    "website": addr_info["website"],
                    "score": parse_score(row.get("评分")),
                    "open_time": norm_text(row.get("开放时间")),
                    "ticket_min_price": min_price,
                    "ticket_type_count": ticket_cnt,
                    "ticket_names": ticket_names,
                    "suggest_duration": parse_duration(row.get("建议游玩时间")),
                    "best_season_raw": norm_text(row.get("建议季节"))[:80],
                    "category": guess_category(name_zh),
                    "intro": intro,
                    "url": norm_text(row.get("链接")),
                    "image": norm_text(row.get("图片链接")),
                    "source_page": norm_text(row.get("Page")),
                })
                city_counter[city] = city_counter.get(city, 0) + 1

    print("原始记录 %d 条 -> 清洗后 %d 条（去重/过滤 %d 条）" % (raw_rows, len(scenics), dropped))

    # ---------- 写景点表 ----------
    scenic_path = os.path.join(OUT_DIR, "scenic.csv")
    with open(scenic_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scenics[0].keys()))
        w.writeheader()
        w.writerows(scenics)
    print("已写出: %s" % scenic_path)

    # ---------- 写城市表 ----------
    city_path = os.path.join(OUT_DIR, "city.csv")
    with open(city_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "spot_count"])
        for c, n in sorted(city_counter.items(), key=lambda x: -x[1]):
            w.writerow([c, n])
    print("已写出: %s（%d 个城市）" % (city_path, len(city_counter)))

    # ---------- 季节表 + 关系 ----------
    season_set = {}
    season_rel = []
    for idx, s in enumerate(scenics):
        seas = parse_season(s["best_season_raw"])
        for sea in seas:
            season_set.setdefault(sea, 0)
            season_set[sea] += 1
            season_rel.append({"scenic_name": s["name"], "city": s["city"], "season": sea})

    season_path = os.path.join(OUT_DIR, "season.csv")
    with open(season_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "scenic_count"])
        for k, v in sorted(season_set.items(), key=lambda x: -x[1]):
            w.writerow([k, v])
    print("已写出: %s" % season_path)

    rel_season_path = os.path.join(OUT_DIR, "relation_scenic_season.csv")
    with open(rel_season_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenic_name", "city", "season"])
        w.writeheader()
        w.writerows(season_rel)
    print("已写出: %s" % rel_season_path)

    # ---------- 类别表 + 关系 ----------
    cat_set = {}
    cat_rel = []
    for s in scenics:
        cat_set[s["category"]] = cat_set.get(s["category"], 0) + 1
        cat_rel.append({"scenic_name": s["name"], "city": s["city"], "category": s["category"]})

    cat_path = os.path.join(OUT_DIR, "category.csv")
    with open(cat_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "scenic_count"])
        for k, v in sorted(cat_set.items(), key=lambda x: -x[1]):
            w.writerow([k, v])
    print("已写出: %s" % cat_path)

    rel_cat_path = os.path.join(OUT_DIR, "relation_scenic_category.csv")
    with open(rel_cat_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenic_name", "city", "category"])
        w.writeheader()
        w.writerows(cat_rel)
    print("已写出: %s" % rel_cat_path)

    # ---------- 统计报告 ----------
    scores = [float(s["score"]) for s in scenics if s["score"] != ""]
    prices = [float(s["ticket_min_price"]) for s in scenics if s["ticket_min_price"] != ""]
    top_cities = sorted(city_counter.items(), key=lambda x: -x[1])[:10]

    lines = []
    lines.append("城市景点知识图谱 —— 数据预处理报告")
    lines.append("=" * 46)
    lines.append("数据来源   : %s" % src_dir)
    lines.append("城市文件数 : %d" % len(files))
    lines.append("原始记录数 : %d" % raw_rows)
    lines.append("清洗后记录 : %d" % len(scenics))
    lines.append("去重/过滤  : %d" % dropped)
    lines.append("城市实体数 : %d" % len(city_counter))
    lines.append("季节实体数 : %d" % len(season_set))
    lines.append("类别实体数 : %d" % len(cat_set))
    if scores:
        lines.append("评分覆盖   : %d 条，均值 %.2f，最高 %.1f" % (len(scores), sum(scores) / len(scores), max(scores)))
    if prices:
        lines.append("票价覆盖   : %d 条，最低价均值 %.1f 元" % (len(prices), sum(prices) / len(prices)))
    lines.append("")
    lines.append("景点数量 Top10 城市:")
    for c, n in top_cities:
        lines.append("  %-8s %d 个景点" % (c, n))
    lines.append("")
    lines.append("景点类别分布:")
    for k, v in sorted(cat_set.items(), key=lambda x: -x[1]):
        lines.append("  %-8s %d" % (k, v))

    report_path = os.path.join(OUT_DIR, "preprocess_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\n已写出: %s" % report_path)


if __name__ == "__main__":
    main()
