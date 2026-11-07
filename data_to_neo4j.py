# -*- coding: utf-8 -*-
"""
=============================================================
 data_to_neo4j.py
 用 py2neo 把预处理后的城市景点数据批量导入 Neo4j，构建领域知识图谱
=============================================================

使用前准备：
  1. 先在 Neo4j Browser 里建好并启动一个数据库；
  2. 通过环境变量 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD 配置连接；
  3. 先运行 preprocess_citydata.py 生成 cleaned/ 下的 CSV；
  4. 右击运行本文件。

图谱模式（Schema）：
  实体：
    (:City     {name, spot_count})                 城市
    (:Scenic   {sid, name, city, address, ...})    景点
    (:Season   {name, scenic_count})               适宜季节
    (:Category {name, scenic_count})               景点类别
  关系：
    (:City)   -[:HAS_SPOT]     -> (:Scenic)        城市拥有景点
    (:Scenic) -[:BEST_SEASON]  -> (:Season)        景点适宜季节
    (:Scenic) -[:BELONGS_TO]   -> (:Category)      景点所属类别

依赖：pip install py2neo
"""

import os
import sys
import csv
import time
import argparse

from py2neo import Graph

# ============ 连接配置：改成你自己的 ============
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
if not NEO4J_PASSWORD:
    raise SystemExit("请先设置环境变量 NEO4J_PASSWORD（Neo4j 密码），详见 README。")
# ==============================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned")
BATCH_SIZE = 1000


def to_float(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except ValueError:
        return None


def to_int(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return int(float(v))
    except ValueError:
        return None


def read_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print("缺少数据文件: %s\n请先运行 preprocess_citydata.py" % path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def create_constraints(g):
    """建立唯一性约束，既保证实体不重复，又大幅加快 MERGE 速度"""
    stmts = [
        "CREATE CONSTRAINT city_name IF NOT EXISTS "
        "FOR (n:City) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT scenic_sid IF NOT EXISTS "
        "FOR (n:Scenic) REQUIRE n.sid IS UNIQUE",
        "CREATE CONSTRAINT season_name IF NOT EXISTS "
        "FOR (n:Season) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT category_name IF NOT EXISTS "
        "FOR (n:Category) REQUIRE n.name IS UNIQUE",
    ]
    for s in stmts:
        g.run(s)
    print("[1/6] 约束创建完成")


def clear_graph(g):
    print("      清空旧数据...")
    g.run("MATCH (n) DETACH DELETE n")
    print("      旧数据已清空")


def import_cities(g, rows):
    for i, batch in enumerate(chunks(rows, BATCH_SIZE), 1):
        g.run(
            "UNWIND $rows AS row "
            "MERGE (c:City {name: row.name}) "
            "SET c.spot_count = row.spot_count",
            rows=[{"name": r["name"], "spot_count": to_int(r["spot_count"])} for r in batch],
        )
        print("      城市 %d/%d" % (min(i * BATCH_SIZE, len(rows)), len(rows)), end="\r")
    print("[2/6] 城市实体导入完成：%d 个" % len(rows))


def import_seasons(g, rows):
    for batch in chunks(rows, BATCH_SIZE):
        g.run(
            "UNWIND $rows AS row "
            "MERGE (s:Season {name: row.name}) "
            "SET s.scenic_count = row.scenic_count",
            rows=[{"name": r["name"], "scenic_count": to_int(r["scenic_count"])} for r in batch],
        )
    print("[3/6] 季节实体导入完成：%d 个" % len(rows))


def import_categories(g, rows):
    for batch in chunks(rows, BATCH_SIZE):
        g.run(
            "UNWIND $rows AS row "
            "MERGE (c:Category {name: row.name}) "
            "SET c.scenic_count = row.scenic_count",
            rows=[{"name": r["name"], "scenic_count": to_int(r["scenic_count"])} for r in batch],
        )
    print("[4/6] 类别实体导入完成：%d 个" % len(rows))


def import_scenics(g, rows):
    """导入景点，并与城市建立 HAS_SPOT 关系"""
    total = len(rows)
    done = 0
    for batch in chunks(rows, BATCH_SIZE):
        payload = []
        for r in batch:
            payload.append({
                "sid": "%s|%s" % (r["city"], r["name"]),
                "name": r["name"],
                "name_zh": r["name_zh"] or None,
                "name_en": r["name_en"] or None,
                "city": r["city"],
                "address": r["address"] or None,
                "district": r["district"] or None,
                "phone": r["phone"] or None,
                "website": r["website"] or None,
                "score": to_float(r["score"]),
                "open_time": r["open_time"] or None,
                "ticket_min_price": to_float(r["ticket_min_price"]),
                "ticket_type_count": to_int(r["ticket_type_count"]),
                "ticket_names": r["ticket_names"] or None,
                "suggest_duration": r["suggest_duration"] or None,
                "best_season_raw": r["best_season_raw"] or None,
                "category": r["category"] or None,
                "intro": r["intro"] or None,
                "url": r["url"] or None,
                "image": r["image"] or None,
            })
        g.run(
            "UNWIND $rows AS row "
            "MERGE (c:City {name: row.city}) "
            "MERGE (s:Scenic {sid: row.sid}) "
            "SET s.name = row.name, s.name_zh = row.name_zh, s.name_en = row.name_en, "
            "    s.city = row.city, s.address = row.address, "
            "    s.district = row.district, s.phone = row.phone, s.website = row.website, "
            "    s.score = row.score, s.open_time = row.open_time, "
            "    s.ticket_min_price = row.ticket_min_price, "
            "    s.ticket_type_count = row.ticket_type_count, "
            "    s.ticket_names = row.ticket_names, "
            "    s.suggest_duration = row.suggest_duration, "
            "    s.best_season_raw = row.best_season_raw, "
            "    s.category = row.category, "
            "    s.intro = row.intro, s.url = row.url, s.image = row.image "
            "MERGE (c)-[:HAS_SPOT]->(s)",
            rows=payload,
        )
        done += len(batch)
        print("      景点 %d/%d" % (done, total), end="\r")
    print("[5/6] 景点实体 + HAS_SPOT 关系导入完成：%d 个" % total)


def import_relations(g, rows, rel_type, target_label, key_field):
    """导入 景点 -> 季节 / 类别 的关系"""
    total = 0
    for batch in chunks(rows, BATCH_SIZE):
        payload = [{"sid": "%s|%s" % (r["city"], r["scenic_name"]),
                    "target": r[key_field]} for r in batch]
        g.run(
            "UNWIND $rows AS row "
            "MATCH (s:Scenic {sid: row.sid}) "
            "MATCH (t:%s {name: row.target}) "
            "MERGE (s)-[:%s]->(t)" % (target_label, rel_type),
            rows=payload,
        )
        total += len(batch)
    print("[6/6] %s 关系导入完成：%d 条" % (rel_type, total))


def verify(g):
    print("\n" + "=" * 46)
    print("图谱导入结果校验")
    print("=" * 46)
    q = "MATCH (n) RETURN count(n) AS c"
    print("节点总数        :", g.run(q).data()[0]["c"])
    q = "MATCH ()-[r]->() RETURN count(r) AS c"
    print("关系总数        :", g.run(q).data()[0]["c"])
    for label in ("City", "Scenic", "Season", "Category"):
        c = g.run("MATCH (n:%s) RETURN count(n) AS c" % label).data()[0]["c"]
        print("%-14s : %d" % (label, c))
    print("-" * 46)
    print("示例查询：上海评分最高的 5 个景点")
    rows = g.run(
        "MATCH (c:City {name:'上海'})-[:HAS_SPOT]->(s:Scenic) "
        "WHERE s.score IS NOT NULL "
        "RETURN s.name AS 景点, s.score AS 评分, s.district AS 区县 "
        "ORDER BY s.score DESC LIMIT 5"
    ).data()
    for r in rows:
        print("   %-24s %.1f   %s" % (r["景点"], r["评分"], r["区县"] or ""))
    print("=" * 46)


def main():
    parser = argparse.ArgumentParser(description="城市景点知识图谱批量导入")
    parser.add_argument("--reset", action="store_true", help="导入前先清空数据库")
    args = parser.parse_args()

    if not os.path.isdir(DATA_DIR):
        print("找不到 cleaned/ 目录，请先运行 preprocess_citydata.py")
        sys.exit(1)

    print("连接 Neo4j: %s" % NEO4J_URI)
    g = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    g.run("RETURN 1").data()
    print("连接成功\n")

    t0 = time.time()

    if args.reset:
        clear_graph(g)
    create_constraints(g)

    cities = read_csv("city.csv")
    seasons = read_csv("season.csv")
    categories = read_csv("category.csv")
    scenics = read_csv("scenic.csv")
    rel_season = read_csv("relation_scenic_season.csv")
    rel_category = read_csv("relation_scenic_category.csv")

    import_cities(g, cities)
    import_seasons(g, seasons)
    import_categories(g, categories)
    import_scenics(g, scenics)
    import_relations(g, rel_season, "BEST_SEASON", "Season", "season")
    import_relations(g, rel_category, "BELONGS_TO", "Category", "category")

    verify(g)
    print("\n耗时 %.1f 秒" % (time.time() - t0))
    print("打开 http://localhost:7474 即可在 Browser 中查看图谱")


if __name__ == "__main__":
    main()
