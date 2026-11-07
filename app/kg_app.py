# -*- coding: utf-8 -*-
"""
实验四 · 领域知识图谱应用（综合 Web 应用）
=========================================
技术栈：Flask（后端） + py2neo（连 Neo4j） + ECharts（前端可视化）
功能模块：① 景点检索  ② 智能推荐  ③ 子图可视化  ④ 统计看板  ⑤ 简易问答

启动：
    1) 启动 Neo4j 数据库（默认 http://localhost:7474 / bolt://localhost:7687）
    2) pip install -r requirements.txt，并按 README 配置环境变量
    3) cd city_kg/app && python kg_app.py
    4) 浏览器打开 http://localhost:5000

修复记录（本次排查出的主要问题）：
  [严重] ORDER BY ... NULLS LAST 是 Cypher 非法语法 -> 推荐/问答 500 报错。
         改为 ORDER BY (CASE WHEN s.score IS NULL THEN 1 ELSE 0 END) ASC, s.score DESC
  [严重] WHERE 写在 OPTIONAL MATCH 之后 -> 过滤被静默忽略（Cypher 语义：不匹配的
         行会被保留、变量置 null），导致推荐/问答永远返回全局结果。改用 EXISTS{} 子查询。
  [严重] 季节取值不一致：库里是 春/夏/秋/冬/四季皆宜，前端写死 春季/夏季 -> 季节筛选恒为空。
         改为下拉项全部由数据库动态生成（/api/opts）。
  [严重] 数据库只在启动时连一次，断开后永不重连 -> 每个请求前 ensure_db() 自动重连。
  [一般] 子图关系方向反了（startNode/endNode 逻辑写反）-> 直接用 startNode(r)/endNode(r)。
  [一般] ECharts 依赖外网 CDN，断网即图表模块全哑 -> 本地 app/static/echarts.min.js，
         缺失时才回退 CDN，并给出明确提示。
  [一般] 重复 echarts.init 未 dispose；面板隐藏时图表尺寸为 0；无回车提交；HTML 未转义。
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from flask import Flask, request, jsonify, send_from_directory, render_template_string

try:
    from py2neo import Graph
except ImportError:
    Graph = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(APP_DIR)
STATIC_DIR = os.path.join(APP_DIR, "static")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["JSON_AS_ASCII"] = False

_graph = None
_db_ok = False
_last_err = None


# ---------------- 数据库连接（支持自动重连） ----------------
def connect():
    """尝试连接 Neo4j，可反复调用。"""
    global _graph, _db_ok, _last_err
    if Graph is None:
        _last_err = "py2neo 未安装，请 pip install py2neo"
        _db_ok = False
        return False
    try:
        g = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name=NEO4J_DB)
        g.run("RETURN 1").data()
        _graph, _db_ok, _last_err = g, True, None
        return True
    except Exception as e:
        _db_ok, _last_err = False, str(e)
        return False


def ensure_db():
    """每次请求前调用：先探活，失败则重连。这样"先起应用、后起数据库"也能自愈。"""
    global _db_ok
    if _db_ok and _graph is not None:
        try:
            _graph.run("RETURN 1").data()
            return True
        except Exception:
            _db_ok = False
    return connect()


def _db_error():
    tip = "请确认 Neo4j 已启动，并已设置环境变量 NEO4J_PASSWORD。"
    return jsonify({"ok": False, "error": "Neo4j 未连接：%s %s" % (tip, _last_err or "")}), 503


def _err(e):
    return jsonify({"ok": False, "error": str(e)}), 500


def _limit(v, default=30, hi=200):
    try:
        n = int(v)
    except Exception:
        n = default
    return max(1, min(n, hi))


def _page_args(default_size=30):
    """统一的翻页参数：返回 (page, size, offset)。

    页码必须夹(range clamp)一下——用户手改 URL 传 page=99999 时，
    若直接用 (page-1)*size 当 SKIP，深翻会让 Neo4j 白扫一大片。
    """
    size = _limit(request.args.get("size"), default=default_size, hi=100)
    try:
        page = int(request.args.get("page", 1))
    except Exception:
        page = 1
    page = max(1, min(page, 2000))
    return page, size, (page - 1) * size


def _pager(total, page, size):
    """构造翻页信息。total 可能为 0（此时 pages=1，前端不显示翻页条）。

    若请求的页码超过总页数（例如换了筛选条件后旧页码还留在前端），
    自动回落到最后一页——否则用户会看到一片空白还以为是没数据。
    """
    pages = max(1, -(-total // size)) if size else 1
    page = max(1, min(page, pages))
    start = (page - 1) * size
    return {"page": page, "pages": pages, "size": size, "total": total,
            "from": (start + 1) if total else 0,
            "to": min(start + size, total)}


connect()


# ---------------- Cypher 模板 ----------------
# 注意：绝不使用 f-string/format 渲染下面这些语句——EXISTS { ... } 的花括号会被吞掉。
_ORDER_DESC = ("ORDER BY (CASE WHEN s.score IS NULL THEN 1 ELSE 0 END) ASC, "
               "s.score DESC, s.sid ASC")   # 把 null 排到最后（Neo4j 的 DESC 默认把 null 排最前）
# ⚠️ 末尾的 s.sid 绝对不能删：它是分页稳定性的保证。
# 大量景点评分并列（青岛一城就有 20 个 4.6、19 个 4.5），只按 score 排序时并列行内部顺序
# 在不同查询间并不保证一致，翻页会出现"同一条重复出现在两页、而另有条目永远取不到"。
# sid（城市|景点名）全库唯一，补上它做次级排序键即成稳定排序。

# 多值关系（一个景点可能有多个 BEST_SEASON）不能用 OPTIONAL MATCH 展开：
# 会把同一景点复制成多行 -> 结果里出现重复条目、且显示的季节是随机的一个。
# 改用 Cypher 模式推导式（pattern comprehension）在 RETURN 里聚合成列表，保证一行一个景点。
# 注意：Cypher 的行注释是 //，千万不要写 --（那是匿名关系模式 MATCH (a)--(b)）。
# 下面两个字段用于前端的图片展示与官网外链：
# image 景点图片(35.2% 有) / website 景点官网(4.4% 有)
# 注：原本还返回过 url（第三方站点详情页），但该站点的链接大多已失效，已移除。
_SELECT = """
       s.name AS name, s.city AS city, s.score AS score,
       s.district AS district, s.address AS address,
       s.open_time AS open_time, s.ticket_min_price AS ticket,
       substring(coalesce(s.intro, ''), 0, 110) AS intro,
       coalesce(s.image,'') AS image,
       coalesce(s.website,'') AS website,
       [(s)-[:BEST_SEASON]->(x:Season) | x.name] AS season_list,
       [(s)-[:BELONGS_TO]->(y:Category) | y.name] AS category_list"""

# 推荐/问答：WHERE 必须放在 OPTIONAL MATCH 之前，并用 EXISTS{} 做真正的过滤。
# 类别用列表 $cats：一句"青岛自然景观"要同时命中 自然山地 + 水域景观 两个类别。
_RECO_CORE = """
MATCH (s:Scenic)
WHERE ($city IS NULL OR EXISTS { MATCH (s)<-[:HAS_SPOT]-(:City {name:$city}) })
  AND ($season IS NULL OR EXISTS { MATCH (s)-[:BEST_SEASON]->(:Season {name:$season}) })
  AND ($cats IS NULL OR EXISTS { MATCH (s)-[:BELONGS_TO]->(c:Category) WHERE c.name IN $cats })
RETURN""" + _SELECT + "\n" + _ORDER_DESC + "\nLIMIT __LIMIT__"

_SEARCH_CORE = """
MATCH (s:Scenic)
WHERE s.name CONTAINS $q OR coalesce(s.city,'') CONTAINS $q
   OR coalesce(s.address,'') CONTAINS $q OR coalesce(s.name_en,'') CONTAINS $q
RETURN""" + _SELECT + "\n" + _ORDER_DESC + "\nLIMIT __LIMIT__"

# 与 _SEARCH_CORE 同款的筛选条件，只用来数总数（检索分页需要精确总页数）
_SEARCH_COUNT = """
MATCH (s:Scenic)
WHERE s.name CONTAINS $q OR coalesce(s.city,'') CONTAINS $q
   OR coalesce(s.address,'') CONTAINS $q OR coalesce(s.name_en,'') CONTAINS $q
RETURN count(s) AS c"""

# 与 _KEYWORD_CORE 同款的筛选条件，只用来数总数
_KEYWORD_COUNT = """
MATCH (s:Scenic)
WHERE ($kw IS NULL OR $kw = '' OR s.name CONTAINS $kw
       OR coalesce(s.district,'') CONTAINS $kw
       OR coalesce(s.address,'') CONTAINS $kw
       OR coalesce(s.name_en,'') CONTAINS $kw)
  AND ($city IS NULL OR $city = '' OR coalesce(s.city,'') = $city
       OR EXISTS { MATCH (s)<-[:HAS_SPOT]-(:City {name:$city}) })
RETURN count(s) AS c"""

# 与 _RECO_CORE 完全同款的筛选条件，只用来数总数（回答里要告诉用户"共 N 个，显示前 M"）
_RECO_COUNT = """
MATCH (s:Scenic)
WHERE ($city IS NULL OR EXISTS { MATCH (s)<-[:HAS_SPOT]-(:City {name:$city}) })
  AND ($season IS NULL OR EXISTS { MATCH (s)-[:BEST_SEASON]->(:Season {name:$season}) })
  AND ($cats IS NULL OR EXISTS { MATCH (s)-[:BELONGS_TO]->(c:Category) WHERE c.name IN $cats })
RETURN count(s) AS c"""

# 某城市（或全国）的真实类别分布——条件太窄没结果时用来"引导用户挑一个"
_CAT_DIST = """
MATCH (s:Scenic)-[:BELONGS_TO]->(c:Category)
WHERE ($city IS NULL OR coalesce(s.city,'') = $city
       OR EXISTS { MATCH (s)<-[:HAS_SPOT]-(:City {name:$city}) })
RETURN c.name AS name, count(s) AS cnt ORDER BY cnt DESC"""

# 关键词检索：把「城市 + 景点关键词」拆开匹配，解决"青岛金沙滩"这种整串匹配不上的问题。
# 命中城市时只在该城市内找；名字命中的排在区域/地址命中之前。
_KEYWORD_CORE = """
MATCH (s:Scenic)
WHERE ($kw IS NULL OR $kw = '' OR s.name CONTAINS $kw
       OR coalesce(s.district,'') CONTAINS $kw
       OR coalesce(s.address,'') CONTAINS $kw
       OR coalesce(s.name_en,'') CONTAINS $kw)
  AND ($city IS NULL OR $city = '' OR coalesce(s.city,'') = $city
       OR EXISTS { MATCH (s)<-[:HAS_SPOT]-(:City {name:$city}) })
RETURN""" + _SELECT + """
ORDER BY (CASE WHEN $kw IS NULL OR $kw = '' THEN 0
               WHEN s.name = $kw THEN 0
               WHEN s.name STARTS WITH $kw THEN 1
               WHEN s.name CONTAINS $kw THEN 2
               ELSE 3 END) ASC,
         (CASE WHEN $city IS NULL OR $city = '' THEN 0
               WHEN coalesce(s.city,'') = $city THEN 0 ELSE 1 END) ASC,
         (CASE WHEN s.score IS NULL THEN 1 ELSE 0 END) ASC, s.score DESC,
         s.sid ASC
LIMIT __LIMIT__"""      # 末尾 s.sid 保证翻页时顺序稳定（详见 _ORDER_DESC 的注释）


def _rows(core, **kw):
    """执行查询并把 None 一律转为空串，方便前端直接渲染。

    另外把 season_list / category_list 合并成字符串：
    一个景点可能同时属于多个季节（库里有 716 个），合并后才不会出现重复条目。

    kw 支持 offset：在 LIMIT 前插入 SKIP 实现翻页。
    """
    lim = _limit(kw.pop("limit", 30))
    off = int(kw.pop("offset", 0) or 0)
    sql = core.replace("__LIMIT__", str(lim))
    if off > 0:
        sql = sql.replace("LIMIT %d" % lim, "SKIP %d LIMIT %d" % (off, lim), 1)
    data = _graph.run(sql, **kw).data()
    out = []
    for r in data:
        d = {}
        for k, v in r.items():
            if isinstance(v, float):
                v = round(v, 2)
            d[k] = "" if v is None else v
        for k in ("season_list", "category_list"):
            vals = sorted({str(x) for x in (d.pop(k, None) or []) if x})
            d[k.replace("_list", "")] = "、".join(vals)
        out.append(d)
    return out


# ---------------- API ----------------
@app.route("/api/health")
def health():
    ensure_db()
    return jsonify({"ok": True, "db": _db_ok, "error": _last_err})


@app.route("/api/llm_status")
def llm_status():
    """大模型接入状态：界面上要能看出"AI 是否真的在用"，否则演示时说不清。"""
    return jsonify({"ok": True, "enabled": bool(LLM_API_KEY), "model": LLM_MODEL,
                    "available": _llm_available(), "cached": len(_LLM_CACHE),
                    "calls": _LLM_STAT["calls"], "ok_cnt": _LLM_STAT["ok"],
                    "cache_hit": _LLM_STAT["cache_hit"],
                    "fail": _LLM_STAT["fail"], "last_error": _LLM_STAT["last_error"]})


@app.route("/api/opts")
def opts():
    """下拉框选项 + 联动计数（facet count）。

    括号里的数字必须是"在当前其它筛选条件下、选它能得到多少条"，否则会误导：
    例如城市显示 90、季节显示 519，但两个一起选只有 7 条。
    因此每个维度统计时都**排除自己那一维**的筛选，避免一选就把同维其它选项打成 0。
    """
    if not ensure_db():
        return _db_error()
    city = request.args.get("city") or None
    season = request.args.get("season") or None
    # 下拉框传的是单个类别字符串；这里统一成列表，与 _RECO_* 模板的 $cats 对齐
    cat_q = request.args.get("category") or None
    cats_q = [cat_q] if cat_q else None
    try:
        cities = _graph.run(
            "MATCH (c:City)-[:HAS_SPOT]->(s:Scenic) "
            "WHERE ($season IS NULL OR EXISTS { MATCH (s)-[:BEST_SEASON]->(:Season {name:$season}) }) "
            "  AND ($cats IS NULL OR EXISTS { MATCH (s)-[:BELONGS_TO]->(c2:Category) WHERE c2.name IN $cats }) "
            "RETURN c.name AS name, count(s) AS cnt ORDER BY cnt DESC, name",
            season=season, cats=cats_q).data()
        seasons = _graph.run(
            "MATCH (s:Scenic)-[:BEST_SEASON]->(se:Season) "
            "WHERE ($city IS NULL OR coalesce(s.city,'') = $city "
            "       OR EXISTS { MATCH (s)<-[:HAS_SPOT]-(:City {name:$city}) }) "
            "  AND ($cats IS NULL OR EXISTS { MATCH (s)-[:BELONGS_TO]->(c2:Category) WHERE c2.name IN $cats }) "
            "RETURN se.name AS name, count(s) AS cnt ORDER BY cnt DESC, name",
            city=city, cats=cats_q).data()
        cat_rows = _graph.run(
            "MATCH (s:Scenic)-[:BELONGS_TO]->(c:Category) "
            "WHERE ($city IS NULL OR coalesce(s.city,'') = $city "
            "       OR EXISTS { MATCH (s)<-[:HAS_SPOT]-(:City {name:$city}) }) "
            "  AND ($season IS NULL OR EXISTS { MATCH (s)-[:BEST_SEASON]->(:Season {name:$season}) }) "
            "RETURN c.name AS name, count(s) AS cnt ORDER BY cnt DESC, name",
            city=city, season=season).data()
        total = _graph.run(_RECO_COUNT, city=city, season=season, cats=cats_q).data()[0]["c"]
        j = lambda r: {"name": r["name"], "cnt": r["cnt"]}
        return jsonify({"ok": True, "cities": [j(x) for x in cities],
                        "seasons": [j(x) for x in seasons],
                        "categories": [j(x) for x in cat_rows],
                        "total": total,
                        "filters": {"city": city, "season": season, "category": cat_q}})
    except Exception as e:
        return _err(e)


@app.route("/api/stats")
def stats():
    if not ensure_db():
        return _db_error()
    try:
        total = _graph.run("MATCH (n) RETURN count(n) AS c").data()[0]["c"]
        rel_total = _graph.run("MATCH ()-[r]->() RETURN count(r) AS c").data()[0]["c"]
        labels = {}
        for lab in ("City", "Scenic", "Season", "Category", "Doc", "NEREntity"):
            labels[lab] = _graph.run("MATCH (n:%s) RETURN count(n) AS c" % lab).data()[0]["c"]
        rels = {r["t"]: r["c"] for r in
                _graph.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c").data()}
        top_cities = _graph.run(
            "MATCH (c:City)-[:HAS_SPOT]->(s) RETURN c.name AS name, count(s) AS cnt "
            "ORDER BY cnt DESC LIMIT 10").data()
        cats = _graph.run(
            "MATCH (s:Scenic)-[:BELONGS_TO]->(c:Category) "
            "RETURN c.name AS name, count(s) AS cnt ORDER BY cnt DESC").data()
        seasons = _graph.run(
            "MATCH (s:Scenic)-[:BEST_SEASON]->(se:Season) "
            "RETURN se.name AS name, count(s) AS cnt ORDER BY cnt DESC").data()
        # 字段完整度：让"为什么这个景点没显示简介"这类疑问在界面上直接有答案
        scenic_total = labels.get("Scenic", 0)
        completeness = []
        for prop, label in _FIELDS:
            c = _graph.run(
                "MATCH (s:Scenic) WHERE s.%s IS NOT NULL "
                "AND trim(coalesce(toString(s.%s),'')) <> '' RETURN count(s) AS c"
                % (prop, prop)).data()[0]["c"]
            completeness.append({"field": prop, "label": label, "cnt": c,
                                 "pct": round(100.0 * c / scenic_total, 1) if scenic_total else 0.0})
        return jsonify({"ok": True, "total": total, "rel_total": rel_total,
                        "labels": labels, "rels": rels,
                        "top_cities": top_cities, "cats": cats, "seasons": seasons,
                        "scenic_total": scenic_total, "completeness": completeness})
    except Exception as e:
        return _err(e)


@app.route("/api/search")
def search():
    if not ensure_db():
        return _db_error()
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"ok": True, "items": [], "answer": "请输入景点名、城市或地址关键词。"})
    try:
        city = _longest_in(q, _city_names())
        kw = _keyword_of(q, city)
        page, size, _ = _page_args(20)

        def paged(core, count_tpl, params):
            """先数总数再取当前页：深翻/刷新时页码信息仍然正确。"""
            t = _graph.run(count_tpl, **params).data()[0]["c"]
            pg = _pager(t, page, size)
            rows = _rows(core, limit=size, offset=(pg["page"] - 1) * size, **params)
            return rows, pg

        if kw:
            # 判断"整句是不是真景点名"：只要存在名字包含整句的景点就走整句分支。
            # 不能沿用"看 items[0] 是否命中"的老写法——翻到第 2 页时 items[0] 已经不是全库最优那条了。
            exact = _graph.run(
                "MATCH (s:Scenic) WHERE s.name CONTAINS $q RETURN count(s) AS c", q=q
            ).data()[0]["c"]
            if exact:
                # ① 整句匹配：保证"青岛大学"精确命中，而不是拆词后被"青岛大学路"抢占
                items, pg = paged(_KEYWORD_CORE, _KEYWORD_COUNT, {"kw": q, "city": None})
                note = "已按整句「%s」匹配" % q
            else:
                # ② 拆成 城市+关键词（"青岛金沙滩" -> 青岛 + 金沙滩）
                items, pg = paged(_KEYWORD_CORE, _KEYWORD_COUNT, {"kw": kw, "city": city})
                note = ("已按【城市=%s】+【关键词=%s】检索" % (city, kw)) if city \
                    else ("已按【关键词=%s】检索" % kw)
                if not items and city:
                    items, pg = paged(_KEYWORD_CORE, _KEYWORD_COUNT, {"kw": kw, "city": None})
                    note = "%s没有匹配到，已放宽到全部地区检索【%s】" % (city, kw)
        elif city:
            # 纯城市：复用推荐。_RECO_CORE 里同时引用了 $season/$cats，必须都传
            pg = _pager(_reco_total(city, None, None), page, size)
            items = _rows(_RECO_CORE, city=city, season=None, cats=None,
                          limit=size, offset=(pg["page"] - 1) * size)
            note = "已按【城市=%s】列出热门景点" % city
        else:
            items, pg = paged(_SEARCH_CORE, _SEARCH_COUNT, {"q": q})
            note = ""
        return jsonify({"ok": True, "items": items, "count": len(items),
                        "total": pg["total"], "page": pg, "answer": note})
    except Exception as e:
        return _err(e)


@app.route("/api/recommend")
def recommend():
    if not ensure_db():
        return _db_error()
    city = request.args.get("city") or None
    season = request.args.get("season") or None
    category = request.args.get("category") or None
    cats = [category] if category else None
    page, size, offset = _page_args(30)
    total = _reco_total(city, season, cats)
    pg = _pager(total, page, size)
    try:
        # 页码被 _pager 夹到合法范围内后重新算 SKIP：避免翻到最后一页时返回空列表
        offset = (pg["page"] - 1) * size
        items = _rows(_RECO_CORE, city=city, season=season, cats=cats,
                      limit=size, offset=offset)
        cond = "、".join(filter(None, [
            city and ("城市=" + city), season and ("季节=" + season),
            category and ("类别=" + category)])) or "不限"
        if items:
            ans = ("按【%s】为您推荐：共 %d 个匹配，按评分排序，当前显示第 %d-%d 个。"
                   % (cond, total, pg["from"], pg["to"]))
        else:
            ans = "按【%s】没有匹配结果，试试放宽条件（例如只选城市）。" % cond
        return jsonify({"ok": True, "answer": ans, "items": items,
                        "count": len(items), "total": total, "page": pg})
    except Exception as e:
        return _err(e)


@app.route("/api/graph")
def graph():
    if not ensure_db():
        return _db_error()
    q = (request.args.get("name") or "").strip()
    try:
        # 解析输入 -> 种子景点。必须"先认出城市、再认景点"，否则搜"青岛"会命中
        # 宜春的「青岛扎啤美食文化广场」（名字里含"青岛"而已）。
        seeds, answer = _resolve_graph_seed(q)

        if not seeds:
            return jsonify({"ok": True, "nodes": [], "edges": [],
                            "count": {"nodes": 0, "edges": 0},
                            "answer": "没有找到与「%s」匹配的城市或景点。" % q})

        rows = _graph.run(_NEIGH, sids=[s["sid"] for s in seeds]).data()
        nodes, edges, seen = [], [], set()

        def key_of(n):
            return (tuple(sorted(n.labels)), n.get("name"))

        def add_node(n):
            if n is None or n.get("name") is None:
                return
            k = key_of(n)
            if k in seen:
                return
            seen.add(k)
            nodes.append({"name": n.get("name"), "label": _label_of(n),
                          "cat": n.get("category") or n.get("district") or ""})

        for r in rows:
            s, t, a, b = r["s"], r["t"], r["a"], r["b"]
            if s:
                add_node(s)
            if a:
                add_node(a)
            if b:
                add_node(b)
            if t and a is not None and b is not None:
                edges.append({"source": a.get("name"), "target": b.get("name"), "type": t})
        return jsonify({"ok": True, "nodes": nodes, "edges": edges,
                        "answer": answer,
                        "count": {"nodes": len(nodes), "edges": len(edges)}})
    except Exception as e:
        return _err(e)


# ---------------- 子图种子解析 ----------------
# 一次拉回候选，在 Python 里排序，避免多次往返数据库。
# ORDER BY 很关键：像"公园"这种高频词有 3345 个候选，不排序就 LIMIT 60 会把真正的正主截掉。
_SCENIC_CAND = """
MATCH (s:Scenic) WHERE s.name CONTAINS $q
WITH s, CASE WHEN s.name = $q THEN 0 WHEN s.name STARTS WITH $q THEN 1 ELSE 2 END AS tier
RETURN s.sid AS sid, s.name AS name, s.city AS city, s.score AS score, tier
ORDER BY tier, (CASE WHEN s.score IS NULL THEN 1 ELSE 0 END) ASC, s.score DESC, s.sid ASC
LIMIT 60"""

# 某城市评分最高的 N 个景点（sid 全库唯一，用它做种子最稳）
_CITY_SEEDS = """
MATCH (:City {name:$c})-[:HAS_SPOT]->(s:Scenic)
WITH s ORDER BY (CASE WHEN s.score IS NULL THEN 1 ELSE 0 END) ASC, s.score DESC, s.sid ASC
LIMIT $n
RETURN s.sid AS sid, s.name AS name"""

# 种子景点的 1 跳邻居
_NEIGH = """
MATCH (s:Scenic) WHERE s.sid IN $sids
OPTIONAL MATCH (s)-[r]-(n)
RETURN s, type(r) AS t, startNode(r) AS a, endNode(r) AS b
LIMIT 600"""


def _scenic_candidates(q, city=None, only_city=False):
    """按"谁更像正主"给候选景点排序。

    原来的写法是 `WITH s LIMIT 1`（无排序）——取到哪条完全随机，
    搜"青岛"时命中了宜春的「青岛扎啤美食文化广场」，这就是那次报障的根因。

    分级规则（关键在于第 1 档）：
      0. 中文名与关键词完全相同                        → "青岛大学"
      1. 关键词之后**紧跟非中文**（即"正主 + 拉丁译名"形态）→ "西湖West Lake"（比"西湖石坊"更像正主）
      2. 关键词之后还有中文                            → "西湖石坊"、"金沙滩海水浴场"
      3. 仅包含关键词                                  → "黄岛金沙滩"
    同档再按评分降序（null 最后）。
    """
    if not q:
        return []
    rows = _graph.run(_SCENIC_CAND, q=q).data()
    if only_city and city:
        # 已经从问句里认出了城市（"青岛金沙滩"），就必须限定在该城市内找，否则会被外地同名景点抢走
        rows = [r for r in rows if r.get("city") == city]

    def rank(r):
        name = r["name"] or ""
        zh = "".join(re.findall(r"[\u4e00-\u9fa5A-Za-z·]+", name)) or name
        if zh == q:
            tier = 0
        elif zh.startswith(q):
            tail = name.split(q, 1)[-1] if q in name else ""
            tier = 1 if not re.match(r"[\u4e00-\u9fa5]", tail) else 2
        else:
            tier = 3
        score = r["score"] if isinstance(r["score"], (int, float)) and r["score"] else -1
        return (tier, -score, len(zh), name)

    rows.sort(key=rank)
    return rows


def _city_seeds(city, n=12):
    return _graph.run(_CITY_SEEDS, c=city, n=n).data()


def _resolve_graph_seed(q):
    """把用户输入解析成 (种子景点列表, 说明文字)。

    判定顺序（这个顺序很关键）：
      ① 输入正好是城市名      -> 该城市评分最高的若干景点（"搜青岛就想看青岛"）
      ② 输入能匹配到景点名    -> 该景点的 1 跳邻居
      ③ 输入里含城市名 + 其余词 -> 用其余词在该城市内找景点（"青岛金沙滩" -> 黄岛金沙滩）
      ④ 输入里含城市名        -> 退化为该城市的景点
      ⑤ 都没有                -> 空，由调用方给出提示
    """
    names = _city_names()
    city_exact = q if q in names else None

    if city_exact:
        seeds = _city_seeds(city_exact)
        if seeds:
            return seeds, ("已展示城市「%s」评分最高的 %d 个景点及其类别/季节关系。"
                           % (city_exact, len(seeds)))

    if q:
        cand = _scenic_candidates(q)
        if cand:
            return cand[:1], "已展示景点「%s」（所属城市 %s）的 1 跳邻居子图。" % (
                cand[0]["name"], cand[0].get("city") or "—")

    city_in_q = _longest_in(q, names) if q else None
    if city_in_q:
        rest = q.replace(city_in_q, "", 1).strip()
        if rest:
            # 已在问句里认出城市 -> 只在该城市内找，避免被外地同名景点抢走
            # （"青岛金沙滩"若不限定，会命中日照的"金沙滩海水浴场"）
            cand = _scenic_candidates(rest, city=city_in_q, only_city=True)
            if cand:
                return cand[:1], "已按「%s + %s」定位到景点「%s」（所属城市 %s）的 1 跳邻居子图。" % (
                    city_in_q, rest, cand[0]["name"], cand[0].get("city") or city_in_q)
            cand = _scenic_candidates(rest)
            if cand:
                return cand[:1], "按「%s + %s」在该城市内没有匹配，已放宽到全部地区，展示景点「%s」的 1 跳邻居子图。" % (
                    city_in_q, rest, cand[0]["name"])
        seeds = _city_seeds(city_in_q)
        if seeds:
            return seeds, ("没有名字含「%s」的景点，改为展示城市「%s」评分最高的 %d 个景点。"
                           % (q, city_in_q, len(seeds)))

    if not q:
        seeds = _city_seeds("北京", 8)
        if seeds:
            return seeds, "未输入关键词，默认展示城市「北京」评分最高的 %d 个景点。" % len(seeds)
    return [], None


def _label_of(node):
    if node is None:
        return "X"
    for lab in ("City", "Season", "Category", "Scenic"):
        if lab in node.labels:
            return lab
    return "X"


# ---------------- 简易问答 ----------------
_SEASON_SYN = {
    "春": "春", "春天": "春", "春季": "春",
    "夏": "夏", "夏天": "夏", "夏季": "夏",
    "秋": "秋", "秋天": "秋", "秋季": "秋",
    "冬": "冬", "冬天": "冬", "冬季": "冬",
    "四季": "四季皆宜", "四季皆宜": "四季皆宜", "全年": "四季皆宜", "任何时候": "四季皆宜",
}
_CITY_CACHE = {"ts": 0.0, "names": []}
_CAT_CACHE = {"ts": 0.0, "names": []}

# 统计看板要展示的字段完整度（属性名, 显示名）
_FIELDS = (("intro", "简介"), ("address", "地址"), ("district", "区域"),
           ("score", "评分"), ("open_time", "开放时间"), ("ticket_min_price", "票价"),
           ("phone", "电话"), ("website", "官网"), ("image", "图片"))


def _cached(cache, cy, ttl=300):
    now = time.time()
    if now - cache["ts"] > ttl or not cache["names"]:
        cache["names"] = [r["name"] for r in _graph.run(cy).data()]
        cache["ts"] = now
    return cache["names"]


def _city_names():
    return _cached(_CITY_CACHE, "MATCH (c:City) RETURN c.name AS name")


def _cat_names():
    return _cached(_CAT_CACHE, "MATCH (c:Category) RETURN c.name AS name")


def _longest_in(q, names, skip=()):
    """在句子里找最长的那个候选词；长度相同时取出现位置更靠前的。

    不能只取"第一个匹配到的"：城市名可能互相包含，且一句里可能出现多个城市。
    """
    best, best_pos = None, 10 ** 9
    for n in names:
        if not n or n in skip:
            continue
        pos = q.find(n)
        if pos < 0:
            continue
        if (best is None or len(n) > len(best)
                or (len(n) == len(best) and pos < best_pos)):
            best, best_pos = n, pos
    return best


# 口语 → 图谱真实类别。用户不会说"自然山地"，会说"自然景观/山水/海边"。
# 注意：这里只放"多字词"，绝不放"岛""山"这类单字——否则"青岛"里的"岛"会被误判成类别"岛屿"。
 # 有些说法一次对应多个类别："自然景观"既含山地也含江河湖海
_CAT_PHRASE = (
    ("自然景观", ("自然山地", "水域景观")),
    ("自然风光", ("自然山地", "水域景观")),
    ("自然风景", ("自然山地", "水域景观")),
    ("山水", ("自然山地", "水域景观")),
)

_CAT_SYN = (
    ("自然山地", ("山地", "山峰", "登山", "爬山")),
    # 只用多字词："湖""河""塔""庙"这类单字会把"西湖""泰山"等真实景点名切坏
    ("水域景观", ("水域", "海边", "海滨", "海滩", "沙滩", "海景", "看海", "瀑布", "温泉", "漂流")),
    ("宗教建筑", ("宗教", "寺庙", "寺院", "教堂", "道观", "佛像", "石窟")),
    ("人文场馆", ("博物馆", "展馆", "纪念馆", "美术馆", "科技馆", "图书馆", "人文")),
    ("公园园林", ("公园", "园林", "花园", "绿地", "植物园")),
    # 注意别把"长城"当类别词——它是具体景点名，放进来会把"长城评分"这类提问带偏
    ("历史遗迹", ("历史遗迹", "历史", "遗迹", "遗址", "古迹", "古建筑", "故居", "文物")),
    ("古村镇", ("古镇", "古村", "村落", "村寨", "民俗村", "老街")),
    ("岛屿", ("海岛", "岛屿")),
    ("主题乐园", ("主题乐园", "游乐园", "游乐场", "乐园", "亲子")),
    ("户外体验", ("户外", "露营", "徒步")),
)


def _detect_cats(q, allowed):
    """从问句里识别类别（可多选）。只看图谱里真实存在的类别，避免映射到不存在的值。"""
    hits = []
    for phrase, cs in _CAT_PHRASE:          # 先处理"一对多"的说法
        if phrase in q:
            for c in cs:
                if c in allowed and c not in hits:
                    hits.append(c)
    for cat, words in _CAT_SYN:
        if cat in allowed and cat not in hits and any(w in q for w in words):
            hits.append(cat)
    return hits or None


# 剥词用的词表：短语与普通同义词合并，长词优先，避免"自然景观"被"自然"截断
_CAT_STRIP = tuple(sorted(
    {w for _c, ws in _CAT_SYN for w in ws} | {p for p, _cs in _CAT_PHRASE},
    key=len, reverse=True))


def _detect_season(q):
    """取最长的那个季节说法：'春天' 要盖过 '春'，否则剥词时会剩下 '天' 这种残渣。"""
    best_w, best_v = None, None
    for w, val in _SEASON_SYN.items():
        if w in q and (best_w is None or len(w) > len(best_w)):
            best_w, best_v = w, val
    return best_v, best_w


# 问句里需要剥掉的"废话"。长的放前面避免被短词先截断；
# 只放功能词，**绝不放可能出现在景点名里的字**（如"山""海""和""园"）。
_QA_STOP = ("开放时间", "营业时间", "在哪个城市", "属于哪个城市", "联系方式", "怎么走",
            "有哪些景点", "有什么景点", "哪些景点", "什么景点", "有哪些", "有什么",
            "什么地方", "推荐一下", "介绍一下", "我想去", "想去", "问一下", "请问",
            "帮我", "查一下", "找一下", "看看", "怎么去", "去哪", "好玩吗", "多少钱",
            "景点", "景区", "好玩", "哪里", "哪儿", "哪些", "适合", "什么", "怎么",
            "怎样", "如何", "时候", "时间", "季节", "值得", "门票", "票价", "开放",
            "营业", "评分", "几分", "多少分", "多少", "费用", "贵不贵", "地址", "位置",
            "在哪儿", "在哪里", "几点", "电话", "号码", "路线", "公交", "地铁",
            "推荐", "介绍", "一下", "吗", "呢", "啊", "的", "了")


def _clean_entity(s):
    """只做"去标点 + 去疑问词"的轻量清洗，**不剥类别同义词**，用于事实问句里的实体名。"""
    t = s or ""
    for w in _QA_STOP:
        t = t.replace(w, " ")
    return re.sub(r"[\s，。？！、；：,.?!;:~·\-—_“”\"'‘’（）()《》]+", " ", t).strip()


def _keyword_of(q, *removes):
    """剥掉城市/类别/季节说法与停用词后，剩下的就是真正的景点关键词。

    除了已识别的类别名，还要剥掉**口语同义词**（"自然景观""海边"等），
    否则它们会残留成关键词，把查询带偏到无关的同名景点上。
    """
    t = q
    for piece in removes:
        if piece:
            t = t.replace(piece, " ")
    for w in _SEASON_SYN:
        t = t.replace(w, " ")
    for w in _CAT_STRIP:                  # 长词优先，避免"自然景观"被"自然"截断后残留"景观"
        t = t.replace(w, " ")
    for w in _QA_STOP:
        t = t.replace(w, " ")
    t = re.sub(r"[\s，。？！、；：,.?!;:~·\-—_“”\"'‘’（）()《》]+", " ", t)
    return t.strip()


def _reco_total(city=None, season=None, cats=None):
    return _graph.run(_RECO_COUNT, city=city, season=season, cats=cats).data()[0]["c"]


def _cat_dist(city):
    """某范围内真实存在的类别分布——用来在"条件没命中"时引导用户改选。"""
    return _graph.run(_CAT_DIST, city=city).data()


# 事实型问句：问某个具体景点的属性，直接给一句话答案，而不是甩一串列表。
# 形式基本是「实体 + 属性词」，所以取**属性词之前的部分**当实体名最稳：
# "青岛大学地址" -> 实体"青岛大学"；"故宫门票多少" -> 实体"故宫"。
_FACT_PAIRS = (
    ("ticket", ("门票", "票价", "多少钱", "费用", "贵不贵")),
    ("score", ("评分", "几分", "多少分", "好评")),
    ("open_time", ("开放时间", "营业时间", "几点开门", "几点")),
    ("address", ("地址", "在哪儿", "在哪里", "怎么走", "位置")),
    ("phone", ("电话", "联系方式", "号码")),
    ("city", ("在哪个城市", "属于哪个城市", "哪个城市")),
)
_FACT_LABEL = {"ticket": "门票", "score": "评分", "open_time": "开放时间",
               "address": "地址", "phone": "电话", "city": "所在城市"}


def _best_entity(name_kw, city=None, limit=15):
    """按景点名选出最"像正主"的那一个。

    多个候选时（"泰山"可命中"泰山西湖/泰山风景区"）：
    ① 中文名与关键词完全一致 → ② 以关键词开头 → ③ 仅包含；
    同档取名字更完整的（正主通常是全称，如"泰山风景区"而非"泰山西湖"），再按评分降序。
    """
    cands = _rows(_KEYWORD_CORE, kw=name_kw, city=city, limit=limit)
    cands = [c for c in cands if name_kw in (c["name"] or "")]
    if not cands:
        return None
    def _rank(c):
        zh = "".join(re.findall(r"[\u4e00-\u9fa5·]+", c["name"] or "")) or (c["name"] or "")
        tier = 0 if zh == name_kw else (1 if zh.startswith(name_kw) else 2)
        score = c["score"] if isinstance(c["score"], (int, float)) and c["score"] else -1
        return (tier, -score, -len(zh), c["name"] or "")
    return sorted(cands, key=_rank)[0]


def _answer_fact(q, city, limit):
    """识别"XX的门票多少"这类问法。命中则返回 (答案文本, 卡片列表)，否则 None。"""
    field, word = None, None
    for f, words in _FACT_PAIRS:
        for w in words:
            if w in q and (word is None or len(w) > len(word)):
                field, word = f, w
    if not field:
        return None
    # 属性词之前的部分 = 实体名。这里**只去标点与疑问词，不去类别同义词**，
    # 否则"长城评分"里的"长城"会被当类别词剥掉，永远问不出答案。
    head = q.split(word)[0]
    name_kw = _clean_entity(head) or _clean_entity(q)
    if len(name_kw) < 2:
        return None
    it = _best_entity(name_kw, city)
    items = [it]                                  # 只展示被回答的那一个，避免"答A却显示B"
    label = _FACT_LABEL[field]
    val = it.get(field)
    if field == "ticket" and val not in ("", None):
        val = "约 ¥%s 起" % val
    if val not in ("", None):
        return "【%s】%s：%s" % (it["name"], label, val), items
    return "【%s】的%s在原始数据中未采集到。" % (it["name"], label), items


# ---------------- 大模型意图解析（Agnes AI，OpenAI 兼容协议） ----------------
# 说明：模型只负责"把自然语言翻译成查询条件"，**所有事实值一律从 Neo4j 取**，
#       绝不采用模型自己给出的数值（实测它会把"故宫门票"编造成"旺季40元"）。
# 密钥只从环境变量读取；留空则自动降级为内置规则解析，仓库不含任何真实密钥。
LLM_API_KEY = os.environ.get("AGNES_API_KEY", "")
LLM_BASE_URL = os.environ.get("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1")
LLM_MODEL = os.environ.get("AGNES_MODEL", "agnes-2.5-flash")
LLM_TIMEOUT = int(os.environ.get("AGNES_TIMEOUT", "20"))
# ⚠️ 该模型是**推理型**模型：会先消耗几百 token 做内部思考再输出。
#    max_tokens 给小了会出现"思考用完、正文为空"——表现为 JSON 截断/空响应。必须留足。
LLM_MAX_TOKENS = int(os.environ.get("AGNES_MAX_TOKENS", "900"))

_LLM_CACHE = {}          # 问题 -> 解析结果（免费额度有限，必须缓存）
_LLM_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_cache.json")
_LLM_STAT = {"calls": 0, "ok": 0, "fail": 0, "cache_hit": 0,
             "last_error": None, "cooldown_until": 0.0}

# 事实字段：模型只能指认"查哪一个"，值必须由图谱给出
_FACT_BY_WORD = {"门票": "ticket", "票价": "ticket", "多少钱": "ticket", "费用": "ticket",
                 "评分": "score", "几分": "score", "多少分": "score", "好评": "score",
                 "开放时间": "open_time", "营业时间": "open_time", "几点": "open_time",
                 "地址": "address", "在哪儿": "address", "在哪里": "address", "位置": "address",
                 "电话": "phone", "联系方式": "phone", "号码": "phone"}

_LLM_SYS = """你是"城市旅游知识图谱"的查询意图解析器，输出且仅输出一个 JSON 对象（不要解释、不要代码块）。

【最重要】严禁编造任何事实数据。你只负责判断"要查哪个字段"，值一律留空交给数据库。
  例：问"故宫门票多少钱" → 应输出 fact:"门票"；绝不能自己写出价格。

字段含义：
  city       : 城市名，必须是下面【可选城市】里的原文；没有就 null
  season     : 只能是 春/夏/秋/冬/四季皆宜 之一；没有就 null
  categories : 数组，只能从【可选类别】里选，可多选。口语要映射：
               自然景观/山水/风光 → ["自然山地","水域景观"]；
               海边/海滨/海滩/温泉/湖 → ["水域景观"]；
               寺庙/教堂/道观 → ["宗教建筑"]；古镇/古村/老街 → ["古村镇"]
  entity     : 用户明确问到的**某一个具体景点全名**（如"故宫博物院"）。不确定就 null
  keyword    : 景点名里可能出现的**实词**（如"金沙滩""西湖"）。
               严禁填动词/数量词/疑问词（如 去、玩、有哪些、几个、什么）；
               用户只是在按条件筛选、并没有指定景点名时，必须填 null
  fact       : 只能取 门票/票价/评分/开放时间/地址/电话 之一；没问属性就 null

【可选季节】%s
【可选类别】%s
【可选城市】%s

输出示例：{"city":"青岛","season":"春","categories":["水域景观"],"keyword":null,"entity":null,"fact":null}"""


def _load_llm_cache():
    """免费额度有限，把解析结果落盘：重启后同一问题不再重复消耗额度。"""
    global _LLM_CACHE
    try:
        with open(_LLM_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _LLM_CACHE = {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass


def _save_llm_cache():
    try:
        with open(_LLM_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_LLM_CACHE, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


_load_llm_cache()


def _llm_available():
    """额度用尽或连续失败时进入冷却，冷却期内直接走规则解析，避免每次都干等超时。"""
    return bool(LLM_API_KEY) and time.time() >= _LLM_STAT["cooldown_until"]


def _llm_parse(q):
    """调用大模型解析问句。成功返回规范化后的 dict，失败返回 None（调用方回退规则解析）。"""
    if q in _LLM_CACHE:
        # 缓存命中也要计数：否则界面上"已解析 N 次"会一直显示 0，看起来像大模型没在工作
        _LLM_STAT["cache_hit"] += 1
        return _LLM_CACHE[q]
    if not _llm_available():
        return None
    try:
        seasons = [r["name"] for r in _graph.run(
            "MATCH (s:Season) RETURN s.name AS name ORDER BY name").data()]
        cats = [r["name"] for r in _graph.run(
            "MATCH (c:Category) RETURN c.name AS name ORDER BY name").data()]
        cities = _city_names()
    except Exception:
        seasons = ["春", "夏", "秋", "冬", "四季皆宜"]
        cats, cities = [], []

    sys_prompt = _LLM_SYS % ("、".join(seasons) or "春、夏、秋、冬、四季皆宜",
                             "、".join(cats), "、".join(cities))
    body = {"model": LLM_MODEL, "temperature": 0, "max_tokens": LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": q}]}
    req = urllib.request.Request(
        LLM_BASE_URL + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + LLM_API_KEY,
                 "Content-Type": "application/json"})
    _LLM_STAT["calls"] += 1
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
            data = json.load(r)
        raw = data["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as e:
        # 429 = 免费额度用尽：进冷却，别再硬试
        if e.code == 429:
            _LLM_STAT["cooldown_until"] = time.time() + 60
        _LLM_STAT["fail"] += 1
        _LLM_STAT["last_error"] = "HTTP %s" % e.code
        return None
    except Exception as e:
        _LLM_STAT["fail"] += 1
        _LLM_STAT["last_error"] = str(e)[:80]
        return None

    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        _LLM_STAT["fail"] += 1
        _LLM_STAT["last_error"] = "JSON 解析失败"
        return None
    _LLM_STAT["ok"] += 1
    out = _normalize_llm(obj, set(cities), set(seasons), set(cats), q)
    out["raw"] = obj
    _LLM_CACHE[q] = out
    _save_llm_cache()
    return out


def _extract_json(text):
    """模型偶尔会在 JSON 外包 ```json 围栏或前缀文字，这里做稳健提取。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            pass
    return None


# 模型常把"去/几个/有哪些"这类废词塞进 keyword，直接拿来检索会得到莫名其妙的兜底回答。
# 这里做一层过滤：太短或属于废词的关键词一律丢弃，让问句改走"按条件推荐"。
_KW_STOP = {"去", "玩", "看", "有", "哪", "哪里", "哪儿", "几个", "哪些", "什么", "地方", "景点",
            "景区", "好玩", "适合", "推荐", "想", "看看", "一下", "最好", "最", "热门", "免费",
            "便宜", "知名", "著名", "一日游", "游", "打卡", "值得", "逛", "转", "走走", "求",
            "给", "来", "说", "问", "查", "找", "要", "的", "了", "吗", "呢", "吧", "啊", "和"}


def _clean_kw(kw):
    """清洗关键词：去标点 → 逐条剔除废词片段 → 长度不足就作废。

    模型会把"推荐宗教""好玩吗"这类词整块塞进来，只做整词匹配拦不住，必须按片段剔除。
    """
    kw = re.sub(r"[\s，。？！、；：,.?!;:·\-—_“”\"'‘’（）()《》]+", "", kw or "").strip()
    if not kw:
        return ""
    for w in sorted(_KW_STOP, key=len, reverse=True):
        kw = kw.replace(w, "")
    return kw if len(kw) >= 2 else ""


def _exists_kw(kw):
    """该关键词在图谱里是否真的能查到景点——用来挡掉模型臆造的关键词/实体。"""
    if not kw:
        return False
    try:
        return _graph.run(_KEYWORD_COUNT, kw=kw, city=None).data()[0]["c"] > 0
    except Exception:
        return False


def _normalize_llm(obj, cities, seasons, cats, question=""):
    """把模型输出收敛到库里真实存在的取值：不认识的统统丢弃，绝不把脏值带进查询。

    这里的原则是**保守取用**：模型可以帮我们"多看出一层意图"，
    但凡是它凭空加的筛选条件（例如问门票却被它加上"四季皆宜"），一律不予采信。
    """
    def s(v):
        v = ("" if v is None else str(v)).strip()
        return v if v and v.lower() not in ("null", "none", "无") else ""

    city = s(obj.get("city"))
    # 模型可能输出"青岛市"，去掉行政区划后缀再校验
    for suf in ("特别行政区", "自治州", "地区", "市", "县", "盟"):
        if city.endswith(suf) and len(city) > len(suf):
            city = city[: -len(suf)]
            break
    if city not in cities:
        city = ""

    # 季节：必须是库里的值，且**问句里确实提到了季节**——防止模型给"故宫门票"安一个"四季皆宜"
    season = s(obj.get("season"))
    if season not in seasons or not _detect_season(question)[0]:
        season = ""

    # 类别：必须是库里的值，且问句里有对应依据（直接出现类别名，或命中口语同义词）
    raw_cats = obj.get("categories")
    if isinstance(raw_cats, str):
        raw_cats = re.split(r"[,，、;；/|]| and ", raw_cats)
    cat_list = []
    rule_cats = _detect_cats(question, set(cats) - {"其他"}) or []
    for c in (s(x) for x in (raw_cats or [])):
        if c and c in cats and c != "其他" and (c in question or c in rule_cats):
            cat_list.append(c)
    cat_list = list(dict.fromkeys(cat_list))          # 去重保序

    fact = s(obj.get("fact"))
    field = None
    for w, f in _FACT_BY_WORD.items():
        if w in fact:
            field = f
            break

    # 实体/关键词：库里查不到就当作没说，避免把臆造的词当成景点去检索
    entity = s(obj.get("entity"))
    if entity and (entity == city or not _exists_kw(entity)):
        entity = ""
    keyword = _clean_kw(obj.get("keyword"))
    if keyword and (keyword == city or not _exists_kw(keyword)):
        keyword = ""

    return {"city": city or None, "season": season or None,
            "categories": cat_list or None, "keyword": keyword or None,
            "entity": entity or None, "fact": field}


@app.route("/api/qa")
def qa():
    if not ensure_db():
        return _db_error()
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"ok": True, "items": [], "parsed": {},
                        "answer": "请输入问题，例如：青岛自然景观 / 故宫门票多少 / 北京有哪些景点"})
    limit = request.args.get("limit", 12)
    try:
        # ---- 大模型解析（失败会自动回退到下面的规则解析，绝不阻塞问答） ----
        ai = None
        if (request.args.get("ai", "1") or "1").lower() not in ("0", "false", "no", "off"):
            ai = _llm_parse(q)

        city = _longest_in(q, _city_names())               # 城市：全部从库里取，取最长匹配
        season, season_word = _detect_season(q)            # 季节：春天/春季 -> 春
        named_cat = _longest_in(q, _cat_names(), skip=("其他",))   # 直接说了类别全名
        cats = _detect_cats(q, set(_cat_names()) - {"其他"}) or []  # 口语映射到的类别（可多个）
        if named_cat and named_cat not in cats:
            cats.append(named_cat)
        cats = cats or None
        # 去掉城市/类别/同类口语/季节词后，剩下的才是"要找的景点名"
        kw = _keyword_of(q, city, named_cat, season_word)

        entity, field = None, None
        if ai:
            # AI 给出的值优先；AI 没给的（null）仍用规则解析的结果兜底
            city = ai["city"] or city
            season = ai["season"] or season
            if ai["categories"]:
                cats = ai["categories"]
            entity = ai["entity"]
            field = ai["fact"]
            # AI 在场时以它的判断为准：关键词取它给的（已过滤废词），没有就用它识别的实体；
            # 两者都为空说明这是个"按条件筛选"的问题，keyword 置空 -> 自然走推荐分支。
            kw = ai["keyword"] or entity or ""
            # 模型有时会把城市名带进关键词（"青岛金沙滩" -> "青岛金沙滩"），剔掉
            if kw and city and kw.startswith(city) and len(kw) > len(city):
                kw = kw[len(city):].strip() or kw
            kw = _clean_kw(kw) or ("" if not entity else kw)

        # 关键词去废词：整词就是废词则丢弃（"推荐几个宗教建筑" -> kw="几个"），
        # 能清洗出更干净的实词且库中确实查得到才替换。绝不无脑删字——
        # 否则"和平公园"里的"和"会被当成废词删掉，这是踩过的坑。
        if kw and kw in _KW_STOP:
            kw = ""
        elif kw:
            c = _clean_kw(kw)
            if c and c != kw and _exists_kw(c):
                kw = c

        cond = "、".join(filter(None, [
            city and ("城市=" + city),
            cats and ("类别=" + "/".join(cats)),
            season and ("季节=" + season)]))
        parsed = {"city": city, "season": season, "categories": cats,
                  "category": (cats[0] if cats and len(cats) == 1 else None), "keyword": kw,
                  "entity": entity, "fact": field, "ai": bool(ai),
                  "model": (LLM_MODEL if ai else None),
                  "ai_error": (None if ai else _LLM_STAT["last_error"])}

        def reply(ans, items, **extra):
            d = {"ok": True, "answer": ans, "items": items, "count": len(items), "parsed": parsed}
            d.update(extra)
            return jsonify(d)

        # ⓪ 大模型指认了"具体景点 + 要查哪个字段"：字段值一律由图谱给出，不用模型给的数
        if field and entity:
            it = _best_entity(entity, city)
            if it:
                label = _FACT_LABEL[field]
                val = it.get(field)
                if field == "ticket" and val not in ("", None):
                    val = "约 ¥%s 起" % val
                ans = ("【%s】%s：%s" % (it["name"], label, val)) if val not in ("", None) \
                    else ("【%s】的%s在原始数据中未采集到。" % (it["name"], label))
                return reply(ans, [it])

        # ① 事实型问句：问某个景点的属性，直接答（"故宫门票多少"）
        fact = _answer_fact(q, city, limit)
        if fact:
            return reply(fact[0], fact[1])

        # ② 有具体景点名 —— 先整句精确匹配，再拆成"城市+关键词"
        if kw:
            shown_kw, full = q, True
            items = _rows(_KEYWORD_CORE, kw=q, city=None, limit=limit)
            # 整句匹配必须命中"名字"才算（否则"杭州西湖"会靠地址命中苏堤，压掉真正的"西湖"）
            if not (items and q in (items[0]["name"] or "")):
                items = []
            total = _graph.run(_KEYWORD_COUNT, kw=q, city=None).data()[0]["c"] if items else 0
            scope = "全部地区"
            if not items:
                full, shown_kw = False, kw
                items = _rows(_KEYWORD_CORE, kw=kw, city=city, limit=limit)
                total = _graph.run(_KEYWORD_COUNT, kw=kw, city=city).data()[0]["c"]
                scope = ("%s内" % city) if city else "全部地区"
            if items:
                how = "整句" if full else "关键词"
                return reply("已识别【%s】：按%s「%s」在%s共找到 %d 个匹配景点，展示前 %d 个——"
                             % (cond or "关键词检索", how, shown_kw, scope, total, len(items)), items)
            # 名称没命中：**不再静默跳到全国**。改为留在原地解释，并给出该范围内的选择。
            if city or cats or season:
                more = _graph.run(_KEYWORD_COUNT, kw=kw, city=None).data()[0]["c"]
                head = "「%s」在%s没有匹配到。" % (shown_kw, city or "该条件下")
                if more:
                    head += "全国范围内有 %d 个含此关键词的景点，去掉城市限制再试即可。" % more
                dist = _cat_dist(city)
                if dist:
                    head += "%s的类别分布：%s。" % (city or "全库",
                             "、".join("%s %d" % (d["name"], d["cnt"]) for d in dist[:6]))
                rec = _rows(_RECO_CORE, city=city, season=season, cats=cats, limit=limit)
                return reply(head + "以下先按【%s】为您推荐——" % (cond or "不限"), rec)

            items = _rows(_SEARCH_CORE, q=q, limit=limit)
            return reply("未识别到城市/类别条件，以下按整句「%s」检索——" % q, items)

        # ③ 只有城市/类别/季节 -> 推荐（带"逐级放宽"，且始终不擅自跨出用户给的城市）
        if city or season or cats or ("推荐" in q) or ("适合" in q):
            items = _rows(_RECO_CORE, city=city, season=season, cats=cats, limit=limit)
            if items:
                return reply("已识别【%s】：共 %d 个匹配，按评分展示前 %d 个——"
                             % (cond or "不限", _reco_total(city, season, cats), len(items)), items)
            # 类别没命中 -> 用户问的是"某一类"，此时**引导比放宽更有用**：
            # 直接告诉他这个范围里根本没有这一类，并列出真实可选的类别
            if cats and _reco_total(city=city, season=None, cats=cats) == 0:
                dist = _cat_dist(city)
                tip = "%s没有「%s」类的景点。" % (city or "全库", "/".join(cats))
                if dist:
                    tip += "%s可选类别：%s。" % (
                        city or "全库",
                        "、".join("%s %d" % (d["name"], d["cnt"]) for d in dist[:8]))
                rec = _rows(_RECO_CORE, city=city, season=None, cats=None, limit=limit)
                return reply(tip + "以下先按【%s】为您推荐——"
                             % (("城市=" + city) if city else "不限"), rec)
            # 只有季节这类"弱条件"没命中时才放宽
            relax = []
            if season:
                relax = [("去掉季节", dict(city=city, season=None, cats=cats))]
                if city:
                    relax.append(("只保留城市", dict(city=city, season=None, cats=None)))
            for label, args in relax:
                more = _rows(_RECO_CORE, limit=limit, **args)
                if more:
                    return reply("按【%s】没有结果，已放宽（%s）后共 %d 个，展示前 %d 个——"
                                 % (cond or "不限", label, _reco_total(**args), len(more)), more)
            dist = _cat_dist(city)
            tip = "按【%s】没有匹配到。" % (cond or "不限")
            if dist:
                tip += "%s可选类别：%s。换个类别试试？" % (
                    city or "全库", "、".join("%s %d" % (d["name"], d["cnt"]) for d in dist[:8]))
            return reply(tip, [])

        # ④ 什么都没识别出来 -> 整句检索兜底
        items = _rows(_SEARCH_CORE, q=q, limit=limit)
        return reply("未识别到明确的城市/类别/季节条件，以下按关键词检索——", items)
    except Exception as e:
        return _err(e)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


# ---------------- 前端页面 ----------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>城市旅游知识图谱应用</title>
<!-- 优先用本地 ECharts，断网也能画图；缺失时才回退 CDN -->
<script src="/static/echarts.min.js"></script>
<style>
 *{box-sizing:border-box;margin:0;padding:0}
 body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f4f6f9;color:#1f2430;font-size:14px}

 /* ---------- 顶部 ---------- */
 header{background:linear-gradient(135deg,#1f6feb,#1fab89);color:#fff;padding:20px 28px;
   box-shadow:0 2px 8px rgba(0,0,0,.08)}
 header h1{font-size:21px;font-weight:600;letter-spacing:.5px}
 header p{opacity:.9;font-size:12.5px;margin-top:6px}
 .tabs{display:flex;gap:6px;padding:10px 24px;background:#fff;border-bottom:1px solid #e5e7eb;
   flex-wrap:wrap;position:sticky;top:0;z-index:20}
 .tabs button{border:1px solid transparent;background:#f2f5f9;color:#4a5364;padding:9px 18px;
   border-radius:20px;cursor:pointer;font-size:14px;transition:.15s}
 .tabs button:hover{background:#e8eef7}
 .tabs button.active{background:#1f6feb;color:#fff;box-shadow:0 2px 6px rgba(31,111,235,.28)}

 .panel{display:none;padding:20px 24px;max-width:1180px;margin:0 auto}
 .panel.active{display:block}

 /* ---------- 控件 ---------- */
 .row{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;align-items:center}
 input,select{padding:9px 13px;border:1px solid #d6dce5;border-radius:8px;font-size:14px;background:#fff}
 input:focus,select:focus{outline:none;border-color:#1f6feb;box-shadow:0 0 0 3px rgba(31,111,235,.12)}
 button.go{background:#1f6feb;color:#fff;border:none;border-radius:8px;padding:9px 20px;
   cursor:pointer;font-size:14px;transition:.15s}
 button.go:hover{background:#1a5fd0} button.go:disabled{background:#a9c2e9;cursor:progress}

 /* ---------- 提示条 ---------- */
 .hint{background:#f0f6ff;border:1px solid #d6e6ff;color:#2b4a7d;padding:9px 13px;
   border-radius:8px;font-size:12.5px;margin-bottom:14px;line-height:1.6}
 .summary{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;
   background:#eaf7f1;border:1px solid #bfe8d5;color:#0a6b45;padding:9px 14px;
   border-radius:8px;margin-bottom:12px;font-size:13px}
 .summary .n{color:#087f5b;font-weight:600}
 .warn{background:#fff6e8;border:1px solid #ffdca8;color:#8a5300;padding:10px 14px;
   border-radius:8px;margin-bottom:12px;font-size:13px}
 .muted{color:#8a94a6;font-size:12px}
 .empty{background:#fff;border:1px dashed #d6dce5;border-radius:12px;padding:34px;
   text-align:center;color:#8a94a6;font-size:13px}

 /* ---------- 分页条 ---------- */
 .pager{display:flex;justify-content:center;align-items:center;gap:6px;flex-wrap:wrap;margin:18px 0 4px}
 .pager .pg{min-width:34px;height:34px;padding:0 10px;border:1px solid #d8e0ea;background:#fff;
   color:#384052;border-radius:8px;cursor:pointer;font-size:13px;transition:.15s}
 .pager .pg:hover:not(:disabled){border-color:#1f6feb;color:#1f6feb;background:#f4f8ff}
 .pager .pg.cur{background:#1f6feb;border-color:#1f6feb;color:#fff;font-weight:600}
 .pager .pg:disabled{opacity:.42;cursor:not-allowed}
 .pager .pgap{color:#9aa4b2;padding:0 2px;user-select:none}
 .pager .pginfo{margin-left:8px;color:#8a94a6;font-size:12px}

 /* ---------- 问答面板 ---------- */
 .aibar{display:flex;align-items:center;gap:8px;padding:8px 13px;border-radius:8px;
   background:#eef6ff;border:1px solid #d6e6ff;color:#2b4a7d;font-size:13px;margin-bottom:12px}
 .aibar .dot{width:8px;height:8px;border-radius:50%;background:#c3ccd8;flex:none}
 .aibar.on .dot{background:#22c55e;box-shadow:0 0 0 3px rgba(34,197,94,.18)}
 .aibar.wait .dot{background:#f59f00;box-shadow:0 0 0 3px rgba(245,159,0,.18)}
 .aibar.off .dot{background:#c3ccd8}
 .aibar b{font-weight:600}
 .qbox{display:flex;gap:10px}
 .qbox input{flex:1;padding:12px 14px;border:1px solid #d8e0ea;border-radius:10px;font-size:15px;
   background:#fff;transition:.15s}
 .qbox input:focus{outline:none;border-color:#1f6feb;box-shadow:0 0 0 3px rgba(31,111,235,.12)}
 .qbox button.go{padding:0 26px;border-radius:10px;font-size:15px}
 .qegs{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px}
 .qeg{display:inline-block;padding:6px 12px;background:#fff;border:1px solid #dfe6ef;border-radius:16px;
   color:#41506a;font-size:12.5px;cursor:pointer;transition:.15s;user-select:none}
 .qeg:hover{border-color:#1f6feb;color:#1f6feb;background:#f5f9ff}
 .qres-empty{margin-top:16px;text-align:center;color:#9aa4b2;font-size:13px;padding:26px 0}
 .pcond{display:flex;flex-wrap:wrap;gap:6px;margin:-4px 0 12px}
 .pcond .pc{display:inline-block;background:#f2f6fb;border:1px solid #dfe6ef;border-radius:14px;
   padding:3px 10px;font-size:12px;color:#41506a}
 .pcond .pc b{color:#1f6feb;font-weight:600}

 /* ---------- 结果卡片网格 ---------- */
 .results{display:grid;grid-template-columns:repeat(auto-fill,minmax(272px,1fr));gap:16px}
 .spot{background:#fff;border:1px solid #e6eaf0;border-radius:12px;overflow:hidden;
   display:flex;flex-direction:column;transition:.18s}
 .spot:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(20,40,80,.10);border-color:#cfe0f7}
 .spot .pic{position:relative;height:150px;flex:none;background:#eef1f6}
 .spot .pic img{width:100%;height:100%;object-fit:cover;display:block}
 .spot .noimg{width:100%;height:100%;display:flex;align-items:center;justify-content:center;
   background:linear-gradient(135deg,#f2f6fb,#e4ebf5)}
 /* 无图时用景点名首字做占位，不依赖 emoji 字体，各系统表现一致 */
 .spot .ph{font-size:42px;font-weight:700;color:#c6d2e2;user-select:none}
 .spot .score{position:absolute;right:8px;top:8px;background:rgba(255,255,255,.94);color:#e8590c;
   font-weight:700;font-size:12.5px;padding:3px 9px;border-radius:14px;
   box-shadow:0 1px 4px rgba(0,0,0,.15)}
 .spot .body{padding:12px 14px 14px;display:flex;flex-direction:column;gap:8px;flex:1}
 .spot h3{font-size:15px;font-weight:600;line-height:1.4}
 .spot h3 a{color:#16324f;text-decoration:none}
 .spot h3 a:hover{color:#1f6feb}
 .chips{display:flex;flex-wrap:wrap;gap:5px}
 .chip{font-size:11.5px;padding:2px 8px;border-radius:5px;background:#eef2f8;color:#4a5568}
 .chip.city{background:#e7f1ff;color:#1f6feb}
 .chip.cat{background:#e8f7f0;color:#0b7a52}
 .chip.season{background:#fff0f6;color:#a61e4d}
 .desc{font-size:12.5px;color:#5b6577;line-height:1.65;display:-webkit-box;
   -webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
 .foot{margin-top:auto;padding-top:2px;display:flex;justify-content:space-between;
   align-items:center;gap:8px}
 .lnk{font-size:12px;color:#1f6feb;text-decoration:none;border:1px solid #cfe3ff;
   border-radius:14px;padding:3px 11px;background:#f5f9ff}
 .lnk:hover{background:#e6f0ff}

 /* ---------- 统计看板 ---------- */
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
 .stat{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;text-align:center}
 .stat b{font-size:26px;color:#1f6feb}
 #graphbox{width:100%;height:520px}
 .chartbox{background:#fff;border:1px solid #e5e7eb;border-radius:10px}
</style></head>
<body>
<header><h1>城市旅游领域知识图谱应用</h1><p>Neo4j · BERT-CRF(NER) · BERT-RE · ECharts ｜ 实验四交付</p></header>
<div id="dbwarn" class="warn" style="display:none;margin:12px 24px">⚠️ Neo4j 未连接：请确认数据库已启动、环境变量已配置。页面会自动重试。</div>
<div class="tabs">
 <button data-t="search" class="active">🔍 景点检索</button>
 <button data-t="reco">💡 智能推荐</button>
 <button data-t="gpanel">🕸 子图可视化</button>
 <button data-t="board">📊 统计看板</button>
 <button data-t="qa">💬 问答</button>
</div>

<div id="search" class="panel active">
 <div class="hint">试试：故宫 / 杭州 / 西湖 / 温泉；也支持「城市+景点」连写，如 <b>青岛金沙滩</b></div>
 <div class="row"><input id="sq" style="flex:1" placeholder="输入景点名、城市或地址关键词">
 <button class="go" onclick="doSearch()">搜索</button></div>
 <div id="sres"></div>
</div>

<div id="reco" class="panel">
 <div class="hint">括号里的数字会<b>跟随其它条件联动</b>：选定「北京」后，季节与类别显示的就是北京有多少个。不选即全国。</div>
 <div class="row">
  <select id="rcity" data-ph="城市（不限）"><option value="">城市（不限）</option></select>
  <select id="rseason" data-ph="季节（不限）"><option value="">季节（不限）</option></select>
  <select id="rcat" data-ph="类别（不限）"><option value="">类别（不限）</option></select>
  <button class="go" onclick="doReco()">推荐</button>
 </div>
 <div id="rres"></div>
</div>

<div id="gpanel" class="panel">
 <div class="hint">输入<b>城市名</b>看该城市的景点子图（如 青岛），或输入<b>景点名</b>看它的 1 跳邻居（如 故宫）。
  箭头方向：城市 → 景点 → 类别 / 季节；可拖动节点、滚轮缩放。</div>
 <div class="row"><input id="gq" style="flex:1" placeholder="输入城市名或景点名，如：青岛 / 故宫 / 西湖"
   onkeydown="if(event.key==='Enter'){doGraph(false);}">
 <button class="go" onclick="doGraph(false)">查询子图</button>
 <button class="go" onclick="doGraph(true)">看北京示例</button></div>
 <div id="gsum"></div>
 <div id="graphbox" class="chartbox"></div>
</div>

<div id="board" class="panel"><div id="boardbox"></div></div>

<div id="qa" class="panel">
 <div class="aibar" id="llmbar"><span class="dot" id="llmdot"></span><span id="llmtxt">正在检测大模型…</span></div>

 <div class="qbox">
  <input id="qq" placeholder="用自然语言提问，例如：青岛有哪些适合春天去的海边景点" autocomplete="off"
         onkeydown="if(event.key==='Enter'){doQA();}">
  <button class="go" id="qbtn" onclick="doQA()">提问</button>
 </div>

 <div class="qegs">
  <span class="qeg" onclick="ask('青岛有哪些适合春天去的海边景点')">🌊 青岛春天的海边景点</span>
  <span class="qeg" onclick="ask('青岛金沙滩好玩吗')">🏖 青岛金沙滩</span>
  <span class="qeg" onclick="ask('故宫门票多少钱')">🎫 故宫门票多少</span>
  <span class="qeg" onclick="ask('西湖开放时间')">🕒 西湖开放时间</span>
  <span class="qeg" onclick="ask('北京有哪些景点')">🏙 北京有哪些景点</span>
  <span class="qeg" onclick="ask('推荐几个宗教建筑')">🛕 推荐宗教建筑</span>
 </div>

 <div id="qres"></div>
</div>

<script>
const $=s=>document.querySelector(s);
const ESC=c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]);
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,ESC);
const charts={};

function getChart(id){
  const el=document.getElementById(id); if(!el) return null;
  const old=echarts.getInstanceByDom(el); if(old) old.dispose();
  const c=echarts.init(el); charts[id]=c; return c;
}
function ensureEcharts(cb){
  if(typeof echarts!=='undefined'){ cb(); return; }
  // 本地文件没加载出来才回退 CDN；任何一路失败/超时都必须给出可见提示，不能静默卡住
  const s=document.createElement('script');
  let done=false;
  const fail=(msg)=>{ if(done) return; done=true; alert(msg); };
  s.src='https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js';
  s.onload=()=>{ if(!done){ done=true; cb(); } };
  s.onerror=()=>fail('图表库加载失败：请确认 app/static/echarts.min.js 存在，或本机可联网后刷新页面。');
  setTimeout(()=>fail('图表库加载超时（8s）：请确认 app/static/echarts.min.js 存在后刷新页面。'),8000);
  document.head.appendChild(s);
}
async function api(u){
  // 关键：fetch 本身 reject（断连/服务未起）也必须被捕获，否则调用方会永远停在"加载中"
  const ctl=new AbortController();
  const timer=setTimeout(()=>ctl.abort(),15000);
  try{
    const r=await fetch(u,{cache:'no-store',signal:ctl.signal});
    let t='';
    try{ t=await r.text(); }catch(e){ t=''; }
    if(!r.ok) return {ok:false,error:'服务返回 HTTP '+r.status};
    try{ return JSON.parse(t); }
    catch(e){ return {ok:false,error:'响应解析失败：'+String(t).slice(0,80)}; }
  }catch(e){
    const msg=(e&&e.name==='AbortError')
      ? '请求超时（15s）：请检查 Neo4j 与后端服务是否还在运行'
      : '网络请求失败：'+((e&&e.message)||e);
    return {ok:false,error:msg};
  }finally{ clearTimeout(timer); }
}

document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); $('#'+b.dataset.t).classList.add('active');
  if(b.dataset.t==='board') loadBoard();
  setTimeout(()=>{for(const k in charts) try{charts[k].resize()}catch(e){}},30);
});

let optsReady=false;
async function checkHealth(){
  const j=await api('/api/health');
  $('#dbwarn').style.display = j.db ? 'none':'block';
  // 只在"首次连通/断后恢复"时加载一次，避免每 6 秒都重算一遍联动计数
  if(j.db && !optsReady){ optsReady=true; loadOpts(); }
  if(!j.db) optsReady=false;
}
setInterval(checkHealth,6000);
checkHealth();

// 大模型状态：让界面上明确显示"AI 是否真的在用"，而不是口头说接入了
async function loadLlm(){
  const bar=$('#llmbar'), el=$('#llmtxt'); if(!bar||!el) return;
  try{
    const j=await api('/api/llm_status');
    if(!j.enabled){
      bar.className='aibar off';
      el.innerHTML='⚠️ <b>未配置大模型</b>　当前使用本地规则解析，功能不受影响';
      return;
    }
    // 缓存命中也计入"已解析"，否则反复问同一句时界面会一直显示 0，像是没在工作
    const total=(j.calls||0)+(j.cache_hit||0);
    bar.className='aibar '+(j.available?'on':'wait');
    el.innerHTML='🤖 <b>大模型 '+esc(j.model)+'</b>　'
      + (j.available?'已就绪':'额度冷却中（暂时改用本地规则解析）')
      +'　｜　本次会话已解析 <b>'+total+'</b> 次'
      +'（模型调用 '+(j.calls||0)+' · 命中缓存 '+(j.cache_hit||0)+' · 失败 '+(j.fail||0)+'）'
      +'　｜　缓存 '+j.cached+' 条'
      + (j.last_error?'　｜　最近错误：'+esc(j.last_error):'');
  }catch(e){ el.textContent='状态获取失败'; }
}
loadLlm();

// 任一筛选条件变化 -> 其余下拉框的计数随之刷新
['#rcity','#rseason','#rcat'].forEach(sel=>{
  $(sel).addEventListener('change',()=>{ loadOpts(); });
});

function fillOpts(j){
  if(!j||!j.ok) return;
  const fill=(sel,arr,unit)=>{
    const el=$(sel);
    // data-ph 缺失时兜底，否则第一项会渲染成字面量 "undefined"
    const ph=el.dataset.ph||el.getAttribute('data-ph')||'（不限）';
    el.innerHTML='<option value="">'+esc(ph)+'</option>';
    arr.forEach(x=>{el.insertAdjacentHTML('beforeend',
      '<option value="'+esc(x.name)+'">'+esc(x.name)+'（'+x.cnt+unit+'）</option>');});
  };
  fill('#rcity',j.cities,'个景点');
  fill('#rseason',j.seasons,'个景点');
  fill('#rcat',j.categories,'个景点');
}
// 括号里的数字随其它下拉框联动刷新：选了北京之后，季节/类别显示的就是"北京有多少个"
async function loadOpts(){
  const c=$('#rcity').value||'', s=$('#rseason').value||'', cat=$('#rcat').value||'';
  const j=await api('/api/opts?city='+encodeURIComponent(c)
                   +'&season='+encodeURIComponent(s)+'&category='+encodeURIComponent(cat));
  if(!j||!j.ok) return;
  fillOpts(j);
  // 重新填充后把用户已选的值放回去（若该选项在新条件下计数为 0 会自动回落"不限"）
  $('#rcity').value=c; $('#rseason').value=s; $('#rcat').value=cat;
}

// 只放行 http/https，避免数据库里的脏值变成 javascript: 之类的危险链接
const safeUrl=u=>{const s=String(u==null?'':u).trim();return /^https?:\/\//i.test(s)?s:'';};
// 图片加载失败（图床防盗链/断网）时换成占位块，不留破图
function imgFail(el, ch){ el.parentNode.innerHTML='<div class="noimg"><span class="ph">'+esc(ch||'景')+'</span></div>'; }
// 无图占位：取景点名首个汉字
const phChar = name => { const m=String(name||'').match(/[\u4e00-\u9fa5]/); return m?m[0]:(String(name||'景')[0]||'景'); };

// 每个结果面板记住"上一次请求的 URL"，翻页时只需追加 &page=N
const _req = {};

// 页码窗口：首页、末页、当前页左右各 2 页，中间用 … 折叠
function pagerHTML(sel, pg){
  // ⚠️ 必须判空：只有检索/推荐返回 page 字段，问答接口没有（传进来是 undefined）。
  //    之前在这里直接解构 undefined 会抛 TypeError，导致问答卡片永远停在"思考中…"。
  if(!pg || !pg.pages || pg.pages<=1) return '';
  const {page,pages,total,size}=pg;
  let h='<div class="pager">';
  h+='<button class="pg"'+(page<=1?' disabled':'')+' onclick="gotoPage(\''+sel+'\','+(page-1)+')">上一页</button>';
  const win=[]; const W=2;
  for(let i=1;i<=pages;i++){
    if(i===1||i===pages||Math.abs(i-page)<=W) win.push(i);
    else if(win[win.length-1]!=='…') win.push('…');
  }
  win.forEach(i=>{
    h += (i==='…')
      ? '<span class="pgap">…</span>'
      : '<button class="pg'+(i===page?' cur':'')+'" onclick="gotoPage(\''+sel+'\','+i+')">'+i+'</button>';
  });
  h+='<button class="pg"'+(page>=pages?' disabled':'')+' onclick="gotoPage(\''+sel+'\','+(page+1)+')">下一页</button>';
  h+='<span class="pginfo">第 '+pg.from+'-'+pg.to+' 条 / 共 '+total+' 条（每页 '+size+' 条）</span>';
  h+='</div>';
  return h;
}

async function gotoPage(sel,page){
  const st=_req[sel];
  if(!st||st.busy) return;
  st.busy=true;
  const el=$(sel);
  el.innerHTML='<div class="card muted" style="background:#fff">加载中…</div>';
  try{
    const sep = st.url.indexOf('?')>=0 ? '&' : '?';
    const j = await api(st.url+sep+'page='+page+'&size='+st.size);
    renderItems(sel,j,{url:st.url,size:st.size});
    // 翻页后回到结果区顶部，否则停在上一页的滚动位置会以为没变化
    el.scrollIntoView({behavior:'smooth',block:'start'});
  } finally { st.busy=false; }
}

function renderItems(sel,j,opt){
  opt = opt || {};
  const el=$(sel);
  let h='';
  if(!j||!j.ok){el.innerHTML='<div class="warn">'+esc((j&&j.error)||'请求失败')+'</div>';return;}
  if(opt.url) _req[sel]={url:opt.url,size:opt.size||30,busy:false};
  const items=j.items||[];
  const pg=j.page;
  if(j.answer){
    const cnt = pg
      ? ('第 '+pg.from+'-'+pg.to+' 条 / 共 '+pg.total+' 条')
      : (items.length+' 条结果');
    h+='<div class="summary"><span>'+esc(j.answer)+'</span>'
      +(items.length?'<span class="n">'+cnt+'</span>':'')+'</div>';
    // 命中大模型解析时把"它到底理解了什么"直接列出来，便于演示与排查
    if(j.parsed&&j.parsed.ai){
      const p=j.parsed;
      const chips=[];
      if(p.city)       chips.push(['城市',p.city]);
      if(p.categories) chips.push(['类别',(p.categories||[]).join('/')]);
      if(p.season)     chips.push(['季节',p.season]);
      if(p.entity)     chips.push(['景点',p.entity]);
      if(p.keyword&&p.keyword!==p.entity) chips.push(['关键词',p.keyword]);
      if(p.fact)       chips.push(['查',{ticket:'门票',score:'评分',open_time:'开放时间',
                                        address:'地址',phone:'电话'}[p.fact]||p.fact]);
      h+='<div class="pcond"><span class="pc">🤖 大模型理解</span>';
      if(chips.length) chips.forEach(c=>{ h+='<span class="pc">'+esc(c[0])+' <b>'+esc(c[1])+'</b></span>'; });
      else             h+='<span class="pc">（未识别出限定条件，按整句检索）</span>';
      h+='</div>';
      h+='<div class="muted" style="margin:-6px 0 12px">事实数据全部来自图数据库，模型只负责把话翻译成查询条件</div>';
    }
    // 简介缺失是源数据问题（全库仅 50.5% 有），在这里统一说明一次，避免每张卡片都挂一行灰字
    const noIntro=items.filter(x=>!x.intro).length;
    if(noIntro) h+='<div class="muted" style="margin:-4px 0 12px">'
      +'其中 '+noIntro+' 条在原始数据中未采集到简介（完整度见「统计看板」）</div>';
  }
  if(items.length===0){el.innerHTML=h+'<div class="empty">无结果，试试其它关键词或放宽条件。</div>';return;}
  // 分页条只在结果区底部放一条，上方不放（保持检索/推荐面板原有排版）
  h+='<div class="results">';
  items.forEach(it=>{
    // 只保留官网与图片（第三方站点链接大多已失效，不再展示）
    const img=safeUrl(it.image), site=safeUrl(it.website);
    h+='<div class="spot">';
    // 图片区
    h+='<div class="pic">';
    const ph=phChar(it.name);
    if(img) h+='<img src="'+esc(img)+'" alt="'+esc(it.name)+'" loading="lazy"'
            +' referrerpolicy="no-referrer" onerror="imgFail(this,\''+esc(ph)+'\')">';
    else    h+='<div class="noimg"><span class="ph">'+esc(ph)+'</span></div>';
    if(it.score!==''&&it.score!=null) h+='<span class="score">★ '+esc(it.score)+'</span>';
    h+='</div>';
    // 文字区
    h+='<div class="body">';
    h+='<h3>'+(site
        ? '<a href="'+esc(site)+'" target="_blank" rel="noopener noreferrer" title="打开官网">'+esc(it.name)+'</a>'
        : esc(it.name))+'</h3>';
    h+='<div class="chips">';
    if(it.city)     h+='<span class="chip city">📍 '+esc(it.city)+'</span>';
    if(it.category) h+='<span class="chip cat">'+esc(it.category)+'</span>';
    if(it.season)   h+='<span class="chip season">'+esc(it.season)+'</span>';
    if(it.district) h+='<span class="chip">'+esc(it.district)+'</span>';
    h+='</div>';
    if(it.intro) h+='<div class="desc">'+esc(it.intro)+'…</div>';
    // 页脚只在确实有内容时渲染，避免没官网的卡片挂一个孤零零的占位符
    const hasSite=!!site, hasTicket=(it.ticket!==''&&it.ticket!=null);
    if(hasSite||hasTicket){
      h+='<div class="foot">';
      if(hasSite) h+='<a class="lnk" href="'+esc(site)+'" target="_blank" rel="noopener noreferrer">🔗 官网</a>';
      if(hasTicket) h+='<span class="muted"'+(hasSite?'':' style="margin-left:auto"')+'>🎫 ¥'+esc(it.ticket)+'</span>';
      h+='</div>';
    }
    h+='</div></div>';   // /body /spot
  });
  h+='</div>';              // /results
  h+= pagerHTML(sel,pg);    // 分页条：只在结果区底部放一条
  el.innerHTML=h;
}

async function doSearch(){
  const q=$('#sq').value.trim(); if(!q){alert('请输入关键词');return;}
  $('#sres').innerHTML='<div class="card muted">查询中…</div>';
  const url='/api/search?q='+encodeURIComponent(q);
  renderItems('#sres',await api(url),{url:url,size:20});
}
async function doReco(){
  const c=$('#rcity').value,s=$('#rseason').value,cat=$('#rcat').value;
  $('#rres').innerHTML='<div class="card muted">查询中…</div>';
  const url='/api/recommend?city='+encodeURIComponent(c)+
            '&season='+encodeURIComponent(s)+'&category='+encodeURIComponent(cat);
  renderItems('#rres',await api(url),{url:url,size:30});
}
// 点击示例问题：填入输入框并直接提问
function ask(q){
  $('#qq').value=q;
  doQA();
}

async function doQA(){
  const q=$('#qq').value.trim();
  if(!q){ $('#qq').focus(); return; }
  const btn=$('#qbtn'); const old=btn?btn.textContent:'';
  $('#qres').innerHTML='<div class="qres-empty">🤖 正在理解你的问题…</div>';
  if(btn){ btn.textContent='理解中…'; btn.disabled=true; }
  try{
    const j=await api('/api/qa?q='+encodeURIComponent(q));
    renderItems('#qres',j);   // 问答是"给一句话答案"，不分页；相关的 N 条只是佐证
  } finally {
    if(btn){ btn.textContent=old||'提问'; btn.disabled=false; }
    loadLlm();                 // 每次提问后刷新计数，否则状态条永远停在页面加载时的数值
  }
}
function doGraph(sample){
  ensureEcharts(()=>_doGraph(sample));
}
function busy(on){
  document.querySelectorAll('#gpanel button.go').forEach(b=>{
    if(on){ if(!b.dataset.old) b.dataset.old=b.textContent; b.textContent='查询中…'; b.disabled=true; }
    else if(b.dataset.old){ b.textContent=b.dataset.old; delete b.dataset.old; b.disabled=false; }
  });
}
async function _doGraph(sample){
  const box=$('#graphbox'), bar=$('#gsum');
  busy(true);
  try{
    const q=sample?'':$('#gq').value.trim();
    box.innerHTML='<div class="muted" style="padding:20px">加载中…</div>';
    if(bar) bar.innerHTML='';
    const j=await api('/api/graph?name='+encodeURIComponent(q));
    if(!j||!j.ok){ box.innerHTML='<div class="warn">查询失败：'+esc((j&&j.error)||'未知错误')+'</div>'; return; }
    // 说明文字常显：让用户明确知道"现在画的是城市子图还是某个景点的子图"
    if(bar&&j.answer) bar.innerHTML='<div class="summary"><span>'+esc(j.answer)+'</span>'
      +'<span class="n">'+j.count.nodes+' 节点 / '+j.count.edges+' 关系</span></div>';
    if((j.nodes||[]).length===0){
      box.innerHTML='<div class="warn">'+esc(j.answer||'无结果')+'</div>'; return; }
    // 注意：graph 系列必须先声明 categories，legend.data 才能与之对应；
    // 否则 legend 会把整条 graph 数据过滤掉，表现为"canvas 有了但一片空白"。
    const CATS=['City','Scenic','Season','Category'];
    const color={City:'#1f6feb',Scenic:'#1fab89',Season:'#f59f00',Category:'#9c36b5'};
    // 节点名可能很长（如"青岛扎啤美食文化广场QingDao ZhaPi…"），显示时截断，避免文字互相压叠
    const short=s=>{s=String(s||'');return s.length>12?s.slice(0,11)+'…':s;};
    const data=(j.nodes||[]).map(n=>({name:n.name,value:1,
        short:short(n.name),
        category:Math.max(0,CATS.indexOf(n.label)),
        symbolSize:n.label==='Scenic'?16:24}));
    const links=(j.edges||[]).map(e=>({source:e.source,target:e.target,
        label:{show:true,formatter:e.type,fontSize:9},lineStyle:{curveness:0.1}}));
    // legend 只列出本图真实出现的类别（原先是把类别名当系列名传，才触发上面的空图问题）
    const used=CATS.filter(l=>(j.nodes||[]).some(n=>n.label===l));
    // 先算好 option 再清空容器：绘图出错时不会留下白板，而是显示错误原因
    const opt={tooltip:{formatter:p=>p.dataType==='edge'?esc(p.data.type||''):esc(p.data.name||'')},
      legend:[{data:used,top:0}],
      series:[{type:'graph',layout:'force',roam:true,draggable:true,
        categories:CATS.map(l=>({name:l,itemStyle:{color:color[l]||'#888'}})),
        label:{show:true,fontSize:11,position:'right',formatter:p=>(p.data&&p.data.short)||p.name},
        labelLayout:{hideOverlap:true},
        edgeLabel:{show:true,formatter:p=>p.data.type,fontSize:9},
        edgeSymbol:['none','arrow'],edgeSymbolSize:8,
        force:{repulsion:340,edgeLength:110,gravity:0.06},data,links}],
      backgroundColor:'#fff'};
    box.innerHTML='';
    const ch=getChart('graphbox');
    if(!ch){ box.innerHTML='<div class="warn">绘图容器初始化失败。</div>'; return; }
    try{ ch.setOption(opt,true); }
    catch(e){ box.innerHTML='<div class="warn">绘图失败：'+esc((e&&e.message)||e)+'</div>'; return; }
    // 若在隐藏面板里初始化过（宽高为 0），切回本页时补一次 resize
    setTimeout(()=>{ try{ ch.resize(); }catch(e){} },30);
  }catch(e){
    box.innerHTML='<div class="warn">查询异常：'+esc((e&&e.message)||e)+'</div>';
  }finally{
    busy(false);
  }
}
async function loadBoard(){
  const box=$('#boardbox'); box.innerHTML='<div class="card muted">加载中…</div>';
  const j=await api('/api/stats');
  if(!j||!j.ok){box.innerHTML='<div class="warn">'+esc(j&&j.error||'失败')+'</div>';return;}
  let h='<div class="grid">';
  h+='<div class="stat"><b>'+j.total+'</b><br>节点总数</div>';
  h+='<div class="stat"><b>'+j.rel_total+'</b><br>关系总数</div>';
  for(const k in j.labels){ if(j.labels[k]>0) h+='<div class="stat"><b>'+j.labels[k]+'</b><br>'+esc(k)+'</div>'; }
  h+='</div><div class="row" style="margin-top:14px">'
    +'<div style="flex:1"><div id="cc" style="height:280px" class="chartbox"></div></div>'
    +'<div style="flex:1"><div id="cs" style="height:280px" class="chartbox"></div></div></div>'
    +'<div style="margin-top:14px"><div id="ct" style="height:300px" class="chartbox"></div></div>';
  // 字段完整度：解释"为什么有些景点没有简介/地址/评分"
  if(j.completeness&&j.completeness.length){
    h+='<div class="hint" style="margin-top:14px">字段完整度（共 '+j.scenic_total+' 个景点，'
      +'数值为该字段非空的比例）—— 数据来自爬虫，本身就有缺失，并非前端未渲染</div>'
      +'<div style="margin-top:8px"><div id="cf" style="height:300px" class="chartbox"></div></div>';
  }
  box.innerHTML=h;
  ensureEcharts(()=>{
    getChart('cc').setOption({title:{text:'景点类别分布',left:'center'},tooltip:{},
      grid:{left:40,right:20,bottom:80},
      xAxis:{type:'category',data:j.cats.map(x=>x.name),axisLabel:{rotate:35,fontSize:10}},
      yAxis:{},series:[{type:'bar',data:j.cats.map(x=>x.cnt),itemStyle:{color:'#1fab89'}}]});
    getChart('cs').setOption({title:{text:'适宜季节分布',left:'center'},tooltip:{},
      legend:{bottom:0},series:[{type:'pie',radius:'55%',
        data:j.seasons.map(x=>({name:x.name,value:x.cnt}))}]});
    getChart('ct').setOption({title:{text:'Top10 城市景点数',left:'center'},tooltip:{},
      grid:{left:60,right:30,bottom:80},
      xAxis:{type:'category',data:j.top_cities.map(x=>x.name),axisLabel:{rotate:30,fontSize:10}},
      yAxis:{},series:[{type:'bar',data:j.top_cities.map(x=>x.cnt),itemStyle:{color:'#1f6feb'}}]});
    if(j.completeness&&j.completeness.length){
      getChart('cf').setOption({title:{text:'字段完整度（%）',left:'center'},tooltip:{},
        grid:{left:50,right:30,bottom:70},
        xAxis:{type:'category',data:j.completeness.map(x=>x.label),axisLabel:{rotate:25,fontSize:10}},
        yAxis:{max:100},
        series:[{type:'bar',data:j.completeness.map(x=>x.pct),
          itemStyle:{color:'#9c36b5'},
          label:{show:true,position:'top',fontSize:10,formatter:'{c}%'}}]});
    }
  });
  loadOpts();
}
// 回车提交
[['#sq',doSearch],['#gq',()=>doGraph(false)],['#qq',doQA]].forEach(([sel,fn])=>{
  $(sel).addEventListener('keydown',e=>{if(e.key==='Enter')fn();});
});
</script>
</body></html>"""


@app.after_request
def _no_cache(resp):
    """页面与接口一律不缓存：否则改完代码后浏览器仍用旧页面，会出现"改了没生效/点了没反应"。"""
    if resp.mimetype in ("text/html", "application/json"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("[城市旅游知识图谱应用] http://localhost:%d  (Neo4j: %s)" %
          (port, "已连接" if _db_ok else (_last_err or "未连接")))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
