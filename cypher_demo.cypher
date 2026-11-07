// ============================================================
// Neo4j Cypher 操作示例
// 常用 Cypher 示例：约束、统计、检索、推荐、子图
//
// 用法：打开 http://localhost:7474 ，逐条粘贴执行；
//      或者在 Neo4j 安装目录的 bin 下执行
//      cypher-shell -u <user> -p <password> -f 本文件路径
//
// 本图谱标签：City(城市) / Scenic(景点) / Season(季节) / Category(类别)
// 关系：HAS_SPOT / BEST_SEASON / BELONGS_TO
// ============================================================


// ------------------------------------------------------------
// ① 建立一个节点（带属性）
//    文档语法：create(node:label{属性：属性值}) return n
//    说明：文档示例的标签是「人工智能发展史」、节点名「支持向量机」；
//          这里给出通用写法，并用 TEST 标签做演示，避免污染正式数据。
// ------------------------------------------------------------
CREATE (n:TEST {name:'TEST-NAME', 含义:'这是一个演示节点', 创建时间:'2026-09-10'}) RETURN n;

// 再建一个，用于下一步建立关系
CREATE (n:TEST {name:'TEST-NAME1', 含义:'第二个演示节点'}) RETURN n;

// 【文档原版示例，领域换成人工智能发展史时可直接用】
// CREATE (n:人工智能发展史 {name:'支持向量机', 含义:'Support Vector Machines,SVM，软边距的非线性支持向量机'}) RETURN n;


// ------------------------------------------------------------
// ② 建立关系
//    文档语法：
//      MATCH (a:TEST),(b:TEST)
//      WHERE a.name = 'TEST-NAME' AND b.name = 'TEST-NAME1'
//      CREATE (a)-[r:RELTYPE] -> (b)
//      RETURN r
// ------------------------------------------------------------
MATCH (a:TEST), (b:TEST)
WHERE a.name = 'TEST-NAME' AND b.name = 'TEST-NAME1'
CREATE (a)-[r:RELTYPE]->(b)
RETURN r;

// 【真实图谱中建关系示例】把外滩和东方明珠标记为「相邻景点」
// MATCH (a:Scenic {name:'外滩The Bund'}), (b:Scenic {name:'东方明珠 Oriental Pearl Tower'})
// CREATE (a)-[r:相邻景点]->(b) RETURN r;


// ------------------------------------------------------------
// ③ 给关系添加属性
//    文档语法：MATCH p=()-[r:`关系名`]->() SET r={title:" 属性值"} RETURN p
//    注意：用 SET r = {…} 会覆盖该关系原有的全部属性
// ------------------------------------------------------------
MATCH p=()-[r:RELTYPE]->() SET r={title:"主要"} RETURN p;

// 【等价的更安全写法：只加/改一个属性，不影响其它属性】
// MATCH ()-[r:RELTYPE]->() SET r.title = '主要' RETURN r;


// ------------------------------------------------------------
// ④ 给节点添加属性
//    文档语法：
//      MATCH(n:label{name:"实体名"})
//      Set n.属性='属性值'
//      Return n
// ------------------------------------------------------------
MATCH (n:TEST {name:"TEST-NAME"})
SET n.演示属性 = '这是后添加的属性'
RETURN n;

// 【真实图谱示例】给上海迪士尼打上「网红打卡」标签
// MATCH (n:Scenic {name:'上海迪士尼度假区Shanghai Disney Resort'})
// SET n.标签 = '网红打卡'
// RETURN n.name, n.标签;


// ------------------------------------------------------------
// ⑤ 删除节点 / 关系 / 属性
//    文档语法：match(n) detach delete n      ← 删除【全部】节点和关系，危险！
// ------------------------------------------------------------
// 5.1 只删除演示用的 TEST 节点及其关系（推荐，安全）
MATCH (n:TEST) DETACH DELETE n;

// 5.2 删除全部节点和关系（文档原版，执行后图谱清空，需重新运行 data_to_neo4j.py 恢复）
// MATCH (n) DETACH DELETE n;

// 5.3 只删除某类关系
// MATCH ()-[r:BEST_SEASON]->() DELETE r;

// 5.4 只删除节点的某个属性
// MATCH (n:Scenic {name:'外滩The Bund'}) REMOVE n.标签;


// ============================================================
// 附：本图谱常用查询（可直接用于实验报告截图）
// ============================================================

// Q1 查看图谱规模
MATCH (n) RETURN count(n) AS 节点总数;
MATCH ()-[r]->() RETURN count(r) AS 关系总数;

// Q2 查看所有标签及数量
MATCH (n) RETURN labels(n) AS 标签, count(*) AS 数量 ORDER BY 数量 DESC;

// Q3 上海评分最高的 10 个景点
MATCH (c:City {name:'上海'})-[:HAS_SPOT]->(s:Scenic)
WHERE s.score IS NOT NULL
RETURN s.name AS 景点, s.score AS 评分, s.district AS 区县, s.ticket_min_price AS 最低票价
ORDER BY s.score DESC LIMIT 10;

// Q4 春季适宜的自然山地类景点（跨城市推荐）
MATCH (s:Scenic)-[:BEST_SEASON]->(sea:Season {name:'春'})
MATCH (s)-[:BELONGS_TO]->(cat:Category {name:'自然山地'})
WHERE s.score IS NOT NULL
RETURN s.city AS 城市, s.name AS 景点, s.score AS 评分
ORDER BY s.score DESC LIMIT 20;

// Q5 各城市景点数量排行
MATCH (c:City)-[:HAS_SPOT]->(s:Scenic)
RETURN c.name AS 城市, count(s) AS 景点数
ORDER BY 景点数 DESC LIMIT 20;

// Q6 可视化某个城市的子图（Browser 里看 Graph 视图效果最好）
MATCH p=(c:City {name:'丽江'})-[:HAS_SPOT]->(s:Scenic)-[:BELONGS_TO]->(cat:Category)
RETURN p LIMIT 50;

// Q7 同类景点推荐：与「外滩」同类别且评分 4.5 以上的其它景点
MATCH (s1:Scenic {name:'外滩The Bund'})-[:BELONGS_TO]->(cat:Category)
      <-[:BELONGS_TO]-(s2:Scenic)
WHERE s2 <> s1 AND s2.score >= 4.5
RETURN s2.city AS 城市, s2.name AS 景点, s2.score AS 评分
ORDER BY s2.score DESC LIMIT 20;
