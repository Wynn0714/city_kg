# 城市旅游景点知识图谱构建与应用

以城市旅游景点 POI 数据为领域，完成一条完整的知识图谱链路：

**原始数据清洗 → Neo4j 建图 → 远程监督标注 → BERT 实体/关系抽取 → 抽取结果回写图谱 → Flask Web 智能应用**

代码对应课程实验一至实验四，全部功能可复现。仓库只含源代码，不含原始数据、模型权重与任何个人信息。

---

## 一、功能总览

| 阶段 | 文件 | 功能 |
|---|---|---|
| ① 数据预处理 | `preprocess_citydata.py` | 读取分城市原始 CSV，完成字段清洗、地址块拆分（地址/电话/官网）、景点名去噪、类别与适宜季节规则归类、去重，输出 6 张规范化表 + 统计报告 |
| ② 图谱构建 | `data_to_neo4j.py` | 用 py2neo 批量写入 Neo4j：建唯一性约束、MERGE 实体与关系、数值字段转换，支持分批提交与进度输出 |
| ③ 查询示例 | `cypher_demo.cypher` | 约束、统计、景点检索、条件推荐、子图查询等常用 Cypher 语句，可直接在 Neo4j Browser 执行 |
| ④ 远程监督标注 | `ner/ner_label.py`<br>`ner/rel_label.py` | 从结构化字段与景点介绍文本自动生成字级 BIO 实体语料（SCENIC / LOC / TIME / PER / ORG）与关系分类语料（located_in / built_in / related_person / managed_by / no_relation），并按比例划分训练/验证/测试集 |
| ⑤ 实体抽取 | `ner/train_bert.py`<br>`ner/train_bilstm.py` | BERT-CRF 主模型与 BiLSTM-CRF 基线，按实体类别输出 P/R/F1（seqeval），保存权重与指标文件 |
| ⑥ 关系抽取 | `ner/train_re.py` | 在文本中用 `<e1></e1> <e2></e2>` 标记实体对，取 `[CLS]` 向量做关系分类，输出宏平均 F1 与分类报告 |
| ⑦ 端到端抽取入库 | `ner/demo_pipeline.py` | 串联 NER → RE：对景点介绍抽实体、清洗边界噪声、组对判关系，把三元组写入 Neo4j 的**独立子图**（`NEREntity` / `Doc` 节点），不污染主图谱 |
| ⑧ 评估演示 | `ner/eval_demo.py`<br>`ner/eval_demo_re.py` | 在测试集上输出逐类指标与预测/标注对照，用于结果核验 |
| ⑨ Web 智能应用 | `app/kg_app.py` | 见下节，Flask 后端 + 内置前端页面 + ECharts 可视化 |

### Web 应用五大功能

| 功能 | 说明 |
|---|---|
| **景点检索** | 按景点名/城市模糊搜索，返回评分、门票、地址、开放时间、所属城市/类别/季节；检索结果按「完全匹配 > 前缀 > 包含 > 地址匹配」分级排序，景点名与官网可点击跳转 |
| **智能推荐** | 按「城市 / 季节 / 类别」任意组合筛选，按评分降序推荐并分页；筛选项由数据库真实取值动态生成，避免前后端枚举不一致 |
| **子图可视化** | 输入城市名查看该城高分景点子图，或输入景点名查看 1 跳邻居；ECharts 力导向图按节点类型着色，支持拖拽与缩放 |
| **统计看板** | 节点/关系总量、类别分布、季节分布、Top 城市等聚合统计，柱状图 + 饼图展示 |
| **智能问答** | 支持「城市+类型」「具体景点」「事实提问」「条件推荐」四类问法；先做口语→图谱取值映射（如「自然景观」→自然山地+水域景观）、城市最长匹配、属性词切分，再分流到检索/推荐；可选接入大模型做意图解析，但**所有事实值一律从 Neo4j 取**，不接受模型编造的数值 |

应用还内置：数据库断线自动重连、页码/条数夹取防深翻、ECharts 本地静态文件（断网可用）、问答结果缓存与限流降级。

---

## 二、技术栈

- **图数据库**：Neo4j 5.x，py2neo
- **Web**：Flask（后端 API）+ 原生前端 + Apache ECharts（本地静态资源）
- **模型**：PyTorch、Transformers、`bert-base-chinese`、BERT-CRF、BiLSTM-CRF、BERT 关系分类
- **评估**：seqeval、scikit-learn
- **语言**：Python 3.10+

---

## 三、目录结构

```
city_kg/
├── README.md
├── requirements.txt            # 全部 Python 依赖
├── .env.example                # 环境变量模板（复制为 .env 后填写）
├── .gitignore
├── preprocess_citydata.py      # ① 数据预处理
├── data_to_neo4j.py            # ② 导入 Neo4j
├── cypher_demo.cypher          # ③ Cypher 查询示例
├── ner/
│   ├── ner_label.py            # ④ 实体语料远程监督标注
│   ├── rel_label.py            # ④ 关系语料远程监督标注
│   ├── train_bert.py           # ⑤ BERT-CRF 实体抽取
│   ├── train_bilstm.py         # ⑤ BiLSTM-CRF 基线
│   ├── train_re.py             # ⑥ BERT 关系抽取
│   ├── demo_pipeline.py        # ⑦ NER → RE → 写库
│   ├── eval_demo.py            # ⑧ 实体抽取评估演示
│   ├── eval_demo_re.py         # ⑧ 关系抽取评估演示
│   └── metrics_*.json          # 实测指标（供查阅，非运行必需）
└── app/
    ├── kg_app.py               # ⑨ Flask 应用（前后端一体）
    └── static/
        └── echarts.min.js      # ECharts 本地副本，保证离线可用
```

---

## 四、快速开始

### 1. 安装依赖

```bash
git clone https://github.com/Wynn0714/city_kg.git
cd city_kg
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

> 训练 NER/RE 模型建议有 CUDA GPU；仅运行 Web 应用与数据预处理不需要 GPU。
> `bert-base-chinese` 首次训练时由 transformers 自动下载（亦可提前放到本地缓存并设 `TRANSFORMERS_OFFLINE=1`）。

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，按需填写后加载到当前终端（或用系统环境变量）：

| 变量 | 用途 | 默认值 |
|---|---|---|
| `NEO4J_URI` | Neo4j bolt 地址 | `bolt://localhost:7687` |
| `NEO4J_USER` | 用户名 | `neo4j` |
| `NEO4J_PASSWORD` | **密码，必填** | 无 |
| `NEO4J_DB` | 数据库名 | `neo4j` |
| `CITYDATA_DIR` | 原始分城市 CSV 目录 | `./data/raw` |
| `AGNES_API_KEY` | 可选，问答模块的大模型密钥 | 空（自动降级为规则解析） |
| `AGNES_BASE_URL` / `AGNES_MODEL` | 可选，OpenAI 兼容接口地址与模型名 | 见 `.env.example` |

**仓库内不含任何真实密钥或密码**；未配置 `NEO4J_PASSWORD` 时导入脚本会直接提示退出，Web 应用会返回明确的连接错误。

### 3. 准备数据并建图

原始数据为按城市分文件的景点 POI CSV（不含在仓库内，请自行放入 `CITYDATA_DIR` 指定的目录，默认 `data/raw/`）。

```bash
python preprocess_citydata.py     # 输出 cleaned/*.csv 与 preprocess_report.txt
# 启动 Neo4j，确认 Browser 可登录
python data_to_neo4j.py           # 批量导入，构建主图谱
```

`cypher_demo.cypher` 可在 Neo4j Browser 里逐条执行，用于验证导入结果。

### 4. 标注与训练（可选）

```bash
python ner/ner_label.py           # 生成 ner/train|dev|test.bio
python ner/rel_label.py           # 生成 ner/rel_*.jsonl
python ner/train_bilstm.py        # 基线模型
python ner/train_bert.py          # BERT-CRF
python ner/train_re.py            # BERT-RE
python ner/eval_demo.py           # 实体抽取评估 + 预测演示
python ner/eval_demo_re.py        # 关系抽取评估 + 预测演示
python ner/demo_pipeline.py       # 端到端抽取并把三元组写入 Neo4j（加 --no-db 可只抽取不写库）
```

模型权重（`.pt`）与训练语料（`.bio` / `.jsonl`）体量较大，由上述脚本生成，故不在仓库中分发。

### 5. 启动 Web 应用

```bash
cd app
python kg_app.py
# 浏览器打开 http://localhost:5000
```

主要接口：`/api/search`（检索）、`/api/recommend`（推荐）、`/api/graph`（子图）、`/api/stats`（统计看板）、`/api/qa`（问答）、`/api/opts`（筛选项）、`/api/health`（健康检查）。

---

## 五、图谱 Schema

| 实体（标签） | 关键属性 |
|---|---|
| `City` | name, spot_count |
| `Scenic` | sid, name, city, score, ticket, address, phone, website, open_time, intro … |
| `Season` | name, scenic_count |
| `Category` | name, scenic_count |
| `NEREntity` / `Doc` | 模型抽取结果独立子图（由 `demo_pipeline.py` 写入） |

| 关系 | 含义 |
|---|---|
| `(City)-[:HAS_SPOT]->(Scenic)` | 城市拥有景点 |
| `(Scenic)-[:BELONGS_TO]->(Category)` | 景点所属类别 |
| `(Scenic)-[:BEST_SEASON]->(Season)` | 景点适宜季节 |
| `LOCATED_IN` / `BUILT_IN` / `RELATED_PERSON` / `MANAGED_BY` / `CONTAINS` | 模型抽取三元组构成的子图关系 |

---

## 六、技术亮点

| 亮点 | 做法 |
|---|---|
| **BERT-CRF 序列标注** | `bert-base-chinese` 接 CRF 层，用转移约束 + Viterbi 解码保证 BIO 标签序列合法；另训 BiLSTM-CRF 作基线对照（宏 F1 0.7763 → 0.8447） |
| **实体感知的关系抽取** | 向词表加入 `<e1></e1> / <e2></e2>` 特殊 token 并扩展词嵌入，取 `[CLS]` 向量判定关系；负例跨文档采样 |
| **级联抽取流水线** | NER 预测 → 实体边界清洗 → 实体配对 → RE 判定 → 三元组写库；RE 的输入是 NER 的真实输出，而非人工标注或结构化字段 |
| **远程监督标注** | 用结构化字段对齐 + 朝代 / 机构 / 人物规则自动生成 BIO 与关系语料，无需人工标注，并输出标注报告供核查噪声 |
| **LLM 只解析、不造事实** | 大模型仅把自然语言转成查询条件，门票、评分等事实值一律回 Neo4j 取；配套结果缓存、失败冷却与限流降级，未配置密钥时自动改用规则解析 |
| **口语 → 图谱取值映射** | 同义词表（一对多）、城市最长匹配、类别多选、属性词切分与四类意图分流，解决「自然景观」这类口语和图谱枚举值对不上的问题 |
| **Cypher 查询优化** | 用 `EXISTS{}` 子查询做多值关系真过滤（避开 `OPTIONAL MATCH` 后 `WHERE` 被静默忽略的语义陷阱）；`CASE WHEN` 实现空值沉底与检索分级排序；参数化查询 + 分页夹取防深翻 |
| **图谱扩展隔离** | 模型抽取的三元组写入 `NEREntity` / `Doc` 独立子图并锚定来源文档，可追溯、可清理，不污染主图谱 |
| **可视化与健壮性** | ECharts 力导向图本地化、断网可用；每请求探活并自动重连 Neo4j；输出统一转义防 XSS；唯一性约束 + `MERGE` 分批导入保证幂等 |

> 这些做法主要对应三类实际问题：小样本下如何构造训练语料、如何避免大模型编造事实、以及多值关系下图查询如何不被静默放宽。

---

## 七、实验结果（测试集）

以下均为**测试集**实测结果，原始指标保存在 `ner/metrics_*.json`。

> **评估口径**：字级 BIO 与关系分类均报告逐类 Precision / Recall / F1 与宏平均 F1（seqeval / sklearn）。由于 `O` 标签与 `no_relation` 负例占比很高，整体「准确率 accuracy」会被多数类拉高、无法反映实体边界与少数关系的识别能力，因此不以 accuracy 作为主指标。

### 1. 语料规模与划分

| 语料 | 训练集 | 验证集 | 测试集 | 合计 |
|---|---|---|---|---|
| 实体标注（句，字级 BIO） | 10,474 | 2,244 | 2,246 | 14,964 |
| 关系分类（实体对实例） | 48,385 | 10,403 | 10,471 | 69,259 |

| 实体类别 | SCENIC | LOC | TIME | PER | ORG |
|---|---|---|---|---|---|
| 出现次数 | 22,002 | 13,582 | 32,421 | 2,103 | 15,532 |

| 关系类别 | no_relation | built_in | managed_by | located_in | related_person |
|---|---|---|---|---|---|
| 实例数 | 34,514 | 18,871 | 8,595 | 6,814 | 465 |

> 语料由远程监督自动生成：SCENIC / LOC 与 `located_in` 由结构化字段对齐（高精度），TIME / ORG 与部分关系由正则词典抽取（含噪声），PER 与 `related_person` 样本最少、噪声最高，标注报告里也标明了建议人工抽检。

### 2. 实体抽取（字级 BIO）

| 模型 | 宏平均 P | 宏平均 R | **宏平均 F1** |
|---|---|---|---|
| BiLSTM-CRF（基线） | — | — | 0.7763 |
| **BERT-CRF** | — | — | **0.8447** |

**BERT-CRF 测试集逐类结果**（并与基线 F1 对比）：

| 实体类别 | Precision | Recall | F1 | BiLSTM-CRF F1 |
|---|---|---|---|---|
| TIME | 0.9862 | 0.9779 | **0.9820** | 0.9606 |
| PER | 0.9185 | 0.9441 | 0.9311 | 0.6977 |
| LOC | 0.8062 | 0.8510 | 0.8280 | 0.6315 |
| ORG | 0.8209 | 0.8111 | 0.8160 | 0.7971 |
| SCENIC | 0.6369 | 0.8422 | 0.7253 | 0.6193 |
| **宏平均** | — | — | **0.8447** | **0.7763** |

### 3. 关系抽取（BERT-RE）

| 关系类别 | Precision | Recall | F1 |
|---|---|---|---|
| located_in | 0.9708 | 0.9737 | **0.9723** |
| no_relation | 0.8552 | 0.9994 | 0.9217 |
| built_in | 0.9973 | 0.7751 | 0.8723 |
| managed_by | 0.9772 | 0.7779 | 0.8663 |
| related_person | 0.4957 | 0.7600 | 0.6000 |
| **宏平均** | — | — | **0.8465** |

### 4. 训练配置

| 模型 | 预训练 / 结构 | epochs | batch | lr |
|---|---|---|---|---|
| BERT-CRF | `bert-base-chinese` + CRF | 6 | 16 | 2e-5 |
| BiLSTM-CRF | emb=100, hid=128 + CRF | 25 | 64 | 1e-3 |
| BERT-RE | `bert-base-chinese` + `[CLS]` 分类（负例采样比 1.0） | 8 | 16 | 2e-5 |

### 5. 结果分析

- **预训练带来 +0.0684 宏 F1**（0.7763 → 0.8447），增益主要来自召回：PER 0.5607 → 0.9441、LOC 0.5710 → 0.8510、SCENIC 0.5863 → 0.8422，说明上下文语义表示对边界模糊的实体尤其有效。
- **TIME 最易识别**（F1 0.9820），因为朝代/年号表达规整；**SCENIC 最难**（F1 0.7253），精确率仅 0.6369 而召回 0.8422，典型的边界过宽/误吞前后修饰词，也是远程监督标注噪声的主要来源，因此流水线里专门加了实体边界清洗。
- **关系抽取中 `located_in` 最好**（0.9723），`related_person` 最差（0.6000）——后者全库仅 465 条实例且由人物词典生成，样本稀缺叠加标注噪声；`built_in` / `managed_by` 精确率接近 1 但召回约 0.78，说明判得准但漏判较多。
- 复现方式：`python ner/eval_demo.py` 与 `python ner/eval_demo_re.py` 可打印同样的逐类指标及预测/标注对照。

### 6. 图谱规模

一次完整导入后：**31,173 个节点 / 64,914 条关系**，其中景点 30,791、城市 349、类别 11、季节 5，另有模型抽取产生的独立子图（`NEREntity` 13、`Doc` 4）。

---

## 八、说明

- 仓库**不含**：原始 352 个城市 CSV、清洗后的 `cleaned/`、训练语料、模型权重、图谱导出 CSV、实验截图、实验报告文档，以及任何学号/姓名/本机路径/密钥信息。数据、语料、权重、图谱均可由本仓库脚本从原始数据重新生成。
- `app/static/echarts.min.js` 为 Apache ECharts 的本地副本，用于离线运行可视化模块。
- 问答模块的大模型为可选项：未配置密钥时自动走内置规则解析，功能不受影响。
