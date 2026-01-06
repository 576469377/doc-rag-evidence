# 《多模态文档检索增强问答与证据定位系统》用户手册 V0
版本：V0.1
日期：YYYY-MM-DD

## 1. 软件概述
### 1.1 背景与目的
本软件用于对 PDF/图片类文档进行结构化处理，并提供基于检索增强生成（RAG）的问答能力，输出可追溯证据（页码/片段/可选坐标定位），支持批量评测与报表导出。

### 1.2 适用范围
- 适用对象：研究人员/工程人员/文档审阅人员
- 适用场景：标准文档、论文/报告、产品说明、制度文件等的快速问答与证据定位

### 1.3 系统组成
- 文档导入与管理模块
- 文档结构化模块（文本抽取/OCR、块切分、元数据）
- 索引与检索模块（文本检索/向量检索，可混合）
- 证据选择模块（TopK 证据条目）
- 生成与引用模块（答案生成、引用输出）
- 评测与报表模块（批量运行与导出）
- 日志与追溯模块（run 记录）

（可在此插入结构图）
图 1-1 系统模块结构图  
![](figures/fig_1_1_arch.png)

## 2. 运行环境与安装
### 2.1 硬件与软件环境
- 操作系统：macOS / Linux / Windows（任选其一填写）
- Python：3.10+（建议）
- 可选：GPU（如使用本地 VLM/LLM）

### 2.2 安装步骤
1) 获取代码
- git clone <仓库地址>
- cd <项目目录>

2) 创建环境并安装依赖
- pip install -r requirements.txt
（或 conda env create -f environment.yml）

3) 配置文件
- 编辑 configs/app.yaml（或使用默认配置）

### 2.3 启动方式
- 启动 UI：python -m app.ui
- 启动 API（可选）：uvicorn app.api:app --host 0.0.0.0 --port 8000

图 2-1 软件首页（启动后界面）  
![](figures/fig_2_1_home.png)

## 3. 功能操作说明
### 3.1 文档导入与管理
#### 3.1.1 导入文档
步骤：
1) 打开“文档管理”页面
2) 点击“上传”，选择 PDF/图片
3) 提交后自动生成 doc_id，并显示页数与导入时间

预期结果：
- 文档出现在列表中
- data/docs/{doc_id}/ 目录下生成 meta.json 与页面工件

图 3-1 上传文档  
![](figures/fig_3_1_upload.png)

#### 3.1.2 查看与删除
- 查看：点击文档条目可浏览页缩略图/文本片段（如实现）
- 删除：删除后清理对应 artifacts（按实际实现描述）

### 3.2 建索引
步骤：
1) 在“文档管理”选择目标文档
2) 点击“构建索引”
3) 等待完成提示（显示 unit_count、耗时等）

预期结果：
- data/indices/ 下生成索引文件
- build stats 可在界面或日志中查看

图 3-2 构建索引完成提示  
![](figures/fig_3_2_build_index.png)

### 3.3 检索问答与证据定位
步骤：
1) 打开“检索问答”页面
2) 输入问题（可选限定 doc_id）
3) 点击“查询”
4) 查看答案与证据卡片（页码/片段/可选坐标定位）

输出说明：
- 答案区域：显示回答文本
- 证据区域：显示 TopN 证据条目（doc_id/page_id/block_id/snippet/score）
- 定位功能：点击证据跳转到对应页（如实现）或展示页图

图 3-3 问答结果与证据卡片  
![](figures/fig_3_3_qa_result.png)

图 3-4 证据定位（页级或框选定位）  
![](figures/fig_3_4_evidence_loc.png)

### 3.4 批量评测与报表导出
#### 3.4.1 数据集格式
V0 支持 CSV 或 JSON（任选一种说明）
- CSV 列建议：qid,question,answer_gt（可选）,doc_filter（可选）
- JSON 结构建议：参见附录 A

#### 3.4.2 运行评测
步骤：
1) 打开“评测报表”页面
2) 选择评测数据集文件
3) 点击“开始评测”
4) 评测完成后下载 report.json 与 predictions.csv

图 3-5 评测页面  
![](figures/fig_3_5_eval.png)

#### 3.4.3 报表字段说明
- predictions.csv：qid/question/answer_pred/cited_units/latency/status
- report.json：metrics + rows + artifact_paths

## 4. 日志与追溯
### 4.1 单次问答日志
- 路径：data/runs/{query_id}.json
- 内容：query、retrieval.hits、evidence、generation、status、config_snapshot

### 4.2 常见错误与排查
- 文档导入失败：检查文件是否损坏、依赖是否安装
- 无检索结果：检查索引是否构建、doc_filter 是否过窄
- 引用缺失：检查 evidence 是否为空、require_citations 设置

## 5. 版本信息
- V0.1：实现导入、索引、问答证据追溯、评测导出（填写实际完成项）
- 后续计划：重排、块级 bbox、更多指标、权限与多用户（可选）

## 附录 A：评测数据 JSON 示例
```json
{
  "name": "demo_eval",
  "items": [
    {"qid": "q1", "question": "……", "answer_gt": "……"},
    {"qid": "q2", "question": "……"}
  ]
}
