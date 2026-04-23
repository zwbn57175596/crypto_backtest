# PRD: 回测结果页面与分析页面优化

## 1. 背景与目标

### 1.1 现状概述

当前项目包含两个 Web 页面：

- **回测报告页 (`index.html`)** — 展示单次回测的 14 项指标、权益曲线、回撤曲线、逐笔 PnL 及交易明细表
- **参数优化页 (`optimize.html`)** — 展示多组参数优化结果的散点图、热力图及排名表格

两个页面基于 FastAPI + 原生 JS + ECharts 构建，暗色主题，功能基本可用但在用户体验、数据分析深度、交互能力上存在明显短板。

### 1.2 优化目标

1. **提升数据洞察力** — 增加更丰富的分析维度，帮助用户更快定位策略优劣势
2. **增强交互体验** — 改善图表联动、筛选、对比等操作流程
3. **补齐功能缺失** — 导出、对比、基准线等实用功能
4. **优化性能** — 解决大数据量下的渲染瓶颈

---

## 2. 回测报告页优化

### 2.1 指标增强

#### P0 — 新增风险指标卡片

| 指标 | 说明 |
|------|------|
| Calmar Ratio | 年化收益 / 最大回撤 |
| 最大连续盈利/亏损次数 | 连续 Win/Loss Streak |
| 日均收益率 | 按日聚合的平均收益 |
| 收益波动率 | 收益率标准差（年化） |

#### P1 — 指标分组展示

将现有 14 项 + 新增指标分为 3 组卡片：
- **收益类** — 总收益率、年化收益、日均收益率
- **风险类** — 最大回撤、最大回撤持续时间、收益波动率、Sharpe、Sortino、Calmar
- **交易类** — 胜率、盈亏比、总交易数、多/空交易数、平均持仓时间、最大连续盈亏、总手续费、总资金费

### 2.2 图表增强

#### P0 — 权益曲线叠加基准线

- 增加 Buy & Hold 基准线（同期持有不操作的收益曲线）
- 支持开关显示/隐藏基准线
- 用虚线样式区分，图例标注

#### P0 — 月度收益热力图

- 新增日历热力图（ECharts calendar heatmap）
- X 轴月份，Y 轴年份，颜色映射月度收益率
- 快速识别策略在不同时间段的表现

#### P1 — 交易分布统计图

- 新增 PnL 分布直方图（histogram）
- 展示盈亏金额分布，标注均值和中位数
- 可选：按持仓时间分桶的收益分布

#### P1 — 回撤恢复分析

- 在回撤曲线下方标注 Top 3 回撤区间
- 显示每次回撤的起始时间、最低点时间、恢复时间
- 点击高亮对应权益曲线区间

### 2.3 交易明细表增强

#### P0 — 虚拟分页

- 对交易记录实现虚拟滚动（virtual scroll），解决 10k+ 交易的 DOM 性能问题
- 每次只渲染可视区域内的行

#### P1 — 筛选与搜索

- 按方向筛选（仅多单 / 仅空单）
- 按盈亏筛选（仅盈利 / 仅亏损）
- 按时间范围筛选

#### P1 — 交易与图表联动

- 点击交易行高亮权益曲线上对应时间点
- 点击 PnL 柱状图跳转到交易表对应行

### 2.4 优化参数卡片增强（关联批次信息）

#### 问题分析

当前报告页的"优化参数"卡片（`index.html:86-92`）通过 `reports.optimize_result_id` 关联到 `optimize_results` 表，仅展示 score、objective 和参数 key-value。用户无法得知：

- 这组参数来自哪个优化批次
- 那次优化的运行时间
- 那次优化的回测时间范围（start_date ~ end_date）
- 同批次中这组参数的排名

#### P0 — 卡片展示批次上下文

优化参数卡片改版为：

```
┌───────────────────────────────────────────────┐
│ 优化参数                  score: 2.3400 | sharpe_ratio │
│───────────────────────────────────────────────│
│ 批次: #3  2026-04-23 12:00                    │
│ 回测区间: 2026-01-01 ~ 2026-06-30            │
│ 批次排名: 12 / 1000                           │
│───────────────────────────────────────────────│
│ CONSECUTIVE_THRESHOLD          5              │
│ POSITION_MULTIPLIER            1.10           │
│ INITIAL_POSITION_PCT           0.02           │
│ PROFIT_CANDLE_THRESHOLD        3              │
│───────────────────────────────────────────────│
│ [查看该批次全部结果 →]                          │
└───────────────────────────────────────────────┘
```

**新增展示字段：**
- 批次编号 + 运行时间（来自 `optimize_results.batch_id` / `created_at`）
- 回测区间（来自 `optimize_results.start_date` / `end_date`）
- 该参数在批次中的排名（需后端查询同 batch_id 下按 score 排序的位置）

**交互：**
- "查看该批次全部结果" 链接跳转到优化页，并自动选中对应批次

#### P0 — 后端 API 扩展

修改 `GET /api/reports/{id}` 返回值，增加批次上下文：

```json
{
  "optimize_params": {...},
  "optimize_score": 2.34,
  "optimize_objective": "sharpe_ratio",
  "optimize_batch_id": "20260423T120000_MaCross_BTCUSDT",
  "optimize_batch_created_at": "2026-04-23T12:00:00+00:00",
  "optimize_start_date": "2026-01-01",
  "optimize_end_date": "2026-06-30",
  "optimize_rank": 12,
  "optimize_batch_total": 1000
}
```

SQL 调整：在现有 LEFT JOIN 基础上补充字段，排名用窗口函数计算：

```sql
SELECT r.*, o.params_json, o.score, o.objective,
       o.batch_id, o.created_at AS batch_created_at,
       o.start_date, o.end_date,
       (SELECT COUNT(*) + 1 FROM optimize_results o2
        WHERE o2.batch_id = o.batch_id AND o2.score > o.score) AS rank_in_batch,
       (SELECT COUNT(*) FROM optimize_results o3
        WHERE o3.batch_id = o.batch_id) AS batch_total
FROM reports r
LEFT JOIN optimize_results o ON r.optimize_result_id = o.id
WHERE r.id = ?
```

### 2.5 报告对比功能

#### P1 — 多报告对比视图

- 支持选择 2-4 份报告进行并排对比
- 对比内容：核心指标表格 + 权益曲线叠加
- URL 参数支持（可分享对比链接）

### 2.6 数据导出

#### P1 — 导出功能

- 支持导出交易记录为 CSV
- 支持导出权益曲线数据为 CSV
- 支持导出报告摘要为 JSON

---

## 3. 参数优化页优化

### 3.0 优化批次选择（核心需求）

#### 问题分析

当前 `optimize_results` 表缺少批次（batch）概念。每次运行参数优化时，`save_results()` 将所有 trial 以相同的 `created_at` 时间戳写入，但没有字段标识"这是第几次优化运行"。页面按 `strategy + symbol` 筛选时，会将历史上所有优化运行的结果混在一起，导致：

1. 不同时间段（start_date/end_date 不同）的优化结果混合，参数对比失去意义
2. 用同一策略反复调优时，历史版本的结果干扰最新分析
3. 无法对比"这次调参"和"上次调参"的差异
4. 数据量随运行次数不断膨胀，图表越来越杂乱

#### P0 — 数据库增加 batch_id

**方案：** 在 `optimize_results` 表新增 `batch_id` 字段，用于标识同一次优化运行产生的所有 trial。

```sql
ALTER TABLE optimize_results ADD COLUMN batch_id TEXT;
```

- `batch_id` 格式：`{timestamp}_{strategy}_{symbol}`（如 `20260423T120000_MaCross_BTCUSDT`）
- `save_results()` 写入时自动生成 batch_id，同一次调用的所有 trial 共享同一个值
- 兼容旧数据：对已有记录，可用 `created_at` 回填 batch_id（相同 created_at + strategy + symbol = 同一批次）

#### P0 — 批次选择器 UI

**前端交互：**

```
筛选栏布局：
┌─────────────────────────────────────────────────────────────────┐
│ [策略下拉▼]  [交易对下拉▼]                                        │
│                                                                 │
│ 优化批次：                                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ☑ 全选 / 取消全选                                           │ │
│ │ ☑ #3  2026-04-23 12:00  (1000组)  best=2.34  📅 1月-6月    │ │
│ │ ☑ #2  2026-04-20 09:30  (500组)   best=1.87  📅 1月-6月    │ │
│ │ ☐ #1  2026-04-15 14:00  (1000组)  best=1.12  📅 1月-3月    │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ [目标: sharpe_ratio]  [已选 1500 组合]                            │
└─────────────────────────────────────────────────────────────────┘
```

**批次列表信息：**
- 批次序号（按时间倒序编号）
- 运行时间（created_at）
- 该批次的 trial 数量
- 该批次的最佳 score
- 回测时间范围（start_date ~ end_date）
- 优化目标（objective）

**交互规则：**
1. 选择策略+交易对后，自动加载该组合下所有批次
2. 默认选中最新一个批次（而非全选）
3. 用户可勾选多个批次，图表和表格实时更新
4. 全选 = 当前行为（融合所有批次）
5. 批次选择状态通过 URL 参数持久化：`?batches=3,2`

#### P0 — 后端 API 扩展

**新增接口：**

```
GET /api/optimize_results/batches?strategy=X&symbol=Y
```

返回：
```json
[
  {
    "batch_id": "20260423T120000_MaCross_BTCUSDT",
    "batch_number": 3,
    "created_at": "2026-04-23T12:00:00+00:00",
    "count": 1000,
    "best_score": 2.34,
    "objective": "sharpe_ratio",
    "start_date": "2026-01-01",
    "end_date": "2026-06-30"
  },
  ...
]
```

**修改现有接口：**

```
GET /api/optimize_results?strategy=X&symbol=Y&batch_ids=id1,id2
```

新增 `batch_ids` 可选参数，传入时只返回指定批次的数据。不传则返回全部（兼容旧行为）。

#### P1 — 批次对比视图

- 选中 2 个批次时，自动进入"批次对比模式"
- 散点图用不同颜色区分两个批次的数据点
- 表格增加"批次"列，支持按批次排序
- 热力图可切换查看单个批次或合并数据

#### P1 — 批次管理

- 支持为批次添加备注标签（如"调大了 threshold 范围"、"换了 1h 周期"）
- 支持删除整个批次（清理无用的历史优化数据）
- 批次备注保存在 `optimize_batches` 元数据表中

### 3.1 可视化增强

#### P0 — 3D 参数空间散点图

- 当优化参数 >= 3 个时，提供 3D 散点图视图
- X/Y/Z 轴分别映射不同参数，颜色映射得分
- 支持鼠标旋转、缩放

#### P1 — 参数敏感度分析图

- 新增单参数边际效应折线图
- 固定其他参数取最优值，展示目标参数变化对 Score 的影响
- 帮助判断参数是否在稳定区间（robustness 分析）

#### P1 — 帕累托前沿标注

- 在风险-收益散点图上标注帕累托最优解集
- 用连线连接帕累托前沿点
- 帮助用户在收益与风险之间做权衡

### 3.2 表格增强

#### P0 — 列自定义

- 支持用户选择显示/隐藏哪些列
- 记住用户偏好（localStorage）

#### P1 — 高级排序

- 支持多列排序（Shift + 点击表头）
- 排序状态持久化

#### P1 — 批量操作

- 复选框选中多行
- 批量导出选中行的详细报告
- 批量跳转到对应回测报告

### 3.3 筛选增强

#### P0 — 参数范围筛选

- 为每个优化参数提供范围滑块（range slider）
- 实时过滤表格和图表数据
- 支持多参数组合筛选

#### P1 — 指标阈值过滤

- 设置 Sharpe > X、MaxDD < Y 等条件
- 快速剔除不达标的参数组合

---

## 4. 通用优化

### 4.1 性能优化

| 项目 | 优先级 | 说明 |
|------|--------|------|
| 权益曲线自适应降采样 | P0 | 根据屏幕宽度动态计算采样点数，替代硬编码 500 |
| 图表懒加载 | P1 | 仅当图表进入可视区域时初始化 |
| API 响应压缩 | P1 | 启用 gzip/brotli 压缩大型 JSON 响应 |
| Web Worker 计算 | P2 | 将数据处理移至 Web Worker，避免阻塞 UI |

### 4.2 错误处理

#### P0 — API 错误提示

- 所有 API 调用增加 try-catch 和用户可见错误提示
- 网络异常时显示重试按钮
- 加载中状态展示 skeleton 屏

### 4.3 响应式优化

#### P1 — 移动端适配

- 优化移动端图表尺寸与交互（触摸缩放）
- 指标卡片自适应列数（1-2-4 列）
- 交易表格横向滚动

### 4.4 URL 路由

#### P1 — 支持深链接

- 报告页支持 URL 参数指定报告 ID：`/?report=123`
- 优化页支持 URL 参数指定策略和品种
- 浏览器前进/后退按钮正常工作

---

## 5. 技术方案概要

### 前端

- 保持原生 JS + ECharts 方案，不引入框架
- 新增图表类型：calendar heatmap、histogram、scatter3D（需加载 ECharts GL 扩展）
- 虚拟滚动使用自实现或轻量库
- 样式沿用现有暗色主题设计语言

### 后端 API 扩展

| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/optimize_results/batches` | GET | 按 strategy+symbol 返回批次列表及元信息 |
| `/api/optimize_results` (修改) | GET | 新增 `batch_ids` 可选参数，按批次过滤 |
| `/api/reports/compare` | GET | 多报告对比，接受 `ids=1,2,3` 参数 |
| `/api/reports/{id}/benchmark` | GET | 返回 Buy & Hold 基准权益曲线 |
| `/api/reports/{id}/monthly` | GET | 返回月度收益聚合数据 |
| `/api/reports/{id}/export/csv` | GET | 导出交易记录 CSV |
| `/api/optimize_results/sensitivity` | GET | 单参数敏感度分析数据 |

### 数据库

**表结构变更：**

```sql
-- optimize_results 表新增 batch_id 列
ALTER TABLE optimize_results ADD COLUMN batch_id TEXT;

-- 旧数据回填（相同 created_at + strategy + symbol 视为同一批次）
UPDATE optimize_results
SET batch_id = strftime('%Y%m%dT%H%M%S', created_at) || '_' || strategy || '_' || symbol
WHERE batch_id IS NULL;

-- 批次元数据表（可选，用于存储批次备注）
CREATE TABLE IF NOT EXISTS optimize_batches (
    batch_id TEXT PRIMARY KEY,
    label TEXT,              -- 用户自定义备注
    created_at TEXT NOT NULL
);
```

- `save_results()` 函数签名新增 `batch_id` 参数，写入时所有 trial 共享同一值
- 其余新增 API 基于现有 `report_json` 字段计算，无需额外改表

---

## 6. 优先级与里程碑

### Phase 1 — 核心体验（P0）

- [ ] **optimize_results 表增加 batch_id 字段 + 旧数据回填**
- [ ] **新增 `/api/optimize_results/batches` 批次列表接口**
- [ ] **优化页增加批次多选器，支持按批次筛选数据**
- [ ] **`/api/optimize_results` 接口支持 batch_ids 过滤参数**
- [ ] **报告页优化参数卡片增加批次上下文（批次号、回测区间、排名、跳转链接）**
- [ ] 新增风险指标卡片（Calmar、连续盈亏等）
- [ ] 指标分组展示
- [ ] 权益曲线叠加 Buy & Hold 基准线
- [ ] 月度收益热力图
- [ ] 交易表虚拟分页
- [ ] API 错误处理与加载状态
- [ ] 权益曲线自适应降采样
- [ ] 3D 参数空间散点图
- [ ] 参数范围滑块筛选
- [ ] 优化表格列自定义

### Phase 2 — 分析深度（P1）

- [ ] 批次对比视图（双批次颜色区分 + 表格批次列）
- [ ] 批次管理（备注标签 + 批量删除）
- [ ] PnL 分布直方图
- [ ] 回撤恢复分析
- [ ] 交易筛选与图表联动
- [ ] 多报告对比视图
- [ ] 数据导出（CSV/JSON）
- [ ] 参数敏感度分析图
- [ ] 帕累托前沿标注
- [ ] 多列排序与批量操作
- [ ] 指标阈值过滤
- [ ] URL 深链接与响应式优化

### Phase 3 — 性能与体验（P2）

- [ ] Web Worker 数据处理
- [ ] 图表懒加载
- [ ] API 响应压缩

---

## 7. 不做的事项

- 不引入前端框架（React/Vue）— 保持轻量单文件方案
- 不增加用户认证系统 — 纯本地工具
- 不支持实时推送（WebSocket）— 回测是离线任务
- 不做策略编辑器 — 策略仍通过 Python 文件管理
