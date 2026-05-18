# Design: --optimize-id 一键复现 + SIZE_LEVERAGE 重命名

**Date:** 2026-05-19  
**Status:** Approved

## 背景

两个独立的设计问题：

1. 策略参数 `LEVERAGE`（仓位倍数）与 CLI 参数 `--leverage`（交易所杠杆）同名但语义不同，导致手动复现优化结果时容易传错参数。
2. 没有一键从优化记录复现完整回测的方式，需要手动拼接多个参数。

---

## Part 1：重命名 `LEVERAGE` → `SIZE_LEVERAGE`

### 目标

消除策略类变量 `LEVERAGE`（仓位大小倍数）与 CLI `--leverage`（交易所杠杆）的命名冲突。

### 受影响文件

| 文件 | 变更内容 |
|------|---------|
| `strategies/consecutive_reverse.py` | `LEVERAGE = 17` → `SIZE_LEVERAGE = 17`；`self.LEVERAGE` → `self.SIZE_LEVERAGE` |
| `strategies/consecutive_reverse_martingale.py` | 同上；OPTIMIZE_SPACE 里的 key |
| `src/backtest/cuda_strategies/__init__.py` | param_order 列表中 `"LEVERAGE"` → `"SIZE_LEVERAGE"` |
| `src/backtest/optimizer.py` | 3 处硬编码 `"LEVERAGE"` 字符串 |
| `src/backtest/__main__.py` | `strategy_class.LEVERAGE = args.sizing_leverage` → `SIZE_LEVERAGE` |

### 旧 DB 记录兼容

`--optimize-id` 加载 params_json 时：若 key 是 `"LEVERAGE"` 而策略类有 `SIZE_LEVERAGE` 属性（无 `LEVERAGE` 属性），自动映射，不报错。

---

## Part 2：`--optimize-id` 一键复现

### DB Schema 变更

```sql
ALTER TABLE optimize_results ADD COLUMN strategy_path TEXT;
ALTER TABLE optimize_results ADD COLUMN exchange TEXT;
```

新跑的优化自动写入；旧记录两列为 NULL。

### Optimizer 保存逻辑

`cmd_optimize` 保存结果时，额外写入 `strategy_path`（strategy 文件路径）和 `exchange`（交易所名称）。

### CLI 变更

`run` 子命令新增可选参数：

```
--optimize-id INT   从 optimize_results 加载参数，一键复现
```

使用 `--optimize-id` 时，以下参数全部变为 optional：
`--strategy`, `--symbol`, `--interval`, `--start`, `--end`

### cmd_run 加载顺序

```
if args.optimize_id:
    row = DB.get(optimize_results, id=args.optimize_id)
    if row.strategy_path is NULL and args.strategy is None:
        → error: "旧记录缺少 strategy_path，请补传 --strategy"
    args.strategy  = args.strategy  or row.strategy_path
    args.symbol    = args.symbol    or row.symbol
    args.interval  = args.interval  or row.interval
    args.start     = args.start     or row.start_date
    args.end       = args.end       or row.end_date
    args.leverage  = args.leverage  or int(row.leverage)   # 交易所杠杆
    args.exchange  = args.exchange  or row.exchange or "binance"
    # 应用策略参数（含 LEVERAGE→SIZE_LEVERAGE compat）
    apply_optimize_params(strategy_class, json.loads(row.params_json))
```

**优先级规则：** 用户显式传入的参数（如 `--balance 5000`、`--end 2026-05-18`）覆盖 DB 值。

**保存 report 时**写入 `optimize_result_id`，使 web 页面可关联到对应的优化批次。

### apply_optimize_params 逻辑

```python
def _apply_optimize_params(strategy_class, params: dict) -> None:
    for key, value in params.items():
        attr = key
        # compat: old records use "LEVERAGE", new strategies have "SIZE_LEVERAGE"
        if key == "LEVERAGE" and not hasattr(strategy_class, "LEVERAGE") \
                and hasattr(strategy_class, "SIZE_LEVERAGE"):
            attr = "SIZE_LEVERAGE"
        if hasattr(strategy_class, attr):
            setattr(strategy_class, attr, type(getattr(strategy_class, attr))(value))
            print(f"  [param] {attr} = {value}")
```

`--sizing-leverage` 显式传入时在 `_apply_optimize_params` 之后再覆盖，显式值优先。

---

## 测试要点

- 用新跑的优化记录（有 strategy_path/exchange）执行 `--optimize-id`，结果与手动传参一致
- 用旧记录（缺 strategy_path）不传 `--strategy` 时报错；传了 `--strategy` 时正常跑
- 重命名后，新优化写入 `SIZE_LEVERAGE`；旧记录含 `LEVERAGE` 时 compat 映射正常生效
- `--balance` / `--end` 等显式参数优先于 DB 值
