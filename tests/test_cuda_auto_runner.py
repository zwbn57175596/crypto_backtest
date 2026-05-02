from backtest.cuda_runner import CudaAutoOptimizer
from backtest.optimizer import OptimizeResult, load_strategy_optimize_space


def _fake_cuda_run(self):
    combos = self.param_space.grid()
    trials = []
    for combo in combos:
        score = -(
            abs(combo.get("CONSECUTIVE_THRESHOLD", 5) - 6) * 10
            + abs(combo.get("POSITION_MULTIPLIER", 1.1) - 1.25) * 100
            + abs(combo.get("INITIAL_POSITION_PCT", 0.01) - 0.0175) * 1000
            + abs(combo.get("PROFIT_CANDLE_THRESHOLD", 1) - 2) * 5
        )
        report = {
            "net_return": score / 100,
            "annual_return": score / 100,
            "max_drawdown": abs(score) / 1000,
            "sharpe_ratio": score,
            "sortino_ratio": score,
            "win_rate": max(0.0, 1.0 + score / 1000),
            "profit_factor": max(0.0, 2.0 + score / 1000),
            "total_trades": 42.0,
        }
        trials.append({"params": combo, "score": score, "report": report})

    trials.sort(key=lambda item: item["score"], reverse=True)
    return OptimizeResult(
        best_params=trials[0]["params"] if trials else {},
        best_score=trials[0]["score"] if trials else 0.0,
        all_trials=trials,
        objective=self.objective,
        total_trials=len(trials),
        elapsed_seconds=0.01,
    )


def test_cuda_auto_optimizer_refines_beyond_coarse_grid(monkeypatch):
    monkeypatch.setattr("backtest.cuda_runner.CudaGridOptimizer.run", _fake_cuda_run)

    coarse_space = load_strategy_optimize_space("strategies/consecutive_reverse.py")
    optimizer = CudaAutoOptimizer(
        db_path="unused.db",
        strategy_path="strategies/consecutive_reverse.py",
        symbol="BTCUSDT",
        interval="1h",
        start="2024-01-01",
        end="2024-01-31",
        balance=1000.0,
        leverage=50,
    )

    result = optimizer.run()

    assert coarse_space.total_combinations == 1080
    assert result.total_trials > coarse_space.total_combinations
    assert result.best_params["CONSECUTIVE_THRESHOLD"] == 6
    assert result.best_params["POSITION_MULTIPLIER"] == 1.25
    assert result.best_params["INITIAL_POSITION_PCT"] == 0.0175
    assert result.best_params["PROFIT_CANDLE_THRESHOLD"] == 2
