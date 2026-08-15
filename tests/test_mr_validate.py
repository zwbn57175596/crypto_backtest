from scripts.mr_validate import evaluate_gate


def _report(net, sharpe, mdd, trades):
    return {"net_return": net, "sharpe_ratio": sharpe,
            "max_drawdown": mdd, "trades": trades}


def _t(pnl, comm):
    return {"pnl": pnl, "commission": comm}


def test_gate_passes_when_all_green():
    trades = [_t(0.0, 0.02), _t(10.0, 0.02), _t(0.0, 0.02), _t(8.0, 0.02)]
    ok, reasons = evaluate_gate(_report(0.05, 0.8, 0.10, trades))
    assert ok is True
    assert reasons == []


def test_gate_fails_on_negative_net_return():
    trades = [_t(0.0, 0.02), _t(-5.0, 0.02)]
    ok, reasons = evaluate_gate(_report(-0.02, 0.5, 0.10, trades))
    assert ok is False
    assert any("net_return" in r for r in reasons)


def test_gate_fails_on_high_drawdown():
    trades = [_t(0.0, 0.02), _t(10.0, 0.02)]
    ok, reasons = evaluate_gate(_report(0.05, 0.8, 0.30, trades))
    assert ok is False
    assert any("max_drawdown" in r for r in reasons)


def test_gate_fails_on_nonpositive_avg_net_trade():
    # 毛利被手续费吃光：两笔收盘各 +0.01，佣金合计更大
    trades = [_t(0.0, 5.0), _t(0.01, 5.0), _t(0.0, 5.0), _t(0.01, 5.0)]
    ok, reasons = evaluate_gate(_report(0.05, 0.8, 0.10, trades))
    assert ok is False
    assert any("avg_net_trade" in r for r in reasons)
