import pytest
from backtest.optimizer import ParamSpace


class TestParamSpace:
    def test_int_range(self):
        space = ParamSpace({"X": (10, 30, 10)})
        assert space.grid() == [{"X": 10}, {"X": 20}, {"X": 30}]

    def test_float_range(self):
        space = ParamSpace({"Y": (1.0, 2.0, 0.5)})
        results = space.grid()
        assert len(results) == 3
        assert results[0] == {"Y": 1.0}
        assert results[1] == {"Y": 1.5}
        assert results[2] == {"Y": 2.0}

    def test_choice_list(self):
        space = ParamSpace({"Z": [1, 2, 3]})
        assert space.grid() == [{"Z": 1}, {"Z": 2}, {"Z": 3}]

    def test_cartesian_product(self):
        space = ParamSpace({"A": (1, 2, 1), "B": [10, 20]})
        combos = space.grid()
        assert len(combos) == 4
        assert {"A": 1, "B": 10} in combos
        assert {"A": 2, "B": 20} in combos

    def test_total_combinations(self):
        space = ParamSpace({"A": (1, 3, 1), "B": [10, 20]})
        assert space.total_combinations == 6

    def test_empty_space(self):
        space = ParamSpace({})
        assert space.grid() == [{}]
        assert space.total_combinations == 1
