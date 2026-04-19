from __future__ import annotations

import itertools
from dataclasses import dataclass, field


class ParamSpace:
    """Defines a parameter search space for strategy optimization."""

    def __init__(self, space: dict):
        self._space = space
        self._axes: dict[str, list] = {}
        for name, spec in space.items():
            if isinstance(spec, list):
                self._axes[name] = spec
            elif isinstance(spec, tuple) and len(spec) == 3:
                min_val, max_val, step = spec
                if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                    self._axes[name] = list(range(min_val, max_val + 1, step))
                else:
                    values = []
                    v = float(min_val)
                    while v <= float(max_val) + float(step) * 0.01:
                        values.append(round(v, 10))
                        v += float(step)
                    self._axes[name] = values
            else:
                raise ValueError(f"Invalid param spec for '{name}': {spec}")

    @property
    def total_combinations(self) -> int:
        if not self._axes:
            return 1
        result = 1
        for values in self._axes.values():
            result *= len(values)
        return result

    def grid(self) -> list[dict]:
        if not self._axes:
            return [{}]
        names = list(self._axes.keys())
        value_lists = [self._axes[n] for n in names]
        return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]
