import math
from typing import Iterable, Sequence


class ConstantParamScheduler:
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, where: float) -> float:
        return self.value


class LinearParamScheduler:
    def __init__(self, start_value: float, end_value: float):
        self.start_value = float(start_value)
        self.end_value = float(end_value)

    def __call__(self, where: float) -> float:
        where = float(min(max(where, 0.0), 1.0))
        return self.start_value + where * (self.end_value - self.start_value)


class CosineParamScheduler:
    def __init__(self, start_value: float, end_value: float):
        self.start_value = float(start_value)
        self.end_value = float(end_value)

    def __call__(self, where: float) -> float:
        where = float(min(max(where, 0.0), 1.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * where))
        return self.end_value + (self.start_value - self.end_value) * cosine


class CompositeParamScheduler:
    def __init__(
        self,
        schedulers: Sequence,
        lengths: Sequence[float],
        interval_scaling: Iterable[str] = None,
    ):
        if len(schedulers) == 0:
            raise ValueError("CompositeParamScheduler requires at least one scheduler")
        if len(schedulers) != len(lengths):
            raise ValueError("schedulers and lengths must have the same length")

        self.schedulers = list(schedulers)
        self.lengths = [float(length) for length in lengths]
        total = sum(self.lengths)
        if total <= 0:
            raise ValueError("CompositeParamScheduler lengths must sum to > 0")
        self.lengths = [length / total for length in self.lengths]

        if interval_scaling is None:
            interval_scaling = ["rescaled"] * len(self.schedulers)
        self.interval_scaling = list(interval_scaling)
        if len(self.interval_scaling) != len(self.schedulers):
            raise ValueError("interval_scaling must match schedulers length")

    def __call__(self, where: float) -> float:
        where = float(min(max(where, 0.0), 1.0))
        start = 0.0
        for idx, (scheduler, length, scaling) in enumerate(
            zip(self.schedulers, self.lengths, self.interval_scaling)
        ):
            end = start + length
            is_last = idx == len(self.schedulers) - 1
            if where <= end or is_last:
                if scaling == "rescaled":
                    local_where = (where - start) / max(length, 1e-12)
                elif scaling == "fixed":
                    local_where = where
                else:
                    raise ValueError(f"Unsupported interval_scaling='{scaling}'")
                return scheduler(local_where)
            start = end
        return self.schedulers[-1](1.0)
