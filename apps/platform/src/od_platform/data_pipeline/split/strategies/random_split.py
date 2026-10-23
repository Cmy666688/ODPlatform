# apps/platform/src/od_platform/data_pipeline/split/strategies/random_split.py
"""随机划分:不按类别,种子固定即可复现。requires_labels=False。"""
import random

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.registry import SplitOptions, register_strategy
from od_platform.data_pipeline.split.strategies._common import (
    seeded_shuffled,
    three_way_counts,
    validate_rates,
)


@register_strategy(SplitStrategy.RANDOM, requires_labels=False)
def random_split(stems: list[str], options: SplitOptions) -> dict[str, list[str]]:
    validate_rates(options.train_rate, options.val_rate)  # 只校验,比例切片在 _common 里做
    n = len(stems)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    rng = random.Random(options.random_state)
    shuffled = seeded_shuffled(sorted(stems), rng)
    n_train, n_val, _ = three_way_counts(n, options.train_rate, options.val_rate)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
