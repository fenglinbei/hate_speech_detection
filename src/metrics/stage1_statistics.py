"""Paired Stage-1 factorial, bootstrap, Holm, and gate utilities."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Any, Callable, Mapping, Sequence


STATISTICS_VERSION = "stage1-paired-statistics/v1"


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def percentile(values: Sequence[float], probability: float) -> float:
    """Deterministic linear-interpolation percentile (R/NumPy type 7)."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(_finite(value, "percentile value") for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def percentile_interval(values: Sequence[float], level: float = 0.95) -> list[float]:
    if not 0.0 < level < 1.0:
        raise ValueError("interval level must be in (0, 1)")
    tail = (1.0 - level) / 2.0
    return [percentile(values, tail), percentile(values, 1.0 - tail)]


def bootstrap_two_sided_p(replicates: Sequence[float]) -> float:
    """Frozen add-one two-sided bootstrap sign p-value."""

    if not replicates:
        raise ValueError("bootstrap p-value requires replicates")
    b = len(replicates)
    nonpositive = sum(float(value) <= 0.0 for value in replicates)
    nonnegative = sum(float(value) >= 0.0 for value in replicates)
    return min(1.0, 2.0 * min((1 + nonpositive) / (b + 1), (1 + nonnegative) / (b + 1)))


def bootstrap_indices(
    *,
    frame_size: int,
    seed: int,
    family: str,
    replicate: int,
) -> tuple[int, ...]:
    """One family-specific query draw shared across all cells and fixed seeds."""

    if frame_size <= 0:
        raise ValueError("bootstrap frame must be non-empty")
    if replicate < 0:
        raise ValueError("replicate must be non-negative")
    material = f"{seed}\x00{family}\x00{replicate}".encode("utf-8")
    stream_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    rng = random.Random(stream_seed)
    return tuple(rng.randrange(frame_size) for _ in range(frame_size))


def factorial_effects(cell_metrics: Mapping[str, float]) -> dict[str, float]:
    required = {"C0", "CL", "CD", "CLD"}
    if set(cell_metrics) != required:
        missing = sorted(required - set(cell_metrics))
        extra = sorted(set(cell_metrics) - required)
        raise ValueError(f"factorial cells must be exactly C0/CL/CD/CLD; missing={missing}, extra={extra}")
    c0 = _finite(cell_metrics["C0"], "C0")
    cl = _finite(cell_metrics["CL"], "CL")
    cd = _finite(cell_metrics["CD"], "CD")
    cld = _finite(cell_metrics["CLD"], "CLD")
    return {
        "L": cl - c0,
        "D": cd - c0,
        "interaction": cld - cl - cd + c0,
    }


def placebo_effects(cell_metrics: Mapping[str, float]) -> dict[str, float]:
    required = {"C0", "CL", "CD", "PL", "PD"}
    missing = sorted(required - set(cell_metrics))
    if missing:
        raise ValueError(f"placebo cells missing: {missing}")
    values = {key: _finite(cell_metrics[key], key) for key in required}
    return {
        "L_relevance_advantage": values["CL"] - values["PL"],
        "D_relevance_advantage": values["CD"] - values["PD"],
        "L_shape": values["PL"] - values["C0"],
        "D_shape": values["PD"] - values["C0"],
    }


def _validate_seed_cells(
    seed_cells: Mapping[int, Mapping[str, Sequence[Any]]],
    required_conditions: Sequence[str],
) -> int:
    if not seed_cells:
        raise ValueError("fixed-seed bootstrap requires at least one seed")
    frame_size: int | None = None
    required = set(required_conditions)
    for seed, cells in seed_cells.items():
        if set(cells) != required:
            raise ValueError(f"seed {seed} has a non-exact condition set")
        sizes = {len(cells[condition]) for condition in required_conditions}
        if len(sizes) != 1:
            raise ValueError(f"seed {seed} has unpaired condition lengths")
        current = next(iter(sizes))
        if frame_size is None:
            frame_size = current
        elif current != frame_size:
            raise ValueError("fixed seeds have different master-frame lengths")
    if not frame_size:
        raise ValueError("fixed-seed master frame must be non-empty")
    return frame_size


def fixed_seed_paired_bootstrap(
    *,
    seed_cells: Mapping[int, Mapping[str, Sequence[Any]]],
    metric: Callable[[Sequence[Any]], float],
    effect: Callable[[Mapping[str, float]], Mapping[str, float]],
    conditions: Sequence[str],
    family: str,
    bootstrap_seed: int = 42,
    replicates: int = 10_000,
    interval_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap queries once per replicate; never pool or resample seeds."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    frame_size = _validate_seed_cells(seed_cells, conditions)
    ordered_seeds = sorted(seed_cells)

    seed_points: dict[str, dict[str, float]] = {}
    for seed in ordered_seeds:
        cell_points = {
            condition: _finite(metric(seed_cells[seed][condition]), f"metric {seed}/{condition}")
            for condition in conditions
        }
        seed_points[str(seed)] = {
            key: _finite(value, f"effect {key}") for key, value in effect(cell_points).items()
        }
    effect_keys = tuple(next(iter(seed_points.values())).keys())
    for values in seed_points.values():
        if tuple(values.keys()) != effect_keys:
            raise ValueError("effect function returned inconsistent keys")
    points = {
        key: fmean(seed_points[str(seed)][key] for seed in ordered_seeds)
        for key in effect_keys
    }

    draws: dict[str, list[float]] = {key: [] for key in effect_keys}
    for replicate in range(replicates):
        indices = bootstrap_indices(
            frame_size=frame_size,
            seed=bootstrap_seed,
            family=family,
            replicate=replicate,
        )
        per_seed: dict[int, Mapping[str, float]] = {}
        for seed in ordered_seeds:
            cell_metrics = {
                condition: _finite(
                    metric([seed_cells[seed][condition][index] for index in indices]),
                    f"bootstrap metric {seed}/{condition}",
                )
                for condition in conditions
            }
            per_seed[seed] = effect(cell_metrics)
        for key in effect_keys:
            draws[key].append(fmean(float(per_seed[seed][key]) for seed in ordered_seeds))

    return {
        "schema_version": "stage1-fixed-seed-bootstrap/v1",
        "statistics_version": STATISTICS_VERSION,
        "family": family,
        "bootstrap_seed": bootstrap_seed,
        "replicates": replicates,
        "frame_size": frame_size,
        "fixed_seeds": ordered_seeds,
        "resampled_seeds": False,
        "shared_query_draw_within_family": True,
        "seed_points": seed_points,
        "effects": {
            key: {
                "point": points[key],
                "ci": percentile_interval(draws[key], interval_level),
                "p_two_sided": bootstrap_two_sided_p(draws[key]),
                "replicates": draws[key],
            }
            for key in effect_keys
        },
    }


def holm_adjust(
    tests: Sequence[Mapping[str, Any]],
    *,
    p_key: str = "p",
    tie_break_keys: Sequence[str] = ("contrast_order", "endpoint_order"),
) -> list[dict[str, Any]]:
    """Holm step-down adjustment with an explicit deterministic tie-break."""

    if not tests:
        return []
    materialized: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(tests):
        item = dict(raw)
        p = _finite(float(item[p_key]), p_key)
        if not 0.0 <= p <= 1.0:
            raise ValueError("p-values must be in [0, 1]")
        item[p_key] = p
        item["_input_ordinal"] = ordinal
        materialized.append(item)

    def order_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            float(item[p_key]),
            *(item.get(key, 0) for key in tie_break_keys),
            int(item["_input_ordinal"]),
        )

    ordered = sorted(materialized, key=order_key)
    running = 0.0
    m = len(ordered)
    for rank, item in enumerate(ordered, start=1):
        raw_adjusted = min(1.0, (m - rank + 1) * float(item[p_key]))
        running = max(running, raw_adjusted)
        item["holm_rank"] = rank
        item["holm_p"] = min(1.0, running)

    restored = sorted(ordered, key=lambda item: int(item["_input_ordinal"]))
    for item in restored:
        item.pop("_input_ordinal", None)
    return restored


def directional_stability(
    seed_effects: Mapping[int, float],
    *,
    direction: str,
    sesoi: float,
    minimum_seeds: int = 2,
) -> dict[str, Any]:
    """Apply the pre-registered 2/3 direction and reversal guard."""

    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be positive or negative")
    if sesoi < 0:
        raise ValueError("SESOI must be non-negative")
    values = {int(seed): _finite(value, f"seed {seed} effect") for seed, value in seed_effects.items()}
    if not values:
        raise ValueError("directional stability needs seed effects")
    if direction == "positive":
        count = sum(value > 0.0 for value in values.values())
        reversals = sorted(seed for seed, value in values.items() if value <= -sesoi)
    else:
        count = sum(value < 0.0 for value in values.values())
        reversals = sorted(seed for seed, value in values.items() if value >= sesoi)
    return {
        "direction": direction,
        "minimum_seeds": minimum_seeds,
        "seeds_in_direction": count,
        "opposite_sesoi_seed_ids": reversals,
        "passed": count >= minimum_seeds and not reversals,
    }


def classify_behavior_test(
    *,
    point: float,
    ci: Sequence[float],
    holm_p: float,
    seed_effects: Mapping[int, float],
    sesoi: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Classify positive/harm evidence without cancelling trade-offs."""

    if len(ci) != 2 or ci[0] > ci[1]:
        raise ValueError("CI must be [lower, upper]")
    point = _finite(point, "point")
    lower = _finite(ci[0], "ci lower")
    upper = _finite(ci[1], "ci upper")
    holm_p = _finite(holm_p, "holm p")
    positive_stat = point >= sesoi and lower > 0.0 and holm_p < alpha
    harm_stat = point <= -sesoi and upper < 0.0 and holm_p < alpha
    positive_stability = directional_stability(
        seed_effects,
        direction="positive",
        sesoi=sesoi,
    )
    harm_stability = directional_stability(
        seed_effects,
        direction="negative",
        sesoi=sesoi,
    )
    return {
        "stat_positive": positive_stat,
        "stat_harm": harm_stat,
        "positive_stability": positive_stability,
        "harm_stability": harm_stability,
        "confirmatory_positive": positive_stat and positive_stability["passed"],
        "confirmatory_harm": harm_stat and harm_stability["passed"],
    }


def paired_dz(values: Sequence[float]) -> float:
    """Paired standardised mean with the frozen degeneracy policy."""

    clean = [_finite(value, "paired delta") for value in values]
    if not clean:
        raise ValueError("paired d_z requires at least one delta")
    mean = fmean(clean)
    if len(clean) == 1:
        if mean == 0.0:
            return 0.0
        raise ValueError("non-zero d_z is undefined for a one-item cohort")
    sd = stdev(clean)
    if sd == 0.0:
        if mean == 0.0:
            return 0.0
        raise ValueError("degenerate paired deltas: sd=0 with non-zero mean")
    return mean / sd


def fixed_seed_margin_bootstrap(
    *,
    seed_cells: Mapping[int, Mapping[str, Sequence[float]]],
    left_condition: str,
    right_condition: str,
    eligibility_mask: Sequence[bool],
    family: str,
    bootstrap_seed: int = 42,
    replicates: int = 10_000,
    interval_level: float = 0.95,
) -> dict[str, Any]:
    """Frozen paired-margin reducer over one shared master query frame.

    The family-specific bootstrap ordinal draw is made on the full master
    frame, then the prediction-before-scoring eligibility mask is applied.
    Raw mean deltas drive CI/p-values; paired ``d_z`` is a separately reported
    SESOI statistic and follows :func:`paired_dz`'s fail-closed degeneracy
    policy.
    """

    frame_size = _validate_seed_cells(
        seed_cells, (left_condition, right_condition)
    )
    if len(eligibility_mask) != frame_size or any(
        not isinstance(value, bool) for value in eligibility_mask
    ):
        raise ValueError("margin eligibility mask must be a boolean master-frame mask")
    if sum(eligibility_mask) < 2:
        raise ValueError("paired d_z requires at least two eligible master-frame queries")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    ordered_seeds = sorted(seed_cells)

    def deltas(seed: int, indices: Sequence[int]) -> list[float]:
        result = []
        for index in indices:
            if eligibility_mask[index]:
                result.append(
                    _finite(
                        float(seed_cells[seed][left_condition][index])
                        - float(seed_cells[seed][right_condition][index]),
                        f"margin delta {seed}/{index}",
                    )
                )
        return result

    full_indices = tuple(range(frame_size))
    seed_points: dict[str, dict[str, float | int]] = {}
    for seed in ordered_seeds:
        values = deltas(seed, full_indices)
        seed_points[str(seed)] = {
            "raw_mean": fmean(values),
            "d_z": paired_dz(values),
            "effective_n": len(values),
        }
    raw_point = fmean(float(seed_points[str(seed)]["raw_mean"]) for seed in ordered_seeds)
    dz_point = fmean(float(seed_points[str(seed)]["d_z"]) for seed in ordered_seeds)

    raw_draws: list[float] = []
    dz_draws: list[float] = []
    effective_n: list[dict[str, int]] = []
    for replicate in range(replicates):
        indices = bootstrap_indices(
            frame_size=frame_size,
            seed=bootstrap_seed,
            family=family,
            replicate=replicate,
        )
        raw_by_seed: list[float] = []
        dz_by_seed: list[float] = []
        n_by_seed: dict[str, int] = {}
        for seed in ordered_seeds:
            values = deltas(seed, indices)
            if len(values) < 2:
                raise ValueError(
                    f"bootstrap replicate {replicate} has fewer than two eligible margins"
                )
            raw_by_seed.append(fmean(values))
            dz_by_seed.append(paired_dz(values))
            n_by_seed[str(seed)] = len(values)
        raw_draws.append(fmean(raw_by_seed))
        dz_draws.append(fmean(dz_by_seed))
        effective_n.append(n_by_seed)
    return {
        "schema_version": "stage1-fixed-seed-margin-bootstrap/v1",
        "statistics_version": STATISTICS_VERSION,
        "family": family,
        "contrast": f"{left_condition}-{right_condition}",
        "bootstrap_seed": bootstrap_seed,
        "replicates": replicates,
        "frame_size": frame_size,
        "eligible_count": sum(eligibility_mask),
        "fixed_seeds": ordered_seeds,
        "resampled_seeds": False,
        "shared_query_draw_within_family": True,
        "seed_points": seed_points,
        "raw": {
            "point": raw_point,
            "ci": percentile_interval(raw_draws, interval_level),
            "p_two_sided": bootstrap_two_sided_p(raw_draws),
            "replicates": raw_draws,
        },
        "d_z": {
            "point": dz_point,
            "ci": percentile_interval(dz_draws, interval_level),
            "replicates": dz_draws,
        },
        "effective_n_by_replicate": effective_n,
    }


def classify_margin_test(
    *,
    raw_ci: Sequence[float],
    raw_holm_p: float,
    dz_point: float,
    seed_dz: Mapping[int, float],
    sesoi_dz: float = 0.10,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply the registered raw-inference/d_z-SESOI margin decision rule."""

    if len(raw_ci) != 2 or raw_ci[0] > raw_ci[1]:
        raise ValueError("raw margin CI must be [lower, upper]")
    lower = _finite(raw_ci[0], "raw CI lower")
    upper = _finite(raw_ci[1], "raw CI upper")
    holm_p = _finite(raw_holm_p, "raw Holm p")
    dz = _finite(dz_point, "d_z point")
    stat_positive = lower > 0.0 and holm_p < alpha and dz >= sesoi_dz
    stat_harm = upper < 0.0 and holm_p < alpha and dz <= -sesoi_dz
    positive_stability = directional_stability(
        seed_dz, direction="positive", sesoi=sesoi_dz
    )
    harm_stability = directional_stability(
        seed_dz, direction="negative", sesoi=sesoi_dz
    )
    return {
        "stat_positive": stat_positive,
        "stat_harm": stat_harm,
        "positive_stability": positive_stability,
        "harm_stability": harm_stability,
        "confirmatory_positive": stat_positive and positive_stability["passed"],
        "confirmatory_harm": stat_harm and harm_stability["passed"],
    }


__all__ = [
    "STATISTICS_VERSION",
    "bootstrap_indices",
    "bootstrap_two_sided_p",
    "classify_behavior_test",
    "directional_stability",
    "factorial_effects",
    "fixed_seed_margin_bootstrap",
    "fixed_seed_paired_bootstrap",
    "holm_adjust",
    "paired_dz",
    "classify_margin_test",
    "percentile",
    "percentile_interval",
    "placebo_effects",
]
