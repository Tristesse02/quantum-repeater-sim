"""Simulation for a simple entanglement-generation project.

Compares Barrett-Kok and single-click protocols for:
1. a direct Alice-Bob architecture with a midpoint heralding station, and
2. a one-repeater architecture with two elementary links attempted in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
ATTENUATION_LENGTH_KM = 22.0
ATTEMPT_RATE_HZ = 1_000_000.0
DISTANCES_KM = np.arange(10.0, 251.0, 10.0)
ALPHAS = (0.05, 0.10, 0.20)
MONTE_CARLO_TRIALS = 20_000
RNG_SEED = 6904

# Consistent colors and markers used across all plots.
_COLORS: dict[tuple[str, float | None], str] = {
    ("Barrett-Kok", None): "steelblue",
    ("Single-click", 0.05): "forestgreen",
    ("Single-click", 0.10): "darkorange",
    ("Single-click", 0.20): "crimson",
}
_ARCH_MARKER = {"direct": "o", "one-repeater": "s"}
_ARCH_LINE   = {"direct": "-", "one-repeater": "--"}


def _color_for(row: "ProtocolPoint") -> str:
    return _COLORS.get((row.protocol, row.alpha), "gray")


def _marker_for(row: "ProtocolPoint") -> str:
    return _ARCH_MARKER.get(row.architecture, "o")


def _line_for(row: "ProtocolPoint") -> str:
    return _ARCH_LINE.get(row.architecture, "-")


@dataclass(frozen=True)
class ProtocolPoint:
    distance_km: float
    architecture: str
    protocol: str
    alpha: float | None
    photon_path_km: float
    eta: float
    link_success_probability: float
    end_to_end_fidelity: float
    expected_attempts_analytic: float
    expected_attempts_simulated: float
    expected_seconds_analytic: float
    expected_seconds_simulated: float


def transmissivity(path_length_km: float, attenuation_length_km: float = ATTENUATION_LENGTH_KM) -> float:
    """Optical fiber transmissivity eta(L) = exp(-L / L_att)."""
    return float(np.exp(-path_length_km / attenuation_length_km))


def barrett_kok_success(eta: float) -> float:
    """Low-transmissivity double-heralding model from Week 6."""
    return eta**2 / 2.0


def single_click_success(eta: float, alpha: float) -> float:
    """Single-click success probability from Week 6."""
    return 2.0 * alpha * eta - (alpha**2) * (eta**2)


def protocol_success(protocol: str, eta: float, alpha: float | None) -> float:
    if protocol == "Barrett-Kok":
        return barrett_kok_success(eta)
    if protocol == "Single-click":
        if alpha is None:
            raise ValueError("single-click protocol requires alpha")
        return single_click_success(eta, alpha)
    raise ValueError(f"Unknown protocol: {protocol}")


def protocol_fidelity(protocol: str, alpha: float | None) -> float:
    if protocol == "Barrett-Kok":
        return 1.0
    if protocol == "Single-click":
        if alpha is None:
            raise ValueError("single-click protocol requires alpha")
        return 1.0 - alpha
    raise ValueError(f"Unknown protocol: {protocol}")


def werner_swap_fidelity(elementary_fidelity: float) -> float:
    """Approximate post-swap fidelity using the Week 9 Werner-parameter product rule.

    The single-click state is not automatically Werner. This optional estimate treats
    each elementary state as if it had first been twirled to a Werner state with the
    same fidelity, then applies w_post = w_1 w_2.
    """
    werner_parameter = (4.0 * elementary_fidelity - 1.0) / 3.0
    post_swap_parameter = werner_parameter**2
    return (1.0 + 3.0 * post_swap_parameter) / 4.0


def expected_attempts_direct(p_success: float) -> float:
    return 1.0 / p_success


def expected_attempts_one_repeater(p_success: float) -> float:
    """Expected max of two iid geometric random variables with success probability p."""
    expected_min = 1.0 / (1.0 - (1.0 - p_success) ** 2)
    return 2.0 / p_success - expected_min


def simulate_direct_attempts(rng: np.random.Generator, p_success: float, trials: int) -> float:
    samples = rng.geometric(p_success, size=trials)
    return float(samples.mean())


def simulate_repeater_attempts(rng: np.random.Generator, p_success: float, trials: int) -> float:
    left = rng.geometric(p_success, size=trials)
    right = rng.geometric(p_success, size=trials)
    return float(np.maximum(left, right).mean())


def build_dataset(use_werner_swap_for_repeater: bool = True) -> list[ProtocolPoint]:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[ProtocolPoint] = []

    protocol_settings: list[tuple[str, float | None]] = [("Barrett-Kok", None)]
    protocol_settings.extend(("Single-click", alpha) for alpha in ALPHAS)

    for distance in DISTANCES_KM:
        for protocol, alpha in protocol_settings:
            for architecture in ("direct", "one-repeater"):
                if architecture == "direct":
                    photon_path = distance / 2.0
                    eta = transmissivity(photon_path)
                    p_success = protocol_success(protocol, eta, alpha)
                    expected_attempts = expected_attempts_direct(p_success)
                    simulated_attempts = simulate_direct_attempts(rng, p_success, MONTE_CARLO_TRIALS)
                    fidelity = protocol_fidelity(protocol, alpha)
                else:
                    photon_path = distance / 4.0
                    eta = transmissivity(photon_path)
                    p_success = protocol_success(protocol, eta, alpha)
                    expected_attempts = expected_attempts_one_repeater(p_success)
                    simulated_attempts = simulate_repeater_attempts(rng, p_success, MONTE_CARLO_TRIALS)
                    elementary_fidelity = protocol_fidelity(protocol, alpha)
                    if use_werner_swap_for_repeater:
                        fidelity = werner_swap_fidelity(elementary_fidelity)
                    else:
                        fidelity = elementary_fidelity

                rows.append(
                    ProtocolPoint(
                        distance_km=distance,
                        architecture=architecture,
                        protocol=protocol,
                        alpha=alpha,
                        photon_path_km=photon_path,
                        eta=eta,
                        link_success_probability=p_success,
                        end_to_end_fidelity=fidelity,
                        expected_attempts_analytic=expected_attempts,
                        expected_attempts_simulated=simulated_attempts,
                        expected_seconds_analytic=expected_attempts / ATTEMPT_RATE_HZ,
                        expected_seconds_simulated=simulated_attempts / ATTEMPT_RATE_HZ,
                    )
                )

    return rows


def save_csv(rows: list[ProtocolPoint], path: Path) -> None:
    header = ",".join(ProtocolPoint.__dataclass_fields__.keys())
    lines = [header]
    for row in rows:
        values = []
        for key in ProtocolPoint.__dataclass_fields__.keys():
            value = getattr(row, key)
            values.append("" if value is None else str(value))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def label_for(row: ProtocolPoint) -> str:
    if row.protocol == "Barrett-Kok":
        return f"{row.protocol}, {row.architecture}"
    return f"{row.protocol} alpha={row.alpha:.2f}, {row.architecture}"


def plot_metric(rows: list[ProtocolPoint], metric: str, ylabel: str, output: Path, log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped: dict[tuple[str, str, float | None], list[ProtocolPoint]] = {}
    for row in rows:
        key = (row.protocol, row.architecture, row.alpha)
        grouped.setdefault(key, []).append(row)

    for group_rows in grouped.values():
        group_rows.sort(key=lambda item: item.distance_km)
        distances = [row.distance_km for row in group_rows]
        values = [getattr(row, metric) for row in group_rows]
        rep = group_rows[0]
        ax.plot(
            distances, values,
            linestyle=_line_for(rep),
            marker=_marker_for(rep),
            color=_color_for(rep),
            markersize=4,
            label=label_for(rep),
        )

    ax.set_xlabel("Total Alice–Bob distance (km)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=9, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rate_fidelity(rows: list[ProtocolPoint], distances_km: tuple[float, ...], output: Path) -> None:
    """Scatter plot of entanglement rate vs end-to-end fidelity.

    Each (protocol, alpha, architecture) configuration appears as a set of
    points — one per distance — so the reader can see how rate degrades with
    distance while fidelity stays constant under this model.
    """
    from matplotlib.lines import Line2D

    subset = [r for r in rows if r.distance_km in distances_km]
    # Alpha levels map distances to opacity so nearby distances are opaque.
    dist_sorted = sorted(distances_km)
    alpha_map = {d: 0.35 + 0.65 * i / max(len(dist_sorted) - 1, 1) for i, d in enumerate(dist_sorted)}

    fig, ax = plt.subplots(figsize=(9, 6))

    plotted_labels: set[tuple[str, float | None, str]] = set()
    for row in subset:
        rate = 1.0 / row.expected_seconds_analytic
        fid  = row.end_to_end_fidelity
        key  = (row.protocol, row.alpha, row.architecture)
        ax.scatter(
            rate, fid,
            marker=_marker_for(row),
            color=_color_for(row),
            s=80,
            alpha=alpha_map[row.distance_km],
            zorder=5,
            label=label_for(row) if key not in plotted_labels else "_nolegend_",
        )
        plotted_labels.add(key)
        ax.annotate(
            f"{row.distance_km:.0f} km",
            xy=(rate, fid),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
            color=_color_for(row),
            alpha=alpha_map[row.distance_km],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Entanglement rate (pairs / second)", fontsize=12)
    ax.set_ylabel("End-to-end fidelity", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    # Protocol/architecture legend
    ax.legend(fontsize=8, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    # Separate distance-opacity guide
    opacity_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               alpha=alpha_map[d], markersize=7, label=f"{d:.0f} km")
        for d in dist_sorted
    ]
    ax.add_artist(ax.legend(handles=opacity_handles, title="Distance", fontsize=8,
                            loc="lower right"))

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_ratio(rows: list[ProtocolPoint], output: Path) -> None:
    """Log-scale plot of T_direct / T_repeater vs distance for each protocol config."""
    direct_map  = {(r.protocol, r.alpha, r.distance_km): r for r in rows if r.architecture == "direct"}
    repeat_map  = {(r.protocol, r.alpha, r.distance_km): r for r in rows if r.architecture == "one-repeater"}

    combos = sorted({(r.protocol, r.alpha) for r in rows}, key=lambda x: (x[0], x[1] or 0))

    fig, ax = plt.subplots(figsize=(10, 6))

    for protocol, alpha in combos:
        points = []
        for d in sorted(DISTANCES_KM):
            key = (protocol, alpha, d)
            if key in direct_map and key in repeat_map:
                ratio = direct_map[key].expected_seconds_analytic / repeat_map[key].expected_seconds_analytic
                points.append((d, ratio))
        if not points:
            continue
        xs, ys = zip(*points)
        dummy = direct_map[(protocol, alpha, xs[0])]
        lbl = label_for(dummy).replace(", direct", "")
        ax.plot(xs, ys, color=_color_for(dummy), linestyle="-", marker="o", markersize=4, label=lbl)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Break-even (ratio = 1)")
    ax.set_xlabel("Total Alice–Bob distance (km)", fontsize=12)
    ax.set_ylabel("Speedup  $T_\\mathrm{direct}\\,/\\,T_\\mathrm{repeater}$", fontsize=12)
    ax.set_yscale("log")
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = build_dataset()
    save_csv(rows, RESULTS_DIR / "simulation_results.csv")

    plot_metric(
        rows,
        "link_success_probability",
        "Elementary-link success probability per attempt",
        RESULTS_DIR / "success_probability_vs_distance.png",
        log_y=True,
    )
    plot_metric(
        rows,
        "expected_seconds_analytic",
        "Expected waiting time (seconds)",
        RESULTS_DIR / "waiting_time_vs_distance.png",
        log_y=True,
    )
    plot_rate_fidelity(
        rows,
        distances_km=(50.0, 100.0, 150.0, 200.0),
        output=RESULTS_DIR / "rate_fidelity_tradeoff.png",
    )
    plot_speedup_ratio(rows, RESULTS_DIR / "speedup_ratio_vs_distance.png")

    selected = [row for row in rows if row.distance_km in (50.0, 100.0, 150.0, 200.0)]
    save_csv(selected, RESULTS_DIR / "selected_distance_summary.csv")
    print(f"Wrote {len(rows)} rows and plots to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
