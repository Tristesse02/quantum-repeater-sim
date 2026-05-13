# Performance Comparison of Barrett-Kok and Single-Click Entanglement Generation

## Goal

This project compares Barrett-Kok and single-click entanglement generation under optical loss for two simple architectures:

1. **Direct architecture:** Alice and Bob generate entanglement through one midpoint heralding station.
2. **One-repeater architecture:** A midpoint repeater splits the Alice-Bob distance into two elementary links. The two links are attempted in parallel, then a deterministic Bell-state measurement performs entanglement swapping.

The main performance metrics are success probability, fidelity, and expected waiting time.

## Lecture-Based Model

The fiber transmissivity model is

```text
eta(d) = exp(-d / L_att)
```

where `d` is the photon path length and `L_att = 22 km`.

For a total Alice-Bob distance `L`:

- Direct architecture: the heralding station is midway, so each photon travels `L / 2`.
- One-repeater architecture: each elementary link has length `L / 2`, with its own midpoint heralding station, so each photon travels `L / 4`.

The Week 6 formulas used in the simulation are:

```text
Barrett-Kok:
    p_success = eta^2 / 2
    F = 1

Single-click:
    p_success = 2 alpha eta - alpha^2 eta^2
    F = 1 - alpha
```

Here `alpha` is the bright-state population. Increasing `alpha` improves the generation rate but lowers fidelity.

For waiting time, each entanglement attempt is modeled as a Bernoulli trial. The expected direct waiting time in attempts is:

```text
E[T_direct] = 1 / p
```

For the one-repeater case, two independent elementary links are attempted in parallel. The repeater can swap only after both links have succeeded, so the waiting time is the maximum of two independent geometric random variables:

```text
E[T_repeater] = 2 / p - 1 / (1 - (1 - p)^2)
```

The script also runs Monte Carlo simulations to validate these expectations.

## Fidelity Assumption for Swapping

For Barrett-Kok, ideal operations give `F = 1` before and after swapping.

For single-click, the generated state is mixed. To reflect the Week 9 warning that swapping imperfect states reduces final quality, the script uses a Werner-state approximation for repeater-assisted single-click results:

```text
w = (4F - 1) / 3
w_post = w^2
F_post = (1 + 3w_post) / 4
```

This is a modeling approximation because the raw single-click state is not automatically Werner. It is useful as a simple way to include the fidelity penalty of swapping imperfect entanglement.

## How to Run

```powershell
python quantum_repeater_sim.py
```

Outputs are written to `results/`:

- `simulation_results.csv` — full dataset (200 rows, all distances and configurations)
- `selected_distance_summary.csv` — subset at 50, 100, 150, 200 km
- `success_probability_vs_distance.png` — elementary-link success probability vs distance (log scale)
- `waiting_time_vs_distance.png` — expected end-to-end waiting time vs distance (log scale)
- `rate_fidelity_tradeoff.png` — entanglement rate vs fidelity scatter at four distances
- `speedup_ratio_vs_distance.png` — repeater speedup ratio T\_direct / T\_repeater vs distance

## Expected Insights

Barrett-Kok gives the highest fidelity, but its double-heralding requirement makes its success probability much smaller.

Single-click succeeds more often, especially for larger `alpha`, but the price is reduced fidelity. This produces the intended rate-fidelity tradeoff.

The one-repeater architecture reduces the photon travel distance for each elementary attempt. This can substantially reduce waiting time at longer total distances, but it requires memory at the repeater and entanglement swapping. When imperfect elementary states are included, swapping can also reduce the final end-to-end fidelity.

## Limitations

The model assumes ideal detectors, no dark counts, no memory decoherence, no detector inefficiency beyond channel loss, no classical communication delay, and perfect local operations. These assumptions keep the project aligned with the proposal while leaving room for later extensions such as detector dark counts, memory cutoffs, or nonzero swap error.
