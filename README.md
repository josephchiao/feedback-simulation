# Feedback simulation

Rate-equation simulation of a semiconductor laser coupled to a photonic integrated circuit with two external
reflectors:

- **Self-injection loop mirror:** intentional, 25 ps round trip.
- **Parasitic reflection:** unwanted, 300 ps round trip.

It's used to study how much self-injection is needed to keep the laser stable against parasitic feedback. The model
follows the supplementary section S1 of the reference paper (Eqs. S1.3–S1.9, Table S1). It is integrated with RK4,
and one version adds Langevin noise.

Model details, deviations from the paper, validation results and open issues are in
[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

## Setup

Requires Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

| Path | What it is |
|---|---|
| `transient_response.py` | Deterministic model. Runs one turn-on, or a sweep over the self-injection split `coupler_1`. |
| `transient_response_noise.py` | The same model with Langevin noise. The sweep scripts below use this one. |
| `tau_coupler_sweep/` | Stability map: loop-mirror delay `tau_ext1` vs `coupler_1` |
| `parasitic_phase_sweep/` | Stability map: parasitic reflection phase vs reflectance, with and without self-injection |
| `experiment_simulation_1/` | Copy of the parasitic phase sweep, for a separate experiment |
| `coupler_sweep*.png`, `coupler_sweep*_transients/` | Outputs of the `coupler_1` sweeps in the two model files |

## Running

Settings for each script are in its `if __name__ == "__main__":` block.

```bash
python transient_response.py           # sweep = True/False picks a coupler_1 sweep or a single run
python transient_response_noise.py
python tau_coupler_sweep/tau_coupler_sweep.py
python parasitic_phase_sweep/parasitic_phase_sweep.py
```

Where the outputs go:

- **The two model files** write their plots to the current directory, so run them from the repository root.
- **The scripts in subfolders** save their `.png` and `.npz` outputs in their own folder, wherever they're run from.
- **The sweeps** run in parallel on every CPU core. A 768-run parasitic phase map takes about 3 minutes on 8 cores.

The `.npz` files hold the raw sweep results, so a map can be re-plotted without re-running. For example:

```python
import sys, numpy as np
sys.path.insert(0, "parasitic_phase_sweep")
import parasitic_phase_sweep as pps

d = np.load("parasitic_phase_sweep/parasitic_phase_sweep.npz")
pps.plot_maps(list(zip(d["labels"].tolist(), d["couplers"].tolist())), d["levels_dB"], d["phases"],
              d["fluctuation"], float(d["I"]), float(d["total_time"]), float(d["settle_time"]), float(d["dt"]),
              stable_below=3e-2)
```
