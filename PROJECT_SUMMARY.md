# Laser feedback simulation: project summary

Context for picking this work up in a new session. Last updated 2026-10-17.

## Goal

Co-op project at Integrated Laser. Simulate a semiconductor laser coupled to a photonic integrated circuit (PIC) with two external reflectors:

- **Mirror 1:** an intentional on-chip loop mirror that provides self-injection feedback, with a short delay (25 ps).
- **Mirror 2:** a parasitic reflection elsewhere on the PIC, with a long delay (300 ps).

The question is how strong the self-injection must be (the coupler split `coupler_1`, m) to stabilise the laser against the parasitic reflection.

The model follows the supplementary section S1 of a paper the user supplied:

- **Equations:** S1.3–S1.9. S1.6 is the two-reflector transfer function T. S1.7–S1.9 are the rate equations for S, N and φ.
- **Parameters:** Table S1.
- **Reference figures:** Figs. S1.2 and S1.3.

The paper's title is not recorded here; add it.

## Files

Everything is in `feedback simulation/`. The Python 3.13 venv is in `.venv`, and `requirements.txt` lists numpy and matplotlib.

| File | What it is |
|---|---|
| `transient_response.py` | Deterministic model: single run or `coupler_1` sweep, plus plots |
| `transient_response_noise.py` | Copy of the above with Langevin noise. It writes to separate output names. |
| `coupler_sweep.png`, `coupler_sweep_transients/` | Deterministic sweep: overview plot and one turn-on plot per `coupler_1` |
| `coupler_sweep_noise.png`, `coupler_sweep_noise_transients/` | The same outputs for the noisy model |

Run with `.venv/bin/python transient_response.py`. Settings live in the `__main__` block:

- `I`, `frequency`, `laser_frequency` and `total_time` apply to both modes.
- `sweep = True` runs a sweep over `sweep_couplers`, using `settle_time` and `transient_dir`.
- The noise file also has `seed`.

## Model as implemented

State: photon number S, carrier number N and optical phase φ. These are numbers, not densities, even though some variable names say "density".

```
T     = 1 + (1 - r2²)/r2 · (A1 + A2 + 2·A1·A2)                      # Eq. S1.6
Ai    = x_i / (1 + r2·x_i),  x_i = c_i² · r_ext,i · sqrt(S(t-τi)/S(t)) · exp(-j(ω·τi + φ(t) - φ(t-τi)))
        c_1 = coupler_1 (m),  c_2 = sqrt(1 - m²) (n)
dS/dt = [G(N,N_tr)/(1+εS) - 1/τ_ph + 2·ln|T|/τ_D]·S + β·N/τ_n
dN/dt = η·I/q - N/τ_n - G(N,N_tr)/(1+εS)·S
dφ/dt = ½·α_H·G(N,N_th) + arg(T)/τ_D
G(N,X)= G_N·(N - X)
```

Integration:

- **Time-stepping:** RK4 with dt = 0.1 ps. Each RK4 stage gets its own time.
- **Delays:** the delayed values S(t−τ) and φ(t−τ) come from stored history, linearly interpolated, and are 0 before t = 0.
- **Initial state:** S = 1e-9, N = 0, φ = 0. Each run is a step turn-on from zero current.

Output power:

```
P = output_coupling · hν · S · (1/τ_ph − 2 ln|T|/τ_D)
```

- `output_coupling = 0.5` is assumed; the paper doesn't give it.
- The 2 ln|T|/τ_D term subtracts photons the external cavity sends straight back.
- This formula matches the dS photon balance, so it conserves energy. Its known flaw is listed under open issues.

Current parameters (from Table S1 unless noted):

| Quantity | Value | Note |
|---|---|---|
| G_N | 1.13e4 1/s | |
| N_tr | 8.2e6 | |
| N_th | N_tr + 1/(G_N·τ_ph) = 2.06e7 | **Not** Table S1's 7.066e7; see deviations |
| ε | 4.58e-8 | |
| τ_ph | 7.15 ps | |
| τ_D | 6 ps | |
| τ_n | 0.33 ns | |
| β | 3.55e-5 | |
| η | 0.9 | |
| α_H | 4.86 | |
| r_2 | √0.3 | Laser front facet, field reflectivity |
| r_ext1 | 0.95 | User changed from 0.99 |
| r_ext2 | 0.05 | User changed from 0.15 |
| τ_ext1, τ_ext2 | 25 ps, 300 ps | |
| coupler_1 default | 0.565 | m² · r_ext ≈ −10 dB, the paper's level |
| ω (`laser_frequency`) | 1.22e15 rad/s | 2πc/1550 nm. ω·τ sets the feedback phase, which is effectively arbitrary. |
| λ | 1550 nm | |
| I | 50 mA | Solitary threshold is 11.1 mA |

Code layout (both files):

- **`Laser`:**
  - Integration: `rk4_step`, `step`, `run`.
  - Model: `gain`, `history`, `external_cavity_arm`, `external_cavity_transfer_function`, `derivative`.
  - Bookkeeping: `step` also stores `feedback_gain_history` (2 ln|T|/τ_D at each sample) for the power calculation.
- **Module functions:** `photons_to_power_mW`, `plot_run`, `power_extrema`, `intensity_fluctuation` (σ_P/⟨P⟩, standing in for the paper's RF-envelope measure), `sweep_coupler`, `plot_sweep`.
- **Plots:**
  - `plot_run` has three panels: power in mW, N, and Δν = (dφ/dt)/2π in GHz.
  - `plot_sweep` shows a bifurcation diagram (power extrema) and σ_P/⟨P⟩ on a log scale. Its top axis gives the self-injection level 20·log10(m²·r_ext1). Runs with σ_P/⟨P⟩ < 1% are shaded as stable.

### Noise version: what's different

- **When noise is applied:** after each deterministic RK4 step, one Euler–Maruyama Langevin kick is added. The diffusion coefficients follow Agrawal & Dutta, with R_sp = βN/τ_n:
  - ⟨F_S²⟩ = 2R_sp·S
  - ⟨F_φ²⟩ = R_sp/(2S)
  - ⟨F_N²⟩ = 2R_sp·S + 2N/τ_n
  - ⟨F_S F_N⟩ = −2R_sp·S
- **Code:** the kick is computed in `langevin_increment()`.
- **Options:** `Laser(seed=..., noise_scale=...)`. `noise_scale = 1` is the physical strength, not a tuned value, and `noise_scale = 0` turns the noise off.
- **S floor:** S is clamped to at least 1e-9.
- **Smoothing:** power and Δν go through `smooth()`, a 10 ps moving average (`detector_window`, roughly a 100 GHz detector) that stays unbiased at the edges. The sweep's measurements use the same averaged signal.
- **Masking:** Δν is hidden wherever P is below 1% of its peak, because the phase of an almost empty cavity is meaningless.

## Deviations from the paper, and why

1. **N_th for the phase equation** is 2.06e7, the solitary threshold, instead of 7.066e7. With 7.066e7, φ ramps at about −228 GHz. That drift is exact in principle, since it's equivalent to a shift in ω, but it ruins the delayed-phase lookup at practical time steps. The change only shifts the feedback phase by a constant.
2. **ln|T| instead of ln T** in dS/dt. ln T is complex; its imaginary part made S complex and negative.
3. **Which N goes where:** dS and dN use G(N, N_tr), and dφ uses G(N, N_th). The original code used N_th everywhere, which put the threshold at 44.8 mA.
4. **Output power conversion, the 0.5 split, and Δν plotting** were added here. They are not from the paper.

## Bugs fixed during the session (for history)

- **Phase difference outside `-1j`:** the exponent was `exp(-1j*ω*τ + φ - φ(t-τ))`, making φ − φ(t−τ) real. That was the cause of the original blow-up, with S reaching about 1e37.
- **RK4 plumbing:** the signature was wrong, `derivative` didn't take the stage state, and h was applied twice.
- **Stale state in T:** T read `self.S` and `self.phi` instead of each stage's values.
- **History lookups:** negative indices before t = τ; `int()` truncation; history not evaluated at the RK4 stage times. These were replaced by linear interpolation.
- **Time stamp:** `self.t` was advanced before the step instead of after it.

## Validation done

| Check | Result |
|---|---|
| Time-step convergence over the first 250 ps, vs dt = 0.01 ps | error 1.3e-4 at dt = 1 ps; 2e-6 at dt = 0.1 ps (was 190% before interpolation and the N_th change) |
| Refactored T vs the original 4-line S1.6 expression | matches to 10 digits |
| Turn-on transient, dt = 0.1 ps vs 0.025 ps | 2.7e-5 difference |
| Noise file with `noise_scale = 0` vs deterministic file | identical |
| Solitary linewidth from phase diffusion, 6 seeds | 3.53 ± 0.15 MHz; Schawlow–Townes with α and gain compression gives 3.34 MHz (0.12 MHz without α) |
| Linewidth with feedback, m = 0.635, single seed | ≈ 9 kHz vs ≈ 1.6 MHz solitary: self-injection narrowing reproduced |
| Solitary output at 26 mA | ≈ 10.6 mW, vs (I − I_th)·η·hν/q ≈ 10.7 mW |

A 10 ns run at dt = 0.1 ps takes about 4 s. dt = 1 ps is accurate enough (1.3e-4 error) and about 10× faster.

## Results so far (50 mA, one feedback phase, ω = 1.22e15)

**r_ext2 = 0.15** (parasitic arm about −20 dB), deterministic:

| `coupler_1` | Result |
|---|---|
| ≤ 0.40 | chaotic |
| ~0.45–0.60 | stable |
| ~0.61–0.655 | periodic |
| ~0.665–0.87 | stable, most stable around 0.67–0.76 |
| ≥ 0.88 | oscillating |

**r_ext2 = 0.05** (parasitic arm about −29 dB):

- **Deterministic:** stable from about 0.4 to 0.88. Unstable at 0.89–0.9, with a full-depth oscillation at about 40 GHz, i.e. 1/τ_ext1.
- **With noise:**

| `coupler_1` | σ_P/⟨P⟩ |
|---|---|
| ≤ 0.35 | 0.11–0.29, unstable |
| 0.4–0.85 | 0.55–0.78%, the laser's own noise floor (solitary is 0.70%) |
| 0.9 | 1.75%, just over the 1% cutoff |

- **Bistability at m = 0.9:** with noise, the laser starts in the 40 GHz oscillation, then at about 1.2 ns jumps to steady output at −22 GHz. The deterministic run stays oscillating, so two states coexist there and noise picks between them.

## Open issues and decisions pending

1. **Parameter inconsistency (decision pending).** Table S1's τ_ph gives a total cavity loss 1/τ_ph = 1.4e11 1/s. That's smaller than the front facet's own mirror loss, ln(1/R₂)/τ_D = 2.0e11 1/s. Consequences:
   - The feedback gain 2 ln|T|/τ_D can exceed the total loss.
   - The energy-conserving P formula then goes negative during strong-feedback transients (about −6 mW at m = 0.9).
   - A facet-scaled formula was tried and **reverted**: it stayed positive but reported 74.7 mW, more than the 36 mW that 50 mA can produce.
   - The dynamics inherit the inconsistency too: the feedback is probably too strong relative to the loss, which could shift the stable windows.

   Options:
   - (1) Make τ_ph consistent: 1/τ_ph = [ln(1/R₂) + ln(1/R₁)]/τ_D + internal loss. Needs the device's back-facet R₁ and internal loss α_i.
   - (2) Keep τ_ph and scale the feedback term by about 0.35, the front facet's share of the loss. This is a DFB-like assumption.
   - (3) Keep the current state and draw conclusions from S or σ_P/⟨P⟩, not absolute mW.

   The user hasn't chosen yet.
2. **Eq. S1.6 cross-term is suspect (unverified).** Deriving a shared waveguide with combined reflection R = a₁ + a₂ through S1.3 gives T = 1 + (1−r₂²)/r₂ · R/(1 + r₂R). The mixed term there is −2·r₂·a₁·a₂, not the paper's +2·a₁·a₂. The difference in the gain term is up to about 1.6e11 1/s, similar to 1/τ_ph. The code currently follows the paper.
3. **Quasi-static approximation.** S1.6 uses sqrt(S(t−τ)/S(t)) for every round trip. That breaks down during turn-on, when the ratio ranges from 1e-5 to 2.2 in 300 ps, and during oscillations whose period is about τ_ext1. Treat power in the first ~0.3 ns as qualitative. A proper fix is a field-based Lang–Kobayashi model that uses E(t−τ).
4. **Turn-on.** Each run steps from 0 to 50 mA with N = 0. That produces a real ~150 ps delay and a large relaxation spike, the solitary laser included. Not implemented yet: starting from the solitary steady state with the history pre-filled. This would remove the transient and shorten `settle_time`.
5. **Single feedback phase.** All results are for one value of ω·τ. A phase sweep, as in the paper's Fig. S1.2, hasn't been done.
6. **No Langevin noise in the deterministic file.** Use the noise file for stability maps. Its 1% stability cutoff is close to the 0.6–0.8% noise floor.
7. **Noise model limits.** The noise level scales with β. Pump-current, 1/f and thermal noise aren't modelled, so `noise_scale = 1` is a lower bound.
8. **Power location.** P is the net power at the laser facet, not the power at the PIC output (the paper's tunable tap and 1.5 dB loss aren't modelled).
9. **Cosmetic:**
   - The "stable below 0.01" label can overlap the curve in `plot_sweep`.
   - Some variable names say "density" but hold numbers.
   - The `gain()` argument is called `threshold` but receives N_tr for the S and N equations.

## Working preferences (user)

- Wants direct fixes. Wants problems verified with numbers before they're claimed.
- When the user says "present but don't change", diagnose only.
- Likes polished plots. The style uses a light surface, recessive grid, blue/orange/aqua series, panel titles and a stats line.
- Run a quick simulation to confirm before asserting physics.
