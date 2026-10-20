"""Stability map over the parasitic reflector's phase and power reflectance, with and without self-injection.

Uses the model in transient_response_noise.py unchanged. For each configuration (a coupler_1 value; 0 means no
self-injection) and each grid point it sets r_ext2 and phase_2, runs a turn-on, and measures sigma_P / <P> of the
far-exit power after settle_time.

Axes:
  phase_2  extra round-trip phase of the parasitic arm, 0..2pi. frequency * tau_ext2 is held fixed, so this covers
           every parasitic feedback phase.
  R_ext2   power reflectance of the parasitic reflector, r_ext2^2, in dB. The feedback reaching the laser facet is
           weaker by the coupler: 20 log10(n^2 r_ext2) = R_ext2 + 40 log10(n) dB, with n = sqrt(1 - coupler_1^2).

Lives in its own folder; every file it writes goes into that folder (OUTPUT_DIR), wherever it is run from.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # This folder: all plots and data are saved here
sys.path.insert(0, os.path.dirname(OUTPUT_DIR))  # The model lives one folder up

import transient_response_noise as model


def run_point(args):
    """One grid point: returns (config, i, j, sigma_P / <P>, mean far-exit power in mW)."""
    c, i, j, coupler_1, R_dB, phase_2, phase_1, I, frequency, total_time, settle_time, dt, seed = args
    model.r_ext2 = 10 ** (R_dB / 20)  # R_dB = 10 log10(r_ext2^2); read by the model at run time
    laser = model.Laser(coupler_1=coupler_1, seed=seed, phase_1=phase_1, phase_2=phase_2)
    laser.dt = dt
    S, N, phi = laser.run(I, frequency, total_time)
    P = model.smooth(model.far_exit_power_mW(laser, S, phi), laser.dt)[int(settle_time / laser.dt):]
    return c, i, j, model.intensity_fluctuation(P), P.mean()


def sweep_phase_reflection(configs, levels_dB, phases, I, frequency, total_time, settle_time, dt, phase_1=0.0,
                           seed=0, workers=None):
    """Runs every (configuration, R_ext2, phase_2) point in parallel.

    configs: list of (label, coupler_1). Returns fluctuation and mean power arrays of shape
    (len(configs), len(levels_dB), len(phases)). Every run gets its own seed, so the maps are repeatable.
    """
    jobs = []
    for c, (_, coupler_1) in enumerate(configs):
        for i, R_dB in enumerate(levels_dB):
            for j, p2 in enumerate(phases):
                k = (c * len(levels_dB) + i) * len(phases) + j
                jobs.append((c, i, j, coupler_1, R_dB, p2, phase_1, I, frequency, total_time, settle_time, dt, seed + k))

    fluctuation = np.full((len(configs), len(levels_dB), len(phases)), np.nan)
    mean_power = np.full_like(fluctuation, np.nan)
    start = time.time()
    with Pool(workers or os.cpu_count()) as pool:
        for k, (c, i, j, f, p) in enumerate(pool.imap_unordered(run_point, jobs)):
            fluctuation[c, i, j], mean_power[c, i, j] = f, p
            print(f"[{k + 1}/{len(jobs)}] {configs[c][0]}: R_ext2 = {levels_dB[i]:.1f} dB, "
                  f"phase_2 = {phases[j] / np.pi:.2f} pi  fluctuation {f:.2e}  ({time.time() - start:.0f} s)")
    return fluctuation, mean_power


def worst_case_limit(levels_dB, fluctuation, stable_below):
    """Highest R_ext2 up to which every phase is stable (all lower levels too), or None if the lowest level fails."""
    all_stable = np.all(fluctuation < stable_below, axis=1)
    if not all_stable[0]:
        return None
    first_fail = np.flatnonzero(~all_stable)
    return levels_dB[first_fail[0] - 1] if len(first_fail) else levels_dB[-1]


def plot_maps(configs, levels_dB, phases, fluctuation, I, total_time, settle_time, dt, stable_below=1e-2,
              filename=os.path.join(OUTPUT_DIR, "parasitic_phase_sweep.png")):
    """One colour map of sigma_P / <P> per configuration."""

    surface = "#fcfcfb"
    ink = "#0b0b0b"
    ink_2 = "#52514e"
    ink_3 = "#8a8984"
    grid = "#e6e5e1"
    blue, orange = "#2a78d6", "#eb6834"

    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": grid,
        "axes.labelcolor": ink_2,
        "xtick.color": ink_2,
        "ytick.color": ink_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
    })

    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("stability", ["#dcebfa", blue, "#f6c9b3", orange, "#8f2d0b"])
    # Centre the colour change on the stability cutoff: blue below it, orange above
    norm = LogNorm(stable_below / 3, stable_below * 30)

    fig, axes = plt.subplots(1, len(configs), figsize=(8 * len(configs), 8.5), sharey=True, facecolor=surface,
                             squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.07, right=0.92, top=0.78, bottom=0.1, wspace=0.08)

    phase_pi = np.asarray(phases) / np.pi
    for c, ((label, coupler_1), ax) in enumerate(zip(configs, axes)):
        ax.set_facecolor(surface)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        f = fluctuation[c]
        mesh = ax.pcolormesh(phase_pi, levels_dB, np.clip(f, 1e-12, None), cmap=cmap, norm=norm,
                             shading="nearest", edgecolors=surface, linewidth=0.5)
        stable = f < stable_below

        # Worst case over phase: the parasitic reflection this configuration tolerates whatever its phase
        limit = worst_case_limit(levels_dB, f, stable_below)
        if limit is not None:
            step = levels_dB[1] - levels_dB[0] if len(levels_dB) > 1 else 0
            ax.axhline(limit + step / 2, color=ink, linewidth=1, linestyle=(0, (4, 4)))  # Edge between rows
            limit_text = f"stable at every phase up to R_ext2 = {limit:g} dB"
        else:
            limit_text = f"unstable at some phase even at {levels_dB[0]:g} dB"

        n = np.sqrt(1 - coupler_1**2)
        injection = (f"self-injection {20 * np.log10(coupler_1**2 * model.r_ext1):.1f} dB" if coupler_1 > 0
                     else "no self-injection")
        ax.text(0, 1.1, label, transform=ax.transAxes, color=ink, fontsize=14, fontweight="bold", va="bottom")
        ax.text(0, 1.035, f"coupler_1 = {coupler_1:g} ({injection})  ·  facet sees R_ext2 {40 * np.log10(n):+.1f} dB\n"
                f"{limit_text}  ·  {stable.sum()} of {stable.size} stable",
                transform=ax.transAxes, color=ink_2, fontsize=11, va="bottom")
        ax.set_xlabel("Parasitic reflection phase  φ₂ / π")
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
    axes[0].set_ylabel("Parasitic reflector power reflectance  R_ext2 (dB)")

    cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label(r"$\sigma_P\,/\,\langle P\rangle$  at the far exit")

    fig.text(0.07, 0.95, "Stability map  ·  parasitic reflection phase vs strength", color=ink, fontsize=20,
             fontweight="bold")
    fig.text(0.07, 0.915,
             f"{I * 1e3:.1f} mA  ·  r_ext1 = {model.r_ext1}, τ_ext1 = 25 ps, τ_ext2 = 300 ps  ·  "
             f"{fluctuation[0].size} runs per map of {total_time * 1e9:g} ns, measured after {settle_time * 1e9:g} ns, "
             f"dt = {dt * 1e12:g} ps  ·  stable: σ_P/⟨P⟩ < {stable_below:g}",
             color=ink_2, fontsize=12)

    fig.savefig(filename, dpi=150, facecolor=surface)
    return fig


if __name__ == "__main__":
    I = 50e-3  # Drive current in A
    laser_frequency = 1.22e15  # Optical angular frequency 2*pi*c/1550 nm in rad/s
    phase_1 = 0.0  # Extra round-trip phase on the loop-mirror arm in rad (held fixed)

    configs = [("Without self-injection", 0.0), ("With self-injection", 0.565)]  # (label, coupler_1)
    levels_dB = np.arange(-50, -4, 0.5)  # Parasitic reflector power reflectance R_ext2 in dB
    phases = np.linspace(0, 2 * np.pi, 48, endpoint=False)  # phase_2 values; 2pi is the same as 0
    total_time = 1e-8  # Simulation length per run in s
    settle_time = 5e-9  # Time discarded before measuring, in s
    dt = 5e-13  # Time step in s, as in tau_coupler_sweep/tau_coupler_sweep.py
    # sigma_P / <P> below this counts as stable. Without self-injection the noise floor alone is ~0.7-0.9%, so 1% flags
    # noise as instability; real instability jumps straight to 10-30%, and 3% separates the two cleanly
    stable_below = 3e-2
    seed = 0
    workers = None  # Parallel processes; None uses every CPU core

    fluctuation, mean_power = sweep_phase_reflection(configs, levels_dB, phases, I, laser_frequency, total_time,
                                                     settle_time, dt, phase_1, seed, workers)
    # Save the raw numbers so the maps can be re-plotted without re-running
    np.savez(os.path.join(OUTPUT_DIR, "parasitic_phase_sweep.npz"),labels=[c[0] for c in configs], couplers=[c[1] for c in configs],
             levels_dB=levels_dB, phases=phases, fluctuation=fluctuation, mean_power=mean_power, I=I,
             total_time=total_time, settle_time=settle_time, dt=dt, phase_1=phase_1)
    plot_maps(configs, levels_dB, phases, fluctuation, I, total_time, settle_time, dt, stable_below)
    plt.show()
