"""Stability map over the loop-mirror delay tau_ext1 and the self-injection split coupler_1.

Uses the model in transient_response_noise.py unchanged: each grid point builds a Laser, sets its tau_ext1 and
coupler_1, runs a turn-on, and measures sigma_P / <P> of the far-exit power after settle_time.

Feedback phase: the round-trip phase of arm 1 is frequency * tau_ext1 + phase_1. At 1550 nm, changing tau_ext1 by
just 5 fs changes it by 2pi, so a delay sweep at fixed frequency would mostly be a random-phase sweep. With
hold_phase = True, phase_1 is set at each delay so that frequency * tau_ext1 + phase_1 equals its value at
reference_tau_ext1; the map then shows the effect of the delay alone. Set hold_phase = False for the raw behaviour.

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
    """One grid point: returns (i, j, sigma_P / <P>, mean far-exit power in mW)."""
    i, j, tau_ext1, coupler_1, phase_1, phase_2, I, frequency, total_time, settle_time, dt, seed = args
    laser = model.Laser(coupler_1=coupler_1, seed=seed, phase_1=phase_1, phase_2=phase_2)
    laser.dt = dt
    laser.tau_ext1 = tau_ext1
    S, N, phi = laser.run(I, frequency, total_time)
    P = model.smooth(model.far_exit_power_mW(laser, S, phi), laser.dt)[int(settle_time / laser.dt):]
    return i, j, model.intensity_fluctuation(P), P.mean()


def sweep_tau_coupler(taus, couplers, I, frequency, total_time, settle_time, dt, phase_1=0.0, phase_2=0.0,
                      hold_phase=True, reference_tau_ext1=25e-12, seed=0, workers=None):
    """Runs every (tau_ext1, coupler_1) pair in parallel.

    Returns fluctuation and mean power arrays of shape (len(taus), len(couplers)).
    Run (i, j) uses seed + i * len(couplers) + j, so the map is repeatable.
    """
    jobs = []
    for i, tau in enumerate(taus):
        # Keep the arm-1 round-trip phase equal to its value at reference_tau_ext1 (see module docstring)
        p1 = phase_1 + frequency * (reference_tau_ext1 - tau) if hold_phase else phase_1
        p1 = np.mod(p1, 2 * np.pi)
        for j, m in enumerate(couplers):
            jobs.append((i, j, tau, m, p1, phase_2, I, frequency, total_time, settle_time, dt,
                         seed + i * len(couplers) + j))

    fluctuation = np.full((len(taus), len(couplers)), np.nan)
    mean_power = np.full_like(fluctuation, np.nan)
    start = time.time()
    with Pool(workers or os.cpu_count()) as pool:
        for k, (i, j, f, p) in enumerate(pool.imap_unordered(run_point, jobs)):
            fluctuation[i, j], mean_power[i, j] = f, p
            print(f"[{k + 1}/{len(jobs)}] tau_ext1 = {taus[i] * 1e12:.1f} ps, coupler_1 = {couplers[j]:.3f}  "
                  f"fluctuation {f:.2e}  ({time.time() - start:.0f} s)")
    return fluctuation, mean_power


def plot_map(taus, couplers, fluctuation, I, total_time, settle_time, dt, hold_phase, stable_below=1e-2,
             filename=os.path.join(OUTPUT_DIR, "tau_coupler_sweep.png")):
    """Colour map of sigma_P / <P> over tau_ext1 and coupler_1."""

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

    fig, ax = plt.subplots(figsize=(14, 9), facecolor=surface)
    fig.subplots_adjust(left=0.09, right=0.9, top=0.8, bottom=0.1)
    ax.set_facecolor(surface)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    tau_ps = np.asarray(taus) * 1e12
    mesh = ax.pcolormesh(couplers, tau_ps, np.clip(fluctuation, 1e-12, None), cmap=cmap, norm=norm,
                         shading="nearest", edgecolors=surface, linewidth=0.5)
    stable = fluctuation < stable_below

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label(r"$\sigma_P\,/\,\langle P\rangle$  at the far exit")

    ax.set_xlabel("coupler_1  (field split m into the self-injection path)")
    ax.set_ylabel("τ_ext1  (loop-mirror round trip, ps)")

    # Self-injection level on a second x scale (same quantity, relabelled): 20 log10(m^2 r_ext1)
    to_dB = lambda m: 20 * np.log10(np.clip(m, 1e-6, None) ** 2 * model.r_ext1)
    from_dB = lambda dB: np.sqrt(10 ** (dB / 20) / model.r_ext1)
    top = ax.secondary_xaxis("top", functions=(to_dB, from_dB))
    top.set_xticks([-30, -20, -15, -10, -6, -3])
    top.xaxis.set_major_formatter(lambda v, pos: f"{v:g} dB")
    top.tick_params(colors=ink_2, length=0)
    top.spines["top"].set_visible(False)

    fig.text(0.09, 0.94, "Stability map  ·  loop-mirror delay vs self-injection strength", color=ink, fontsize=20,
             fontweight="bold")
    phase_note = "arm-1 feedback phase held fixed" if hold_phase else "arm-1 feedback phase varies with τ_ext1"
    fig.text(0.09, 0.895,
             f"{I * 1e3:.1f} mA  ·  r_ext1 = {model.r_ext1}, r_ext2 = {model.r_ext2}, τ_ext2 = 300 ps  ·  {phase_note}  ·  "
             f"{fluctuation.size} runs of {total_time * 1e9:g} ns, measured after {settle_time * 1e9:g} ns, dt = {dt * 1e12:g} ps\n"
             f"stable: σ_P/⟨P⟩ < {stable_below:g}  ·  {stable.sum()} of {stable.size} stable  ·  "
             f"top axis: self-injection level 20 log10(m² r_ext1)",
             color=ink_2, fontsize=12, va="top")

    fig.savefig(filename, dpi=150, facecolor=surface)
    return fig


if __name__ == "__main__":
    I = 50e-3  # Drive current in A
    laser_frequency = 1.22e15  # Optical angular frequency 2*pi*c/1550 nm in rad/s
    phase_1 = 0.0  # Extra round-trip phase on the loop-mirror arm in rad
    phase_2 = 0.0  # Extra round-trip phase on the parasitic arm in rad
    hold_phase = True  # Keep the arm-1 feedback phase fixed while tau_ext1 changes (see module docstring)
    reference_tau_ext1 = 25e-12  # Delay whose feedback phase is held

    taus = np.linspace(20e-12, 35e-12, 29)  # tau_ext1 values in s
    couplers = np.linspace(0.2, 0.9, 31)  # coupler_1 values
    total_time = 1e-8  # Simulation length per run in s
    settle_time = 5e-9  # Time discarded before measuring, in s
    dt = 5e-13  # Time step in s; 0.5 ps keeps a 380-run map manageable (dt = 1 ps had 1.3e-4 error vs 0.01 ps)
    seed = 0
    workers = None  # Parallel processes; None uses every CPU core

    fluctuation, mean_power = sweep_tau_coupler(taus, couplers, I, laser_frequency, total_time, settle_time, dt,
                                                phase_1, phase_2, hold_phase, reference_tau_ext1, seed, workers)
    # Save the raw numbers so the map can be re-plotted without re-running
    np.savez(os.path.join(OUTPUT_DIR, "tau_coupler_sweep.npz"),taus=taus, couplers=couplers, fluctuation=fluctuation, mean_power=mean_power,
             I=I, total_time=total_time, settle_time=settle_time, dt=dt, hold_phase=hold_phase)
    plot_map(taus, couplers, fluctuation, I, total_time, settle_time, dt, hold_phase)
    plt.show()
