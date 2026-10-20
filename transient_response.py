import os
import numpy as np
import matplotlib.pyplot as plt

c = 3e8  # Speed of light in m/s
q = 1.602e-19  # Elementary charge in C
h = 6.626e-34  # Planck constant in J*s
wavelength = 1550e-9  # Emission wavelength in m
output_coupling = 0.5  # Fraction of the net emitted power leaving through the front facet (assumed, not given in the paper)

differential_gain = 1.13e4
transparency_carrier_density = 8.2e6
epsilon = 4.58e-8 #Gain compression factor
tau_ph = 7.15e-12 #Photon lifetime
tau_D = 6e-12 #Laser cavity round-trip time
# Solitary threshold, where G(N, N_tr) = 1/tau_ph. Table S1's 7.066e7 is inconsistent with the other parameters and
# makes phi ramp at ~-228 GHz; referencing the phase equation here only shifts the feedback phase by a constant.
threshold_carrier_density = transparency_carrier_density + 1 / (differential_gain * tau_ph)
# Rate (1/s) at which a photon in the cavity leaves through the front facet with no external cavity, so the
# solitary front-facet emission is front_escape_rate * S. Tied to the output_coupling assumption until tau_ph is
# made consistent with the facet (see PROJECT_SUMMARY open issue 1); facet physics alone would give
# (1 - r_2^2) / tau_D = 1.17e11 1/s.
front_escape_rate = output_coupling / tau_ph

r_2 = np.sqrt(0.3) #Field reflectivity of the laser front facet (sqrt of R2 = 0.3)

r_ext1 = 0.95 #Field reflectivity of the intentional on-chip loop mirror (self-injection)
r_ext2 = 0.05 #Field reflectivity of the parasitic reflector

beta = 3.55e-5 # Spontaneous emission coupling coefficient
tau_n = 0.33e-9 #Carrier lifetime

eta = 0.9 #Quantum efficiency
alpha_H = 4.86


def photons_to_power_mW(S, feedback_gain=0.0):
    """Converts intracavity photon number S to optical output power in mW.

    Photons leave the cavity at S / tau_ph, but feedback_gain = 2 ln|T| / tau_D of them per photon per second
    are sent straight back by the external cavities, so the net emission rate is S * (1/tau_ph - feedback_gain).
    Each photon carries h * c / wavelength; output_coupling is the share of the net emission through the front facet.

    This matches the photon balance of the dS equation, so it conserves energy. It goes negative whenever
    feedback_gain > 1/tau_ph: Table S1's tau_ph gives a total cavity loss (1.4e11 1/s) smaller than the front facet's
    mirror loss ln(1/R2) / tau_D (2.0e11 1/s) alone, so strong feedback can exceed all the loss the model has.
    """
    return output_coupling * (h * c / wavelength) * S * (1 / tau_ph - feedback_gain) * 1e3


class Laser:

    def __init__(self, coupler_1=0.565, phase_1=0.0, phase_2=0.0):

        self.t = 0.0  # Time
        self.dt = 1e-13  # Time step

        self.S = 1e-9  # Number of photons in the active region
        self.N = 0.0  # Number of carriers in the active region
        self.phi = 0.0  # Phase of the optical field
        self.S_history = [1e-9]
        self.N_history = [0]
        self.phi_history = [0]
        self.feedback_gain_history = [0.0]  # 2 ln|T| / tau_D at each sample, for the net output power

        self.tau_ext1 = 25e-12  # External cavity round-trip time for mirror 1
        self.tau_ext2 = 300e-12  # External cavity round-trip time for mirror 2

        self.coupler_1 = coupler_1  # Field split into the self-injection path to mirror 1 (m in Eq. S1.6)
        self.coupler_2 = np.sqrt(1 - self.coupler_1 ** 2)  # Field split into the parasitic path to mirror 2 (n, m^2 + n^2 = 1)
        self.frequency = 0.0  # Optical angular frequency in rad/s; frequency * tau_ext sets the feedback phase
        # Extra round-trip phase (rad) on each arm, added to frequency * tau_ext. Sweeping these over 0..2pi covers
        # every feedback phase; physically it is a small length or temperature change of that arm.
        self.phase_1 = phase_1  # Loop-mirror arm (mirror 1)
        self.phase_2 = phase_2  # Parasitic arm (mirror 2)

    def rk4_step(self, y, h):
        """Advances the state y by one time step h using 4th-order Runge-Kutta.

        y: current state vector [S, N, phi] at time self.t
        h: time step size
        Returns the state vector one time step h later.
        """
        t = self.t
        k1 = np.array(self.derivative(*y, t))  # Slope at the start of the step
        k2 = np.array(self.derivative(*(y + h*k1/2), t + h/2))  # Slope at the midpoint, using k1 to estimate y there
        k3 = np.array(self.derivative(*(y + h*k2/2), t + h/2))  # Slope at the midpoint again, using the improved k2 estimate
        k4 = np.array(self.derivative(*(y + h*k3), t + h))  # Slope at the end of the step, using k3 to estimate y there

        # Weighted average of the four slopes (midpoint slopes count double)
        return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


    def gain(self, N, threshold):
        """Calculates the gain dG/dN * (N - threshold) in 1/s for carrier number N.

        threshold is the reference carrier number: N_tr for the S and N equations, N_th for the phase equation.
        """
        return differential_gain * (N - threshold)

    def history(self, values, t):
        """Linearly interpolates a stored history (sample i at time i*dt) at time t. Zero before t = 0."""
        if t < 0:
            return 0.0
        x = t / self.dt
        i = int(x)
        if i + 1 >= len(values):
            return values[-1]
        frac = x - i
        return values[i] + frac * (values[i + 1] - values[i])

    def round_trip_field(self, frequency, S, phi, t, coupling, r_ext, tau_ext, phase_offset):
        """x_i in Eq. (S1.6): field returned by one reflector per unit field leaving the facet, one round trip."""
        return coupling**2 * r_ext * np.sqrt(self.history(self.S_history, t - tau_ext) / S) \
            * np.exp(-1j * (frequency * tau_ext + phase_offset + phi - self.history(self.phi_history, t - tau_ext)))

    def external_cavity_arm(self, frequency, S, phi, t, coupling, r_ext, tau_ext, phase_offset):
        """One reflector's term in Eq. (S1.6), including its multiple round trips."""
        feedback = self.round_trip_field(frequency, S, phi, t, coupling, r_ext, tau_ext, phase_offset)
        return feedback / (1 + r_2 * feedback)

    def external_cavity_transfer_function(self, frequency, S, N, phi, t):
        """Calculates the transfer function T of the external cavities at time t, Eq. (S1.6)."""
        arm_1 = self.external_cavity_arm(frequency, S, phi, t, self.coupler_1, r_ext1, self.tau_ext1, self.phase_1)
        arm_2 = self.external_cavity_arm(frequency, S, phi, t, self.coupler_2, r_ext2, self.tau_ext2, self.phase_2)

        # Each mirror's own reflection, plus leakage between the two reflection paths
        return 1 + (1 - r_2**2) / r_2 * (arm_1 + arm_2 + 2 * arm_1 * arm_2)

    def derivative(self, S, N, phi, t):
        """Calculates the derivative of the state vector [S, N, phi] at time t."""

        noise = [0,0,0]

        T = self.external_cavity_transfer_function(self.frequency, S, N, phi, t)

        dS = (self.gain(N, transparency_carrier_density) / (1 + epsilon * S) - 1/tau_ph + 2 * np.log(np.abs(T)) / tau_D) * S + beta * N / tau_n + noise[0]

        dN = eta * self.I / q - N / tau_n - self.gain(N, transparency_carrier_density) / (1 + epsilon * S) * S + noise[1]

        dphi = 0.5 * alpha_H * self.gain(N, threshold_carrier_density) + np.angle(T) / tau_D + noise[2]

        return dS, dN, dphi

    def step(self):
        """Advances the state of the laser by one time step using RK4."""
        self.S, self.N, self.phi = self.rk4_step(np.array([self.S, self.N, self.phi]), self.dt)
        self.t += self.dt
        self.S_history.append(self.S)
        self.N_history.append(self.N)
        self.phi_history.append(self.phi)
        T = self.external_cavity_transfer_function(self.frequency, self.S, self.N, self.phi, self.t)
        self.feedback_gain_history.append(2 * np.log(np.abs(T)) / tau_D)

    def run(self, I, frequency, total_time):
        """Runs the simulation for drive current I (A), optical angular frequency (rad/s) and total time (s)."""
        self.I = I
        self.frequency = frequency
        num_steps = int(total_time / self.dt)
        for _ in range(num_steps):
            self.step()

        return np.array(self.S_history), np.array(self.N_history), np.array(self.phi_history)


    


def far_exit_power_mW(laser, S=None, phi=None):
    """Optical power (mW) leaving the far exit of the PIC, i.e. transmitted past the parasitic reflector (mirror 2).

    P_far(t) = h nu * (1 - r_ext2^2) * n^2 * front_escape_rate * S(t') / |1 + r_2 (x_1(t') + x_2(t'))|^2,
    t' = t - tau_ext2 / 2

      front_escape_rate * S   flux the front facet would emit with no external cavity (photons/s)
      1 / |1 + r_2 R|^2       multiple bounces between facet and reflectors, R = x_1 + x_2 the returned field per unit
                              outgoing field. From E_tx = t_2 E_int - r_2 E_ret and E_ret = R E_tx; the same
                              quasi-static approximation as Eq. (S1.6)
      n^2                     power fraction the coupler sends towards mirror 2
      1 - r_ext2^2            fraction transmitted past mirror 2 (lossless reflector)
      tau_ext2 / 2            one-way travel time from the facet to the far exit; nothing arrives before it (P = 0)

    With no reflectors this reduces to front_escape_rate * S * h nu, the same as photons_to_power_mW(S).
    """
    S = np.array(laser.S_history if S is None else S).real
    phi = np.array(laser.phi_history if phi is None else phi).real
    t = np.arange(len(S)) * laser.dt
    tp = t - laser.tau_ext2 / 2  # Time the light now at the exit left the facet
    arrived = tp >= 0
    tp = np.clip(tp, 0, None)

    # Same history convention as Laser.history: linear interpolation, zero before t = 0
    S_tp = np.interp(tp, t, S)
    phi_tp = np.interp(tp, t, phi)
    R = 0
    for coupling, r_ext, tau_ext, offset in ((laser.coupler_1, r_ext1, laser.tau_ext1, laser.phase_1),
                                             (laser.coupler_2, r_ext2, laser.tau_ext2, laser.phase_2)):
        S_back = np.interp(tp - tau_ext, t, S, left=0.0)
        phi_back = np.interp(tp - tau_ext, t, phi, left=0.0)
        R = R + coupling**2 * r_ext * np.sqrt(S_back / S_tp) \
            * np.exp(-1j * (laser.frequency * tau_ext + offset + phi_tp - phi_back))

    flux = (1 - r_ext2**2) * laser.coupler_2**2 * front_escape_rate * S_tp / np.abs(1 + r_2 * R)**2
    return np.where(arrived, (h * c / wavelength) * flux * 1e3, 0.0)


def plot_run(laser, S, N, phi, I, filename="transient_response.png"):
    """Plots output power, carrier number and frequency deviation against time on stacked panels."""

    surface = "#fcfcfb"
    ink = "#0b0b0b"
    ink_2 = "#52514e"
    ink_3 = "#8a8984"
    grid = "#e6e5e1"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"

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

    t = np.arange(len(S)) * laser.dt * 1e9  # Time in ns
    S, N, phi = S.real, N.real, phi.real
    detuning_GHz = np.gradient(phi, laser.dt) / (2 * np.pi) / 1e9  # Instantaneous frequency deviation from the solitary laser
    P = np.where(np.isfinite(S), photons_to_power_mW(S, np.array(laser.feedback_gain_history)), np.nan)
    P_far = far_exit_power_mW(laser, S, phi)

    N_th = transparency_carrier_density + 1 / (differential_gain * tau_ph)  # Solitary-laser threshold
    I_th = q * N_th / (eta * tau_n)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, facecolor=surface,
                             gridspec_kw={"height_ratios": [1.4, 1, 1], "hspace": 0.28})
    fig.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.07)

    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        # Feedback onsets: light from each external cavity first returns after one round trip
        for tau in (laser.tau_ext1, laser.tau_ext2):
            ax.axvline(tau * 1e9, color=ink_3, linewidth=1, linestyle=(0, (4, 4)), zorder=1)

    def panel_title(ax, text):
        ax.text(0, 1.04, text, transform=ax.transAxes, color=ink, fontsize=13, fontweight="bold", va="bottom")

    # Output power
    ax = axes[0]
    ax.plot(t, P, color=ink_3, linewidth=1.5, solid_capstyle="round", zorder=2, label="net at laser facet")
    ax.plot(t, P_far, color=blue, linewidth=2, solid_capstyle="round", zorder=3, label="far exit (past mirror 2)")
    ax.set_ylim(bottom=min(0, 1.05 * np.nanmin(P)))  # Net power can dip below zero during strong feedback transients
    ax.set_ylabel("Output power (mW)")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    panel_title(ax, "Optical output power")
    last = np.flatnonzero(np.isfinite(P_far))[-1]
    ax.text(1, 1.04, f"far exit: mean {np.nanmean(P_far):.2f} mW  ·  peak {np.nanmax(P_far):.1f} mW  ·  final {P_far[last]:.2f} mW",
            transform=ax.transAxes, color=ink_2, fontsize=11, ha="right", va="bottom")
    tau_last = max(laser.tau_ext1, laser.tau_ext2)
    ax.text(tau_last * 1e9, 0.97,
            f" feedback returns at {laser.tau_ext1 * 1e12:.0f} ps (mirror 1) and {laser.tau_ext2 * 1e12:.0f} ps (PIC reflection)",
            transform=ax.get_xaxis_transform(), color=ink_3, fontsize=10, va="top", ha="left",
            bbox={"facecolor": surface, "edgecolor": "none", "pad": 1})

    # Carrier number
    ax = axes[1]
    ax.plot(t, N / 1e7, color=orange, linewidth=2, solid_capstyle="round", zorder=3)
    ax.set_ylabel("Carriers in cavity (×10$^7$)")
    ax.set_ylim(0, max(np.nanmax(N), N_th) / 1e7 * 1.08)
    panel_title(ax, "Carrier number  N")
    for level, name, va, dy in ((N_th, "solitary threshold", "bottom", 3), (transparency_carrier_density, "transparency", "top", -4)):
        ax.axhline(level / 1e7, color=ink_3, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
        ax.annotate(f"{name}  {level / 1e7:.2f}", (t[-1], level / 1e7), xytext=(0, dy), textcoords="offset points",
                    color=ink_3, fontsize=10, ha="right", va=va)

    # Frequency deviation
    ax = axes[2]
    ax.axhline(0, color=ink_3, linewidth=1, zorder=1)
    ax.plot(t, detuning_GHz, color=aqua, linewidth=2, solid_capstyle="round", zorder=3)
    ax.set_ylabel("Δν (GHz)")
    ax.set_xlabel("Time (ns)")
    panel_title(ax, "Frequency deviation from solitary laser  Δν = (dφ/dt) / 2π")
    ax.set_xlim(t[0], t[-1])

    fig.text(0.09, 0.95, "Turn-on transient with external-cavity feedback", color=ink, fontsize=20, fontweight="bold")
    fig.text(0.09, 0.915,
             f"Step to {I * 1e3:.1f} mA  ·  {I / I_th:.2f} × solitary threshold ({I_th * 1e3:.1f} mA)  ·  "
             f"r_ext1 = {r_ext1}, r_ext2 = {r_ext2}, κ₁ = {laser.coupler_1:.3f}  ·  RK4, dt = {laser.dt * 1e12:g} ps  ·  "
             f"φ₁, φ₂ offsets = {laser.phase_1:.2f}, {laser.phase_2:.2f} rad\n"
             f"net P = {output_coupling}·hν·S·(1/τ_ph − 2 ln|T|/τ_D)  ·  "
             f"far exit P = hν·(1−r_ext2²)·n²·γ_f·S(t−τ₂/2) / |1 + r₂(x₁+x₂)|²",
             color=ink_2, fontsize=12)

    fig.savefig(filename, dpi=150, facecolor=surface)
    return fig


def power_extrema(P):
    """Local maxima and minima of the power trace P. A steady output has none, so its final value is returned."""
    slope = np.sign(np.diff(P))
    turning = np.flatnonzero(slope[1:] != slope[:-1]) + 1
    return P[turning] if len(turning) else P[-1:]


def intensity_fluctuation(P):
    """Normalised intensity fluctuation std(P) / mean(P), a stand-in for the paper's RF-envelope stability measure."""
    return np.std(P) / np.mean(P)


def sweep_coupler(couplers, I, frequency, total_time, settle_time, transient_dir=None, phase_1=0.0, phase_2=0.0):
    """Runs a fresh turn-on for each coupler_1 value and measures the far-exit output after settle_time.

    transient_dir: if given, the transient plot of every run is saved there as coupler_<value>.png.
    Returns one dict per coupler_1 value with the power extrema, mean power and intensity fluctuation.
    """
    if transient_dir is not None:
        os.makedirs(transient_dir, exist_ok=True)
    results = []
    for k, coupler_1 in enumerate(couplers):
        laser = Laser(coupler_1=coupler_1, phase_1=phase_1, phase_2=phase_2)
        S, N, phi = laser.run(I, frequency, total_time)
        if transient_dir is not None:
            fig = plot_run(laser, S, N, phi, I, filename=os.path.join(transient_dir, f"coupler_{coupler_1:.3f}.png"))
            plt.close(fig)  # Keep dozens of figures from piling up in memory
        P = far_exit_power_mW(laser, S, phi)[int(settle_time / laser.dt):]  # Measured where a detector would sit
        result = {"coupler_1": coupler_1, "extrema": power_extrema(P), "mean": P.mean(), "fluctuation": intensity_fluctuation(P)}
        results.append(result)
        print(f"[{k + 1}/{len(couplers)}] coupler_1 = {coupler_1:.3f}  mean {result['mean']:.2f} mW  "
              f"fluctuation {result['fluctuation']:.2e}")
    return results


def plot_sweep(results, I, total_time, settle_time, stable_below=1e-2, filename="coupler_sweep.png"):
    """Plots a bifurcation diagram of output power and the intensity fluctuation against coupler_1."""

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

    couplers = np.array([r["coupler_1"] for r in results])
    fluctuation = np.array([r["fluctuation"] for r in results])
    stable = fluctuation < stable_below

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, facecolor=surface,
                             gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.3})
    fig.subplots_adjust(left=0.09, right=0.96, top=0.82, bottom=0.08)

    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        # Shade the coupler_1 values where the output settled
        for m, is_stable in zip(couplers, stable):
            if is_stable:
                half = np.min(np.diff(couplers)) / 2 if len(couplers) > 1 else 0.01
                ax.axvspan(m - half, m + half, color=blue, alpha=0.06, linewidth=0, zorder=0)

    def panel_title(ax, text, y=1.04):
        ax.text(0, y, text, transform=ax.transAxes, color=ink, fontsize=13, fontweight="bold", va="bottom")

    # Bifurcation diagram: every local max/min of the settled output, one column per coupler_1
    ax = axes[0]
    for r in results:
        ax.plot(np.full(len(r["extrema"]), r["coupler_1"]), r["extrema"], "o", color=blue, markersize=2.5,
                markeredgewidth=0, alpha=0.5, zorder=3)
    ax.set_ylim(bottom=min(0, 1.05 * min(r["extrema"].min() for r in results)))
    ax.set_ylabel("Far-exit power extrema (mW)")
    panel_title(ax, "Bifurcation diagram  ·  local maxima and minima of P after settling", y=1.1)
    ax.text(1, 1.1, "one dot per turning point; a single dot means steady output",
            transform=ax.transAxes, color=ink_2, fontsize=11, ha="right", va="bottom")

    # Self-injection level on a second x scale (same quantity, relabelled): 20 log10(m^2 r_ext1)
    to_dB = lambda m: 20 * np.log10(np.clip(m, 1e-6, None) ** 2 * r_ext1)
    from_dB = lambda dB: np.sqrt(10 ** (dB / 20) / r_ext1)
    top = ax.secondary_xaxis("top", functions=(to_dB, from_dB))
    top.set_xticks([-30, -20, -15, -10, -6, -3])
    top.xaxis.set_major_formatter(lambda v, pos: f"{v:g} dB")
    top.tick_params(colors=ink_2, length=0)
    top.spines["top"].set_visible(False)

    # Intensity fluctuation
    ax = axes[1]
    ax.plot(couplers, np.clip(fluctuation, 1e-12, None), color=orange, linewidth=2, marker="o", markersize=8,
            markeredgecolor=surface, markeredgewidth=2, zorder=3)
    ax.set_yscale("log")
    ax.axhline(stable_below, color=ink_3, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    ax.annotate(f"stable below {stable_below:g}", (couplers[0], stable_below), xytext=(0, -4), textcoords="offset points",
                color=ink_3, fontsize=10, ha="left", va="top")
    ax.set_ylabel(r"$\sigma_P\,/\,\langle P\rangle$")
    ax.set_xlabel("coupler_1  (field split m into the self-injection path)")
    panel_title(ax, "Intensity fluctuation")
    pad = (couplers[-1] - couplers[0]) * 0.03 if len(couplers) > 1 else 0.05
    ax.set_xlim(couplers[0] - pad, couplers[-1] + pad)

    fig.text(0.09, 0.95, "Self-injection strength sweep", color=ink, fontsize=20, fontweight="bold")
    fig.text(0.09, 0.915,
             f"{I * 1e3:.1f} mA  ·  r_ext1 = {r_ext1}, r_ext2 = {r_ext2}  ·  {len(results)} runs of {total_time * 1e9:g} ns, "
             f"measured at the far exit after {settle_time * 1e9:g} ns  ·  {stable.sum()} of {len(results)} stable  ·  "
             f"top axis: self-injection level 20 log10(m² r_ext1)",
             color=ink_2, fontsize=12)

    fig.savefig(filename, dpi=150, facecolor=surface)
    return fig


if __name__ == "__main__":
    I = 50e-3  # Drive current in A
    frequency = 0.0  # Offset added to laser_frequency in rad/s (shifts the feedback phase)
    laser_frequency = 1.22e15  # Optical angular frequency 2*pi*c/1550 nm in rad/s
    phase_1 = 0.0  # Extra round-trip phase on the loop-mirror arm in rad (0..2pi covers every feedback phase)
    phase_2 = 0.0  # Extra round-trip phase on the parasitic arm in rad
    total_time = 1e-8  # Simulation length in s

    sweep = True  # True: sweep coupler_1 over sweep_couplers instead of a single run
    sweep_couplers = np.linspace(0, 0.9, 19)  # coupler_1 values to sweep
    settle_time = 5e-9  # Sweep only: time discarded before measuring the output, in s
    transient_dir = "coupler_sweep_transients"  # Sweep only: folder for each run's transient plot (None to skip)

    if sweep:
        results = sweep_coupler(sweep_couplers, I, frequency + laser_frequency, total_time, settle_time, transient_dir,
                                phase_1, phase_2)
        plot_sweep(results, I, total_time, settle_time)
    else:
        laser = Laser(phase_1=phase_1, phase_2=phase_2)
        S, N, phi = laser.run(I, frequency + laser_frequency, total_time)
        plot_run(laser, S, N, phi, I)
    plt.show()
