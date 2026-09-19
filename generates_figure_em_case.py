import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import spherical_jn
from scipy.optimize import brentq
import pandas as pd

matplotlib.use("Agg")

"""
Generates figures for the companion article:
"Blackbody Radiation in Spherical Cavities: Thermodynamics of an Electromagnetic Field"
(Article II - TE/TM modes)

Mirrors the structure of generates_figures.py (scalar case, Article I), extended to
two independent mode families:
  TE: j_l(x) = 0                    (l >= 1)   [identical to scalar spectrum, but l=0 excluded]
  TM: j_l(x) + x*j_l'(x) = 0        (l >= 1)   [d/dx[x j_l(x)] = 0]

Generated figures (natural units, hbar = c = kB = 1):
  fig1_em_sparse_sector.png     -> TE+TM stem plot, individual modes (x_cut=10, T=35)
  fig2_em_coarse_grained.png    -> TE+TM coarse-grained density vs EM Weyl (x_cut=60)
  fig3_em_panel_deltaomega.png  -> Panel with 3 bin widths (x_cut=300)
  fig4_em_convergence.png       -> Convergence of U_numerical -> U_analytical (EM)

Requirements: numpy scipy matplotlib pandas
"""

# ----------------------------------------------------------------------
# 1. Numerical Core: TE and TM root-finding, EM Weyl density
# ----------------------------------------------------------------------

def te_condition(l, x):
    """TE boundary condition: j_l(x) = 0."""
    return spherical_jn(l, x)

def tm_condition(l, x):
    """TM boundary condition: d/dx[x j_l(x)] = 0  <=>  j_l(x) + x j_l'(x) = 0."""
    return spherical_jn(l, x) + x * spherical_jn(l, x, derivative=True)

def _first_zero(func, l, x0, dx=0.5):
    """Generic bracket-and-refine search for the first positive zero of func(l, x)."""
    x = x0
    while True:
        if func(l, x) * func(l, x + dx) < 0:
            return brentq(lambda z: func(l, z), x, x + dx)
        x += dx / 5

def _zeros_up_to_cutoff(func, l, x_cut, x0, dx=0.5):
    """All positive zeros of func(l, x) with x <= x_cut (true spectral cutoff)."""
    zeros = []
    x = x0
    while x < x_cut + dx:
        if func(l, x) * func(l, x + dx) < 0:
            zero = brentq(lambda z: func(l, z), x, x + dx)
            if zero <= x_cut:
                zeros.append(zero)
            x += dx
        else:
            x += dx / 5
    return zeros

def get_first_zero_te(l):
    return _first_zero(te_condition, l, x0=l + 0.1)

def get_first_zero_tm(l):
    return _first_zero(tm_condition, l, x0=max(l - 3, 0.1))

def get_zeros_up_to_cutoff_te(l, x_cut):
    return _zeros_up_to_cutoff(te_condition, l, x_cut, x0=l + 0.1)

def get_zeros_up_to_cutoff_tm(l, x_cut):
    return _zeros_up_to_cutoff(tm_condition, l, x_cut, x0=max(l - 3, 0.1))

def generate_spectral_cutoff_table_em(x_cut: float, R0: float = 1.0) -> pd.DataFrame:
    """
    Generates a table (mode_type, l, n, x, omega) including ALL TE and TM modes
    with x_nl <= x_cut, for l >= 1 (monopolar l=0 sector excluded for EM).

    For each family, the l-loop breaks once the first (n=1) zero exceeds x_cut,
    since x_{1,l} grows monotonically with l.
    """
    rows = []
    for mode_type, first_zero_fn, zeros_fn in [
        ("TE", get_first_zero_te, get_zeros_up_to_cutoff_te),
        ("TM", get_first_zero_tm, get_zeros_up_to_cutoff_tm),
    ]:
        l = 1
        while True:
            x1 = first_zero_fn(l)
            if x1 > x_cut:
                break
            zeros_l = zeros_fn(l, x_cut)
            for n, zero in enumerate(zeros_l, start=1):
                rows.append([mode_type, l, n, zero])
            l += 1

    df = pd.DataFrame(rows, columns=["mode_type", "l", "n", "x"])
    df["omega"] = df["x"] / R0
    return df

def get_average_energy(omega: float, T: float) -> float:
    """<E> = hbar*omega / (e^(hbar*omega/kB T) - 1) in natural units."""
    return omega / (np.exp(omega / T) - 1)

def get_weyl_density_em(omega: float, R0: float = 1.0, c: float = 1.0) -> float:
    """Leading EM Weyl density: twice the scalar result -> 4 R0^3 w^2/(3 pi c^3)."""
    return (4 * R0 ** 3 / (3 * np.pi * c ** 3)) * omega ** 2

def get_planck_density_em(omega: float, T: float, R0: float = 1.0, c: float = 1.0) -> float:
    return get_weyl_density_em(omega, R0, c) * get_average_energy(omega, T)

def get_analytical_energy_em(T: float, R0: float = 1.0) -> float:
    """Leading EM analytical total energy: twice the scalar result -> (4 pi^3/45) R0^3 T^4."""
    return (4 * np.pi ** 3 / 45) * R0 ** 3 * T ** 4

# ----------------------------------------------------------------------
# 2. Figure: Sparse sector (stem plot), TE and TM distinguished
# ----------------------------------------------------------------------

def generate_fig_em_stem(df: pd.DataFrame, T: float, out_file: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"TE": "steelblue", "TM": "darkorange"}

    # Combined (TE+TM) series, drawn first/behind as a thick light-gray reference
    comb = df.sort_values("omega")
    comb_w = (2 * comb["l"] + 1) * comb["omega"].apply(lambda w: get_average_energy(w, T))
    _, comb_stemlines, _ = ax.stem(comb["omega"], comb_w, basefmt=" ", markerfmt="none")
    plt.setp(comb_stemlines, linewidth=4, color="lightgray", alpha=0.7)
    ax.plot([], [], color="lightgray", linewidth=4, label="TE+TM (combined)")

    for mode_type, sub in df.groupby("mode_type"):
        weights = (2 * sub["l"] + 1) * sub["omega"].apply(lambda w: get_average_energy(w, T))
        markerline, stemlines, baseline = ax.stem(
            sub["omega"], weights, basefmt=" ", label=mode_type,
            linefmt=colors[mode_type], markerfmt="o"
        )
        plt.setp(stemlines, linewidth=1.5, color=colors[mode_type])
        plt.setp(markerline, markersize=6, color=colors[mode_type])

    ax.set_xlabel(r"$\omega_{n,\ell}^{(\alpha)}$", fontsize=14)
    ax.set_ylabel(r"$E^{\mathrm{mult}}_{n,\ell}$", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}  ({len(df)} modes)")

# ----------------------------------------------------------------------
# 3. Figure: Coarse-grained EM density (TE+TM combined, with TE-only and
#    TM-only overlays showing how the two families combine), single panel
# ----------------------------------------------------------------------

def generate_fig_em_coarse_grained(df: pd.DataFrame, T: float, R0: float, delta_omega: float, out_file: str):
    omega_max = df["omega"].max()
    bins = np.arange(0, omega_max + delta_omega, delta_omega)
    centers = (bins[:-1] + bins[1:]) / 2
    colors = {"TE": "steelblue", "TM": "darkorange"}

    weights = (2 * df["l"] + 1) * df["omega"].apply(lambda w: get_average_energy(w, T))
    bin_idx = np.digitize(df["omega"], bins)
    u_binned = np.array([weights[bin_idx == i].sum() for i in range(1, len(bins))]) / delta_omega

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(centers, u_binned, width=delta_omega * 0.9, color="steelblue",
           alpha=0.35, label="coarse-grained (TE+TM)")

    for mode in ["TE", "TM"]:
        sub = df[df["mode_type"] == mode]
        w = (2 * sub["l"] + 1) * sub["omega"].apply(lambda o: get_average_energy(o, T))
        idx = np.digitize(sub["omega"], bins)
        u = np.array([w[idx == i].sum() for i in range(1, len(bins))]) / delta_omega
        ax.plot(centers, u, "--", color=colors[mode], linewidth=1.6, label=f"{mode} only", alpha=0.9)

    omega_cont = np.linspace(0.01, omega_max, 500)
    ax.plot(omega_cont, get_planck_density_em(omega_cont, T, R0), color="darkred", linewidth=2,
            label="leading EM Weyl")

    ax.set_xlabel(r"$\omega$", fontsize=14)
    ax.set_ylabel(r"$u_{\mathrm{EM}}(\omega)$", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}  ({len(df)} modes, bin={delta_omega})")

# ----------------------------------------------------------------------
# 4. Figure: Panel with 3 bin widths (dense sector), with TE-only and
#    TM-only overlays in each subplot
# ----------------------------------------------------------------------

def generate_fig_em_panel_deltaomega(T: float, R0: float, x_cut: float, bin_widths: tuple = (2, 5, 10), out_file: str = "fig_em_panel.png", df: pd.DataFrame = None):
    assert len(bin_widths) == 3
    dw1, dw2, dw3 = bin_widths
    colors = {"TE": "steelblue", "TM": "darkorange"}

    if df is None:
        df = generate_spectral_cutoff_table_em(x_cut, R0)
    else:
        df = df[df["x"] <= x_cut]
    weights = (2 * df["l"] + 1) * df["omega"].apply(lambda w: get_average_energy(w, T))
    omega_max = df["omega"].max()
    te_sub = df[df["mode_type"] == "TE"]
    tm_sub = df[df["mode_type"] == "TM"]
    w_te = (2 * te_sub["l"] + 1) * te_sub["omega"].apply(lambda o: get_average_energy(o, T))
    w_tm = (2 * tm_sub["l"] + 1) * tm_sub["omega"].apply(lambda o: get_average_energy(o, T))

    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.15)
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1], sharey=ax_top_left)
    ax_bottom = fig.add_subplot(gs[1, :], sharey=ax_top_left)

    axes_and_bins = [(ax_top_left, dw1), (ax_top_right, dw2), (ax_bottom, dw3)]

    data = {}
    max_values = []
    for _, dw in axes_and_bins:
        bins = np.arange(0, omega_max + dw, dw)
        centers = (bins[:-1] + bins[1:]) / 2
        bin_idx = np.digitize(df["omega"], bins)
        u_binned = np.array([weights[bin_idx == i].sum() for i in range(1, len(bins))]) / dw
        idx_te = np.digitize(te_sub["omega"], bins)
        idx_tm = np.digitize(tm_sub["omega"], bins)
        u_te = np.array([w_te[idx_te == i].sum() for i in range(1, len(bins))]) / dw
        u_tm = np.array([w_tm[idx_tm == i].sum() for i in range(1, len(bins))]) / dw
        data[dw] = (centers, u_binned, u_te, u_tm)
        max_values.append(u_binned.max())

    omega_cont = np.linspace(0.01, omega_max, 500)
    analytical_curve = get_planck_density_em(omega_cont, T, R0)
    y_max = 1.1 * max(max(max_values), analytical_curve.max())

    for ax, dw in axes_and_bins:
        centers, u_binned, u_te, u_tm = data[dw]
        ax.bar(centers, u_binned, width=dw * 0.9, color="steelblue", alpha=0.3)
        ax.plot(centers, u_te, "--", color=colors["TE"], linewidth=1.1, alpha=0.85)
        ax.plot(centers, u_tm, "--", color=colors["TM"], linewidth=1.1, alpha=0.85)
        ax.plot(omega_cont, analytical_curve, color="darkred", linewidth=2)
        ax.set_xlabel(r"$\omega$", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, y_max)
        ax.text(0.97, 0.95, rf"$\Delta\omega = {dw}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=13,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

    ax_top_left.set_ylabel(r"$u_{\mathrm{EM}}(\omega)$", fontsize=13)
    ax_bottom.set_ylabel(r"$u_{\mathrm{EM}}(\omega)$", fontsize=13)
    plt.setp(ax_top_right.get_yticklabels(), visible=False)

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Generated] {out_file}  ({len(df)} modes, x_cut={x_cut}, bins={bin_widths})")

# ----------------------------------------------------------------------
# 5. Figure: Convergence with true spectral cutoff (EM)
# ----------------------------------------------------------------------

def generate_fig_em_convergence(T: float, R0: float, x_cutoffs: list, out_file: str, df_full: pd.DataFrame = None):
    """
    If df_full (built at max(x_cutoffs)) is provided, each x_cut is obtained by
    filtering df_full instead of rebuilding the root search from scratch
    (root-finding cost scales like x_cut^2, so this avoids redundant O(n^2) work).
    """
    results = []
    U_analytical = get_analytical_energy_em(T, R0)

    if df_full is None:
        df_full = generate_spectral_cutoff_table_em(max(x_cutoffs), R0)
    df_full = df_full.copy()
    df_full["weight"] = (2 * df_full["l"] + 1) * df_full["omega"].apply(lambda w: get_average_energy(w, T))

    for x_cut in x_cutoffs:
        sub = df_full[df_full["x"] <= x_cut]
        U_num = sub["weight"].sum()
        error = abs(U_num - U_analytical) / U_analytical
        results.append((x_cut, len(sub), U_num, error))
        print(f"  x_cut={x_cut:6.1f}  modes={len(sub):6d}  U_num={U_num:12.1f}  error={error:.4f}")

    df_conv = pd.DataFrame(results, columns=["x_cut", "n_modes", "U_numerical", "relative_error"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_conv["x_cut"], df_conv["relative_error"], "o-", color="darkred", markersize=7)
    ax.set_yscale("log")
    ax.set_xlabel(r"Spectral cutoff $x_{\mathrm{cut}}=\omega_{\max}R_0/c$", fontsize=14)
    ax.set_ylabel(r"Relative error (EM)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}")
    return df_conv

def generate_fig_em_surface_cancellation(T: float, x_cutoffs: list, out_file: str, df_full: pd.DataFrame):
    """
    Two-panel figure isolating the subleading Weyl surface term: tracks the relative
    deviation of U_numerical from the leading-order Weyl prediction, separately for
    TE, TM, and their combination, as x_cut is extended. TE and TM saturate at
    opposite nonzero residuals (Dirichlet-type vs Neumann-type surface coefficient),
    which nearly cancel in the combined electromagnetic spectrum
    (Section 6.1 of the companion article).
    """
    U_an_full = get_analytical_energy_em(T, 1.0)
    U_an_half = U_an_full / 2

    df_full = df_full.copy()
    df_full["weight"] = (2 * df_full["l"] + 1) * df_full["omega"].apply(lambda o: get_average_energy(o, T))

    def rel_err(sub, U_an):
        return (sub["weight"].sum() - U_an) / U_an * 100

    te_errs, tm_errs, comb_errs = [], [], []
    for xc in x_cutoffs:
        te_errs.append(rel_err(df_full[(df_full["mode_type"] == "TE") & (df_full["x"] <= xc)], U_an_half))
        tm_errs.append(rel_err(df_full[(df_full["mode_type"] == "TM") & (df_full["x"] <= xc)], U_an_half))
        comb_errs.append(rel_err(df_full[df_full["x"] <= xc], U_an_full))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax in (ax1, ax2):
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.plot(x_cutoffs, te_errs, "o-", color="steelblue", label="TE only", markersize=6)
        ax.plot(x_cutoffs, tm_errs, "s-", color="darkorange", label="TM only", markersize=6)
        ax.plot(x_cutoffs, comb_errs, "^-", color="darkred", label="Combined TE+TM", linewidth=2.2, markersize=7)
        ax.set_xlabel(r"Spectral cutoff $x_{\mathrm{cut}}$", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=11)

    ax1.set_ylabel("Relative deviation (%)", fontsize=13)
    ax1.set_title("Full range", fontsize=12)
    ax1.legend(fontsize=10, loc="lower right")

    ax2.set_xlim(min(x_cutoffs[-8:]), max(x_cutoffs) + 20)
    ax2.set_ylim(-6, 4)
    ax2.set_title(r"Zoom: $x_{\mathrm{cut}}\geq300$ (saturation regime)", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}")
    return pd.DataFrame({"x_cut": x_cutoffs, "TE_pct": te_errs, "TM_pct": tm_errs, "Combined_pct": comb_errs})

# ----------------------------------------------------------------------
# 6. Main Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR = "Figures_EM"
    os.makedirs(OUT_DIR, exist_ok=True)

    T = 35.0
    R0 = 1.0

    print("=== Sanity check: fundamental modes ===")
    print(f"  TE l=1 first zero: {get_first_zero_te(1):.4f}  (expected ~4.4934)")
    print(f"  TM l=1 first zero: {get_first_zero_tm(1):.4f}  (expected ~2.7437)")

    print("=== Figure (Sparse sector, few modes) ===")
    df_low = generate_spectral_cutoff_table_em(x_cut=10, R0=R0)
    generate_fig_em_stem(df_low, T, out_file=f"{OUT_DIR}/fig_em_sparse_sector.png")

    print("=== Figure (Intermediate sector, coarse-grained) ===")
    df_mid = generate_spectral_cutoff_table_em(x_cut=60, R0=R0)
    generate_fig_em_coarse_grained(df_mid, T, R0, delta_omega=3.0, out_file=f"{OUT_DIR}/fig_em_intermediate_sector.png")

    print("=== Building full table once at max cutoff (reused below) ===")
    x_cutoffs_list = [10, 20, 40, 60, 100, 150, 200, 300, 400, 450, 500, 550]
    cache_path = "df_max550.pkl"
    if os.path.exists(cache_path):
        df_max = pd.read_pickle(cache_path)
    else:
        df_max = generate_spectral_cutoff_table_em(max(x_cutoffs_list), R0)
        df_max.to_pickle(cache_path)
    print(f"  full table: {len(df_max)} modes up to x_cut={max(x_cutoffs_list)}")

    print("=== Figure (Dense sector: Panel with Delta_omega = 2, 5, 10) ===")
    generate_fig_em_panel_deltaomega(T, R0, x_cut=300, bin_widths=(2, 5, 10),
                                     out_file=f"{OUT_DIR}/fig_em_panel_deltaomega.png", df=df_max)

    print("=== Figure (Convergence with proper spectral cutoff) ===")
    df_conv = generate_fig_em_convergence(T, R0, x_cutoffs_list, f"{OUT_DIR}/fig_em_convergence.png", df_full=df_max)

    print("=== Figure (Surface-term cancellation: TE vs TM vs combined) ===")
    x_cutoffs_cancel = [30, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550]
    df_cancel = generate_fig_em_surface_cancellation(T, x_cutoffs_cancel, f"{OUT_DIR}/fig_surface_cancellation.png", df_full=df_max)
    print(df_cancel.to_string(index=False))

    print("\nConvergence Table (EM):")
    print(df_conv.to_string(index=False))

    print(f"\nU_analytical_EM(T=35, R0=1) = {get_analytical_energy_em(T, R0):.1f}")
    print(f"\nAll figures successfully saved in: {os.path.abspath(OUT_DIR)}")
