"""
Expansion chamber muffler — mode-matching model.
Rectangular duct sections, rigid walls, anechoic outlet.

Junction matching convention:
  Pressure continuity -> projected onto DUCT modes over aperture
  Velocity continuity -> projected onto CHAMBER modes over full chamber
"""
import numpy as np
from scipy.integrate import quad
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

PLOT_FREQS = [100.0, 570.0, 1143.0, 1800.0]


def compute_modes(W, H, k, N_modes):
    """
    Return kz[M,N] (complex) and norm[M,N] for a rigid rectangular duct.
    Evanescent modes have positive imaginary kz so e^{-jkz·z} decays.
    """
    M = N = N_modes
    m_idx = np.arange(M)
    n_idx = np.arange(N)
    mm, nn = np.meshgrid(m_idx, n_idx, indexing="ij")

    kx = mm * np.pi / W
    ky = nn * np.pi / H
    k2_diff = k**2 - kx**2 - ky**2

    kz = np.where(
        k2_diff >= 0,
        np.sqrt(k2_diff.astype(complex)),
        1j * np.sqrt(-k2_diff.astype(complex)),
    )

    # Norm = (W·H) / ((1+δ_m0)·(1+δ_n0))
    denom = np.where(mm == 0, 1.0, 2.0) * np.where(nn == 0, 1.0, 2.0)
    norm = (W * H) / denom

    return kz, norm


def compute_overlap(W_s, H_s, W_c, H_c, N_modes_s, N_modes_c):
    """
    C[p,q,m,n] = ∫∫_aperture ψ^c_pq(x,y) · ψ^s_mn(x-x1, y-y1) dx dy
    Duct is centered in chamber. Computed via scipy.integrate.quad (once).
    Shape: (N_modes_c, N_modes_c, N_modes_s, N_modes_s).
    """
    x1 = (W_c - W_s) / 2.0
    x2 = x1 + W_s
    y1 = (H_c - H_s) / 2.0
    y2 = y1 + H_s

    P = Q = N_modes_c
    M = N = N_modes_s
    C = np.zeros((P, Q, M, N), dtype=float)

    for p in range(P):
        for m in range(M):
            def Ix_integrand(x, p=p, m=m):
                return (np.cos(p * np.pi * x / W_c) *
                        np.cos(m * np.pi * (x - x1) / W_s))
            Ix, _ = quad(Ix_integrand, x1, x2)

            for q in range(Q):
                for n in range(N):
                    def Iy_integrand(y, q=q, n=n):
                        return (np.cos(q * np.pi * y / H_c) *
                                np.cos(n * np.pi * (y - y1) / H_s))
                    Iy, _ = quad(Iy_integrand, y1, y2)
                    C[p, q, m, n] = Ix * Iy

    return C


def solve_for_freq(f, C_overlap, N_modes_d, N_modes_c):
    """
    Solve modal system at single frequency f.

    Projection rules (standard for area-change junctions):
      Pressure continuity → project onto DUCT modes over aperture → MN eqs
      Velocity continuity → project onto CHAMBER modes over full chamber → PQ eqs

    Unknown vector:
      [0:MN)       B1_mn  — section 1 reflected
      [MN:MN+PQ)   A2_pq  — section 2 forward
      [MN+PQ:MN+2PQ) B2_pq — section 2 backward
      [MN+2PQ:)    A3_mn  — section 3 transmitted

    Returns: B1[M,N], A2[P,Q], B2[P,Q], A3[M,N], kz1, kz2, kz3, norm_s, norm_c
    """
    omega = 2 * np.pi * f
    k = omega / cfg.C

    M = N = N_modes_d
    P = Q = N_modes_c
    MN = M * N
    PQ = P * Q

    kz1, norm_s = compute_modes(cfg.INLET_W, cfg.INLET_H, k, N_modes_d)
    kz2, norm_c = compute_modes(cfg.CHAMBER_W, cfg.CHAMBER_H, k, N_modes_c)
    kz3 = kz1   # outlet duct same geometry as inlet

    L1 = cfg.INLET_L
    L2 = cfg.CHAMBER_L

    kz1f = kz1.ravel()    # [MN]
    kz2f = kz2.ravel()    # [PQ]
    kz3f = kz3.ravel()    # [MN]
    ns_f = norm_s.ravel() # [MN]
    nc_f = norm_c.ravel() # [PQ]

    # Cflat[pq, mn] = C[p,q,m,n]
    Cflat = C_overlap.reshape(PQ, MN)

    n_unk = 2 * MN + 2 * PQ
    A_mat = np.zeros((n_unk, n_unk), dtype=complex)
    b_vec = np.zeros(n_unk, dtype=complex)

    A1 = np.zeros(MN, dtype=complex)
    A1[0] = 1.0  # unit incident plane wave

    phase1_A = np.exp(-1j * kz1f * L1)   # forward phase at junc 1
    phase1_B = np.exp(+1j * kz1f * L1)   # backward phase at junc 1
    phase2_A = np.exp(-1j * kz2f * L2)   # forward phase at junc 2
    phase2_B = np.exp(+1j * kz2f * L2)   # backward phase at junc 2

    # ── Junction 1: PRESSURE → project onto duct modes (MN rows) ──────────
    # (A1_mn·phase1A + B1_mn·phase1B) · norm_s_mn = Σ_pq (A2_pq+B2_pq)·C[pq,mn]
    # → norm_s_mn·phase1B_mn·B1_mn - Σ_pq C[pq,mn]·A2_pq - Σ_pq C[pq,mn]·B2_pq
    #   = -norm_s_mn·phase1A_mn·A1_mn
    row0 = 0
    for mn in range(MN):
        row = row0 + mn
        A_mat[row, mn] = ns_f[mn] * phase1_B[mn]         # B1[mn]
        A_mat[row, MN : MN + PQ] = -Cflat[:, mn]          # A2[pq]
        A_mat[row, MN + PQ : MN + 2 * PQ] = -Cflat[:, mn] # B2[pq]
        b_vec[row] = -ns_f[mn] * phase1_A[mn] * A1[mn]

    # ── Junction 1: VELOCITY → project onto chamber modes (PQ rows) ────────
    # kz2_pq·norm_c_pq·(A2_pq - B2_pq)
    #   = Σ_mn kz1_mn·C[pq,mn]·(A1_mn·phase1A - B1_mn·phase1B)
    # → -Σ_mn kz1_mn·C[pq,mn]·phase1B_mn·B1_mn
    #   - kz2_pq·norm_c_pq·A2_pq + kz2_pq·norm_c_pq·B2_pq
    #   = -Σ_mn kz1_mn·C[pq,mn]·phase1A_mn·A1_mn
    row0 = MN
    for pq in range(PQ):
        row = row0 + pq
        A_mat[row, :MN] = -kz1f * Cflat[pq, :] * phase1_B   # B1[mn]
        A_mat[row, MN + pq] = -kz2f[pq] * nc_f[pq]           # A2[pq]
        A_mat[row, MN + PQ + pq] = +kz2f[pq] * nc_f[pq]      # B2[pq]
        b_vec[row] = -np.dot(kz1f * Cflat[pq, :], phase1_A * A1)

    # ── Junction 2: PRESSURE → project onto duct modes (MN rows) ──────────
    # A3_mn·norm_s_mn = Σ_pq (A2_pq·phase2A + B2_pq·phase2B)·C[pq,mn]
    # → Σ_pq C[pq,mn]·phase2A_pq·A2_pq + Σ_pq C[pq,mn]·phase2B_pq·B2_pq
    #   - norm_s_mn·A3_mn = 0
    row0 = MN + PQ
    for mn in range(MN):
        row = row0 + mn
        A_mat[row, MN : MN + PQ] = Cflat[:, mn] * phase2_A    # A2[pq]
        A_mat[row, MN + PQ : MN + 2 * PQ] = Cflat[:, mn] * phase2_B  # B2[pq]
        A_mat[row, MN + 2 * PQ + mn] = -ns_f[mn]              # A3[mn]
        # RHS = 0

    # ── Junction 2: VELOCITY → project onto chamber modes (PQ rows) ────────
    # kz2_pq·norm_c_pq·(A2_pq·phase2A - B2_pq·phase2B)
    #   = Σ_mn kz3_mn·C[pq,mn]·A3_mn
    # → kz2_pq·norm_c_pq·phase2A_pq·A2_pq - kz2_pq·norm_c_pq·phase2B_pq·B2_pq
    #   - Σ_mn kz3_mn·C[pq,mn]·A3_mn = 0
    row0 = MN + PQ + MN
    for pq in range(PQ):
        row = row0 + pq
        A_mat[row, MN + pq] = kz2f[pq] * nc_f[pq] * phase2_A[pq]    # A2[pq]
        A_mat[row, MN + PQ + pq] = -kz2f[pq] * nc_f[pq] * phase2_B[pq]  # B2[pq]
        A_mat[row, MN + 2 * PQ :] = -kz3f * Cflat[pq, :]             # A3[mn]
        # RHS = 0

    x = np.linalg.solve(A_mat, b_vec)

    B1 = x[:MN].reshape(M, N)
    A2 = x[MN : MN + PQ].reshape(P, Q)
    B2 = x[MN + PQ : MN + 2 * PQ].reshape(P, Q)
    A3 = x[MN + 2 * PQ :].reshape(M, N)

    return B1, A2, B2, A3, kz1, kz2, kz3, norm_s, norm_c


def _power_flux(A, kz, norm, omega):
    """Acoustic power flux for a set of modal amplitudes (propagating only)."""
    W = 0.0
    for idx in np.ndindex(A.shape):
        kz_val = kz[idx]
        if np.imag(kz_val) < 1e-10 and np.real(kz_val) > 1e-10:
            W += (np.real(kz_val) / (2.0 * omega * cfg.RHO)) \
                 * np.abs(A[idx])**2 * norm[idx]
    return W


def compute_tl(freqs, C_overlap, N_modes_d=8, N_modes_c=8):
    """
    Loop over frequencies, return (TL_mm array, all_coeffs list).
    """
    TL = np.zeros(len(freqs))
    all_coeffs = []
    for i, f in enumerate(freqs):
        omega = 2 * np.pi * f
        B1, A2, B2, A3, kz1, kz2, kz3, norm_s, norm_c = solve_for_freq(
            f, C_overlap, N_modes_d, N_modes_c)
        S_in = cfg.INLET_W * cfg.INLET_H
        W_inc = S_in / (2.0 * cfg.RHO * cfg.C)
        W_trans = _power_flux(A3, kz3, norm_s, omega)
        TL[i] = 10.0 * np.log10(W_inc / W_trans) if W_trans > 0 else np.inf
        all_coeffs.append((B1, A2, B2, A3, kz1, kz2, kz3, norm_s, norm_c))
    return TL, all_coeffs


def _pw_formula(freqs):
    """Plane-wave TL reference for expansion chamber."""
    S_in = cfg.INLET_W * cfg.INLET_H
    S_ch = cfg.CHAMBER_W * cfg.CHAMBER_H
    m = S_ch / S_in
    k_arr = 2 * np.pi * np.asarray(freqs) / cfg.C
    return 10.0 * np.log10(
        1.0 + 0.25 * (m - 1.0 / m)**2 * np.sin(k_arr * cfg.CHAMBER_L)**2)


def compute_onaxis_pressure(coeffs_tuple, freq, N_modes_d, N_modes_c, nz=400):
    """
    Evaluate p(x=W/2, y=H/2, z) along the full duct axis.
    Returns z_mm (array), p_complex (array).

    At the centerline x=W/2, y=H/2 of each section:
      cos(m·π·0.5) = cos(m·π/2) — only even-indexed modes contribute.
    Evanescent modes are anchored at their source junction and decay away.
    """
    B1, A2, B2, A3, kz1, kz2, kz3, norm_s, norm_c = coeffs_tuple

    L1, L2, L3 = cfg.INLET_L, cfg.CHAMBER_L, cfg.OUTLET_L
    WI, HI = cfg.INLET_W, cfg.INLET_H
    WC, HC = cfg.CHAMBER_W, cfg.CHAMBER_H

    z_grid = np.linspace(0, L1 + L2 + L3, nz)
    p = np.zeros(nz, dtype=complex)

    for iz, z_val in enumerate(z_grid):

        if z_val <= L1:
            z_loc = z_val
            for mi in range(N_modes_d):
                cos_x = float(np.cos(mi * np.pi / 2))
                if cos_x == 0.0:
                    continue
                for ni in range(N_modes_d):
                    cos_y = float(np.cos(ni * np.pi / 2))
                    if cos_y == 0.0:
                        continue
                    kz_val = kz1[mi, ni]
                    alpha = float(np.imag(kz_val))
                    A1_mn = 1.0 if (mi == 0 and ni == 0) else 0.0
                    if alpha < 1e-10:
                        amp = (A1_mn * np.exp(-1j * kz_val * z_loc) +
                               B1[mi, ni] * np.exp(+1j * kz_val * z_loc))
                    else:
                        B1_junc = B1[mi, ni] * np.exp(+1j * kz_val * L1)
                        amp = B1_junc * np.exp(-alpha * (L1 - z_loc))
                    p[iz] += amp * cos_x * cos_y

        elif z_val <= L1 + L2:
            z_loc = z_val - L1
            for pi in range(N_modes_c):
                cos_x = float(np.cos(pi * np.pi / 2))
                if cos_x == 0.0:
                    continue
                for qi in range(N_modes_c):
                    cos_y = float(np.cos(qi * np.pi / 2))
                    if cos_y == 0.0:
                        continue
                    kz_val = kz2[pi, qi]
                    alpha = float(np.imag(kz_val))
                    if alpha < 1e-10:
                        amp = (A2[pi, qi] * np.exp(-1j * kz_val * z_loc) +
                               B2[pi, qi] * np.exp(+1j * kz_val * z_loc))
                    else:
                        amp_A = A2[pi, qi] * np.exp(-alpha * z_loc)
                        B2_junc = B2[pi, qi] * np.exp(+1j * kz_val * L2)
                        amp_B = B2_junc * np.exp(-alpha * (L2 - z_loc))
                        amp = amp_A + amp_B
                    p[iz] += amp * cos_x * cos_y

        else:
            z_loc = z_val - L1 - L2
            for mi in range(N_modes_d):
                cos_x = float(np.cos(mi * np.pi / 2))
                if cos_x == 0.0:
                    continue
                for ni in range(N_modes_d):
                    cos_y = float(np.cos(ni * np.pi / 2))
                    if cos_y == 0.0:
                        continue
                    kz_val = kz3[mi, ni]
                    alpha = float(np.imag(kz_val))
                    if alpha < 1e-10:
                        amp = A3[mi, ni] * np.exp(-1j * kz_val * z_loc)
                    else:
                        amp = A3[mi, ni] * np.exp(-alpha * z_loc)
                    p[iz] += amp * cos_x * cos_y

    return z_grid * 1e3, p


if __name__ == "__main__":
    N_D = 8
    N_C = 8

    # ── STEP 2: overlap + modes ────────────────────────────────────────────
    print("Computing overlap integrals...")
    C_ov = compute_overlap(cfg.INLET_W, cfg.INLET_H,
                           cfg.CHAMBER_W, cfg.CHAMBER_H,
                           N_D, N_C)
    print(f"  Overlap shape: {C_ov.shape}")
    print(f"  C[0,0,0,0] = {C_ov[0,0,0,0]:.6f}  "
          f"(expect {cfg.INLET_W*cfg.INLET_H:.6f})")
    print("STEP 2 OK")

    # ── STEP 3: single-frequency test ─────────────────────────────────────
    f_test = 500.0
    B1, A2, B2, A3, kz1, kz2, kz3, norm_s, norm_c = solve_for_freq(
        f_test, C_ov, N_D, N_C)

    omega_t = 2 * np.pi * f_test
    S_in = cfg.INLET_W * cfg.INLET_H
    W_inc = S_in / (2.0 * cfg.RHO * cfg.C)
    W_trans = _power_flux(A3, kz3, norm_s, omega_t)
    W_ref   = _power_flux(B1, kz1, norm_s, omega_t)

    TL_mm = 10.0 * np.log10(W_inc / W_trans) if W_trans > 0 else np.inf
    TL_pw = float(_pw_formula([f_test])[0])

    print(f"\nAt {f_test} Hz:")
    print(f"  |A3[0,0]|² = {np.abs(A3[0,0])**2:.6f}")
    print(f"  |B1[0,0]|  = {np.abs(B1[0,0]):.4f}")
    print(f"  W_ref/W_inc  = {W_ref/W_inc:.4f}")
    print(f"  W_trans/W_inc= {W_trans/W_inc:.4f}")
    print(f"  energy balance (should ≈1): {(W_ref+W_trans)/W_inc:.4f}")
    print(f"  TL (mode-matching) = {TL_mm:.2f} dB")
    print(f"  TL (plane-wave)    = {TL_pw:.2f} dB")
    print("STEP 3 OK")

    # ── STEP 4: full frequency sweep ──────────────────────────────────────
    print("\nStep 4: TL sweep...")
    TL_arr, all_coeffs = compute_tl(cfg.FREQS, C_ov, N_D, N_C)
    TL_pw_arr = _pw_formula(cfg.FREQS)
    print(f"  {'freq':>8}  {'TL_mm':>8}  {'TL_pw':>8}")
    for i in range(0, len(cfg.FREQS), max(1, len(cfg.FREQS)//10)):
        print(f"  {cfg.FREQS[i]:8.1f}  {TL_arr[i]:8.2f}  {TL_pw_arr[i]:8.2f}")
    print("STEP 4 OK")

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── STEP 5: on-axis pressure at PLOT_FREQS ────────────────────────────
    print("\nStep 5: on-axis pressure at PLOT_FREQS...")
    onaxis_results = []
    for fq in PLOT_FREQS:
        coeffs = solve_for_freq(fq, C_ov, N_D, N_C)
        z_mm, p_c = compute_onaxis_pressure(coeffs, fq, N_D, N_C)
        p_abs = np.abs(p_c)
        # compute TL at this frequency using stored power-flux approach
        omega_fq = 2 * np.pi * fq
        B1_fq, A2_fq, B2_fq, A3_fq, kz1_fq, kz2_fq, kz3_fq, ns_fq, nc_fq = coeffs
        W_inc_fq = (cfg.INLET_W * cfg.INLET_H) / (2.0 * cfg.RHO * cfg.C)
        W_trans_fq = _power_flux(A3_fq, kz3_fq, ns_fq, omega_fq)
        tl_fq = 10.0 * np.log10(W_inc_fq / W_trans_fq) if W_trans_fq > 0 else np.inf
        onaxis_results.append((fq, tl_fq, z_mm, p_abs))
        print(f"  f={fq:.0f} Hz: max|p|={p_abs.max():.3f} Pa  TL={tl_fq:.2f} dB")
    print("STEP 5 OK")

    # ── STEP 6: plots ─────────────────────────────────────────────────────
    print("\nStep 6: generating plots...")

    # TL curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(cfg.FREQS, TL_arr, "b-o", ms=4, label="Mode-matching")
    ax.semilogx(cfg.FREQS, TL_pw_arr, "r--", lw=1.5, label="Plane-wave formula")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Transmission Loss (dB)")
    ax.set_title("Expansion Chamber TL — Mode Matching vs Plane Wave")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tl_mode_matching.png"), dpi=150)
    print("  Saved tl_mode_matching.png")
    plt.show(block=False)

    # On-axis pressure line plot
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    L1_mm = cfg.INLET_L * 1e3
    L2_mm = cfg.CHAMBER_L * 1e3
    fig, ax = plt.subplots(figsize=(10, 5))
    for (fq, tl_fq, z_mm, p_abs), color in zip(onaxis_results, colors):
        ax.plot(z_mm, p_abs, color=color, lw=1.8,
                label=f"{fq:.0f} Hz  (TL={tl_fq:.1f} dB)")
    ax.axvline(L1_mm, color="gray", lw=1.2, ls="--")
    ax.axvline(L1_mm + L2_mm, color="gray", lw=1.2, ls="--")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("|p| (Pa)")
    ax.set_title("On-axis pressure |p(z)| — Mode Matching")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fname = os.path.join(out_dir, "onaxis_pressure.png")
    fig.savefig(fname, dpi=150)
    print("  Saved onaxis_pressure.png")
    plt.show(block=False)

    print("STEP 6 OK")

    # ── STEP 7: convergence check ─────────────────────────────────────────
    print("\nStep 7: convergence check at 500 Hz...")
    C_ov15 = compute_overlap(cfg.INLET_W, cfg.INLET_H,
                             cfg.CHAMBER_W, cfg.CHAMBER_H, 15, 15)
    B1_15, A2_15, B2_15, A3_15, kz1_15, _, kz3_15, ns_15, _ = solve_for_freq(
        500.0, C_ov15, 15, 15)
    omega_15 = 2 * np.pi * 500.0
    W_trans_15 = _power_flux(A3_15, kz3_15, ns_15, omega_15)
    W_inc_15 = (cfg.INLET_W * cfg.INLET_H) / (2.0 * cfg.RHO * cfg.C)
    TL_15 = 10.0 * np.log10(W_inc_15 / W_trans_15) if W_trans_15 > 0 else np.inf

    # find 500 Hz index in FREQS
    i500 = np.argmin(np.abs(cfg.FREQS - 500.0))
    TL_8 = TL_arr[i500]
    print(f"  TL @ 500 Hz, N=8:  {TL_8:.3f} dB")
    print(f"  TL @ 500 Hz, N=15: {TL_15:.3f} dB")
    print(f"  Difference: {abs(TL_15 - TL_8):.3f} dB")
    print("STEP 7 OK")

    print("\nDone — mode_matching.py complete.")
    plt.show(block=True)
