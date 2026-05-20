import math

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Top contour of a converging–diverging nozzle, piecewise from chamber to exit
# ---------------------------------------------------------------------------
# Zones along the axial coordinate x (throat is at x = 0):
#
#   0. Chamber:               y = rc                                       (x ≤ xc)
#   I.  Converging arc R2:    arc near the chamber                          (xc ≤ x ≤ x2)
#   II. Diagonal converging:  straight line at angle θ                      (x2 ≤ x ≤ x1)
#   III. Throat arc:          smooth turn around the throat                 (x1 ≤ x ≤ xexp)
#                              R1 on the converging side, Rexp on the diverging side
#   IV. Diverging conical:    straight line at angle α (pick α > θ for      (x ≥ xexp)
#                              a pronounced exit slope and a larger exit diameter)
# ---------------------------------------------------------------------------

def _zone_boundaries(rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha):
    R1   = rt * R1_rt
    R2   = R1 * R2_R1
    Rexp = rt * Rexp_rt
    x1   = -R1 * math.sin(theta)                              # II → III (converging)
    r1   =  rt + R1 * (1.0 - math.cos(theta))
    r2   =  rc + R2 * (math.cos(theta) - 1.0)
    x2   =  x1 - (r2 - r1) / math.tan(theta)                  # I  → II
    xc   =  x2 - R2 * math.sin(theta)                         # 0  → I
    xexp =  Rexp * math.sin(alpha)                            # III (diverging) → IV
    return R1, R2, Rexp, xc, x2, x1, xexp, r1, r2


def nozzle_top_contour(x, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha):
    """Local radius y_top(x) of the nozzle's top wall, piecewise across the four zones."""
    R1, R2, Rexp, xc, x2, x1, xexp, r1, _ = _zone_boundaries(
        rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)

    if x <= xc:                       # Chamber (constant radius)
        return rc
    if x <= x2:                       # I: converging arc R2 on the chamber side
        return rc - R2 * (1.0 - math.sqrt(1.0 - ((x - xc) / R2) ** 2))
    if x <= x1:                       # II: diagonal converging at angle θ
        return r1 - (x - x1) * math.tan(theta)
    if x <= 0.0:                      # III, converging side: throat arc R1
        return rt + R1 * (1.0 - math.cos(math.asin(-x / R1)))
    if x <= xexp:                     # III, diverging side: throat arc Rexp
        return rt + Rexp * (1.0 - math.sqrt(1.0 - (x / Rexp) ** 2))
    # IV: diverging conical at angle α
    r_at_xexp = rt + Rexp * (1.0 - math.cos(alpha))
    return r_at_xexp + (x - xexp) * math.tan(alpha)


def nozzle_top_zone(x, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha):
    """Return the zone index (0..4) at axial position x. Useful for plotting/coloring."""
    _, _, _, xc, x2, x1, xexp, _, _ = _zone_boundaries(
        rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)
    if x <= xc:   return 0
    if x <= x2:   return 1
    if x <= x1:   return 2
    if x <= xexp: return 3
    return 4


def nozzle_bot_contour(x):
    """Bottom contour: y = 0 (axis of symmetry) for axi-/symmetric runs."""
    return 0.0


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_nozzle(rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha, L_N,
                n_samples=2000, savepath=None, show=True):
    """Plot the four-zone nozzle contour, mirrored across the axis, and annotate.

    L_N is the axial length of the diverging section (from x = 0 to the exit).
    """
    R1, R2, Rexp, xc, x2, x1, xexp, r1, r2 = _zone_boundaries(
        rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)

    # Sample from a bit before the chamber arc to the exit plane.
    x_left = xc - 0.10 * abs(xc) - 1e-4
    x_right = L_N
    xs = np.linspace(x_left, x_right, n_samples)
    ys = np.array([nozzle_top_contour(x, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha) for x in xs])
    zs = np.array([nozzle_top_zone (x, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha) for x in xs])

    zone_meta = [
        (0, '0 — chamber',                            '#bbbbbb'),
        (1, 'I — converging arc (R2)',                '#1f77b4'),
        (2, 'II — diagonal converging (angle θ)',     '#2ca02c'),
        (3, 'III — throat arc (R1 / Rexp)',           '#ff7f0e'),
        (4, 'IV — diverging conical (angle α)',       '#d62728'),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Top half: colored by zone. Bottom half: light mirror.
    for zid, label, color in zone_meta:
        m = zs == zid
        if not np.any(m):
            continue
        ax.plot(xs[m] * 1e3, ys[m] * 1e3,       color=color, lw=2.2, label=label)
        ax.plot(xs[m] * 1e3, -ys[m] * 1e3,      color=color, lw=2.2, alpha=0.35)

    # Throat plane and exit plane.
    ax.axvline(0.0, color='k', ls=':', lw=0.8)
    ax.axvline(L_N * 1e3, color='k', ls=':', lw=0.8)
    ax.text(0.0, -1.05 * ys.max() * 1e3, 'throat',
            ha='center', va='top', fontsize=8)
    ax.text(L_N * 1e3, -1.05 * ys.max() * 1e3, 'exit',
            ha='center', va='top', fontsize=8)

    # Zone boundary markers along the top wall.
    for xb, name in [(xc, 'xc'), (x2, 'x2'), (x1, 'x1'), (xexp, 'xexp')]:
        yb = nozzle_top_contour(xb, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)
        ax.plot([xb * 1e3], [yb * 1e3], 'k.', ms=5)
        ax.annotate(name, (xb * 1e3, yb * 1e3),
                    textcoords='offset points', xytext=(4, 6), fontsize=8)

    # Axis of symmetry.
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)

    r_exit = nozzle_top_contour(L_N, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)
    info = (
        f"rt = {rt*1e3:.3f} mm,  rc = {rc*1e3:.3f} mm,  r_exit = {r_exit*1e3:.3f} mm\n"
        f"R1 = {R1*1e3:.3f} mm,  R2 = {R2*1e3:.3f} mm,  Rexp = {Rexp*1e3:.3f} mm\n"
        f"θ = {math.degrees(theta):.1f}°,  α = {math.degrees(alpha):.1f}°,  L_N = {L_N*1e3:.1f} mm\n"
        f"area ratio (Aexit / A*) = {(r_exit/rt)**2:.2f}"
    )
    ax.text(0.99, 0.97, info, transform=ax.transAxes,
            ha='right', va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6'))

    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('Four-zone converging–diverging nozzle')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=120)
    if show:
        plt.show()
    return fig, ax


if __name__ == '__main__':
    # Default demonstration parameters — same as rhea_flow_solver.py defaults after the
    # geometry edit (R2_R1 raised so zone I is visible, α raised so zone IV is steep).
    rt       = 1.0e-3
    rc       = 2.5e-3
    R1_rt    = 5.0
    R2_R1    = 0.5
    Rexp_rt  = 2.0
    theta    = math.radians(15.0)
    alpha    = math.radians(20.0)
    L_N      = 30.0e-3
    plot_nozzle(rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha, L_N,
                savepath='nozzle_geometry.png', show=False)
    print('Saved nozzle_geometry.png')
