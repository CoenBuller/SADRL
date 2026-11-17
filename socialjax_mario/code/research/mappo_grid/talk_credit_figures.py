"""Figures for the credit-assignment part of the SADRL 5 talk: toy-example tables and formula images.

Usage, from the repo root:
    ~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/talk_credit_figures.py

Toy example: 2 agents, one step, each picks either option with probability 1/2. B eats an apple (+1) or not (0);
A cleans (+0.5, apples later) or stays (0). G = r_A + r_B. MAPPO: A_hat = G - V(s), V(s) = E[G] = 0.75.
Leave-one-out credit: c^i = G - b^i(a^-i), b^i = E over agent i's own action of G with the others' actions fixed.
Formulas follow research/cc_mappo (loo, loo_single) and docs/cc_mappo_single_critic.md.
Writes docs/talk_2026-11-09/c*_*.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "docs", "talk_2026-11-09")
INK, PURPLE, GREEN, RED = "#333333", "#9b59b6", "#1e8f5a", "#c0392b"
plt.rcParams["mathtext.fontset"] = "cm"

R_EAT, R_CLEAN = 1.0, 0.5
V_S = 0.5 * R_EAT + 0.5 * R_CLEAN  # 0.75


def toy_rows():
    rows = []
    for a_clean, b_eat in [(1, 1), (0, 1), (1, 0), (0, 0)]:
        g = a_clean * R_CLEAN + b_eat * R_EAT
        b_a = b_eat * R_EAT + 0.5 * R_CLEAN  # A's baseline: B's action fixed, average over A's own choice
        b_b = a_clean * R_CLEAN + 0.5 * R_EAT  # B's baseline: A's action fixed, average over B's own choice
        what = f"A {'cleans' if a_clean else 'stays'}, B {'eats' if b_eat else 'does not eat'}"
        rows.append((what, g, g - V_S, g - b_a, b_a, g - b_b, b_b))
    return rows


def sgn(x):
    return f"{x:+.2f}".replace("-", "−")


def colour(x):
    return GREEN if x > 0 else RED


def table(head, body, colours, widths, fname, fs=20):
    fig, ax = plt.subplots(figsize=(13, 0.9 + 0.62 * len(body)))
    ax.axis("off")
    t = ax.table(cellText=body, colLabels=head, loc="center", cellLoc="center", colWidths=widths)
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    n = len(body) + 1
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#cfd6dd")
        if i == 0:
            c.set_height(1.25 / n)
            c.set_text_props(weight="bold", color=INK)
            c.set_facecolor("#eef2f6")
        else:
            c.set_height(0.8 / n)
            col = colours[i - 1][j]
            c.set_text_props(color=col or INK, weight="bold" if col else "normal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def toy_tables():
    rows = toy_rows()
    body = [[w, f"{g:g}", f"{g:g} − 0.75 = {sgn(a)}"] for w, g, a, *_ in rows]
    cols = [[None, None, colour(a)] for _, _, a, *_ in rows]
    table(["What happened", "Team return G", "MAPPO: A gets\nÂ = G − V(s)"], body, cols, [0.42, 0.22, 0.36],
          "c1_toy_mappo.png")
    body = [[w, f"{g:g}", sgn(a), f"{g:g} − {ba:g} = {sgn(ca)}", f"{g:g} − {bb:g} = {sgn(cb)}"]
            for w, g, a, ca, ba, cb, bb in rows]
    cols = [[None, None, colour(a), colour(ca), colour(cb)] for _, _, a, ca, _, cb, _ in rows]
    table(["What happened", "G", "MAPPO:\nA gets Â", "Credit: A gets\nG − b(s, B's action)", "Credit: B gets\nG − b(s, A's action)"],
          body, cols, [0.30, 0.08, 0.14, 0.24, 0.24], "c2_toy_credit.png", fs=18)
    for r in rows:
        print(r)


def formulas(lines, fname, width=13, fs=34, gap=1.0):
    """lines: list of (mathtext, colour). Rendered top to bottom, left aligned."""
    h = gap * len(lines) + 0.2
    fig = plt.figure(figsize=(width, h))
    for k, (txt, col) in enumerate(lines):
        fig.text(0.01, 1 - (k + 0.55) * gap / h, txt, fontsize=fs, color=col, va="center", ha="left")
    fig.savefig(os.path.join(OUT, fname), dpi=200, bbox_inches="tight", pad_inches=0.15, transparent=True)
    plt.close(fig)


def formula_images():
    formulas([
        (r"$\mathrm{MAPPO\ critic:}\quad \min_{\theta}\ \mathbb{E}\left[\left(V_{\theta}(s_t)-G_t\right)^2\right]"
         r"\qquad \hat{A}_t = G_t - V_{\theta}(s_t)$", INK),
        (r"$\mathrm{credit\ critic\ (loo):}\quad \min_{\phi}\ \mathbb{E}\left[\left(V_{\phi}(s_t, a_t^{-i}, i)-G_t\right)^2\right]"
         r"\qquad c_t^i = G_t - V_{\phi}(s_t, a_t^{-i}, i)$", PURPLE),
        ("$G_t$: λ-return (rewards of the next steps, bootstrapped from $V_{\\theta}$)", INK),
    ], "c3_two_losses.png", width=16, fs=30, gap=1.05)
    formulas([
        (r"$\mathrm{COMA:}\quad c^i = Q_{\phi}(s, a^i, a^{-i}) \;-\; \sum_{a'} \pi(a' \mid o^i)\, Q_{\phi}(s, a', a^{-i})$", INK),
        (r"$\mathrm{loo:}\quad\ \ c^i = G_t \;-\; V_{\phi}(s, a^{-i}, i)$", PURPLE),
    ], "c4_coma_vs_loo.png", width=14, fs=34, gap=1.2)
    formulas([
        (r"$V_{\psi}(s, \mathrm{all\ masked}) = V(s)\qquad V_{\psi}(s, \mathrm{mask}\ i) = V(s, a^{-i})$", INK),
        (r"$L(\psi) = \frac{1}{2}\left(V(s_t) - G_t\right)^2 + \frac{1}{2}\cdot\frac{1}{7}\sum_{i=1}^{7}\left(V(s_t, a_t^{-i}) - G_t\right)^2$", PURPLE),
        ("$V(s)$ bootstraps $G_t$ (GAE);   the actor learns from $c_t^i = G_t - V(s_t, a_t^{-i})$", INK),
    ], "c5_single_critic.png", width=14, fs=32, gap=1.35)


if __name__ == "__main__":
    toy_tables()
    formula_images()
