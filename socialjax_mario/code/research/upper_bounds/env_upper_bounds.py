"""Upper bounds on the team return per episode, from the environment code (round 19, B.1).

No training and no simulation: each bound follows from one rule of the env that caps how fast reward can be made,
with every other difficulty (walking, finding partners, coordination) ignored. So these are ceilings, not targets:
a perfect policy would still sit below them. All are team returns (sum over agents), 1000-step episodes, the agent
counts we train with. Constants are read from the code; file:line references are next to each one.

Usage, from the repo root:
    ~/miniconda3/envs/SocialJax/bin/python research/upper_bounds/env_upper_bounds.py
"""
import re

STEPS = 1000


def gift_coin_cells():
    # COINS_POTENTIAL = the 'T' cells of the map (gift.py:286, char_to_int 'T': 0)
    src = open("socialjax/environments/gift/gift.py").read()
    rows = re.findall(r'"([^"]*)"', re.search(r"map_ASCII = \[(.*?)\]", src, re.S).group(1))
    return sum(r.count("T") for r in rows)


def mushrooms(n=5):
    # Team reward per mushroom (mushrooms.py:1042-1058): red 1 (eater), green 2 (2/N each), blue 3 (3/(N-1) to each
    # other agent), orange -0.2 N. Eating freezes the eater (stay_time 10 / 15 / 20, mushrooms.py:1011-1029); the
    # counter drops by 1 before the agent acts (mushrooms.py:847), so one mushroom per 10 / 15 / 20 steps at best.
    # Blue regrows itself (2 checks at p = 0.6 per blue eaten, mushrooms.py:916-925, cap 5), so supply doesn't bind.
    best_rate = max(1 / 10, 2 / 15, 3 / 20)  # blue: 0.15 team reward per agent-step
    return n * STEPS * best_rate


def gift(n=5):
    # The map starts with no coins (gift.py:1072-1100); each empty coin cell spawns one with p = 0.0002 per step
    # (gift.py:835). Gifting (zap an adjacent agent, gift.py:694-708): 1 raw -> 3 refined for the receiver,
    # 1 refined -> 3 final, 1 final -> 1 final. Consuming pays 1 per token in the bag (gift.py:930). So one coin is
    # worth at most 9, and the expected number of coins that appear is at most cells x p x steps (fewer once cells
    # are occupied by coins nobody has picked up).
    cells = gift_coin_cells()
    coins = cells * 0.0002 * STEPS
    return 9 * coins, cells, coins


def pd_arena(n=4):
    # Reward only when one agent zaps another and both hold resources (pd_arena.py:1565-1585); best case both hold
    # only cooperate tokens: 3 + 3 = 6 (payoff, pd_arena.py:175). The zapped agent leaves the map for U{10..100}
    # steps, mean 55 (pd_arena.py:2009); the zapper is frozen 2 x freeze_penalty = 10 steps (pd_arena.py:1712-1745,
    # the penalty is added twice). Tokens regrow at p = 0.1 per spawn cell per step, so supply doesn't bind.
    one_sided = 6 / (1 + 10 + 55)          # one zap: zapper's zap + freeze, zapped's absence
    mutual = 12 / (2 * (1 + 55))           # both zap each other in the same step: both pay, both are removed
    rate = max(one_sided, mutual)
    return n * STEPS * one_sided, n * STEPS * rate


def clean_up(n=7):
    # 122 apple cells, all full at reset; each empty apple cell regrows with p <= 0.05 per step, and 0 while the
    # river's dirt fraction is >= 0.4 (clean_up.py:214-217, 1098-1114). The river starts 47% dirty, so someone has to
    # clean before anything regrows. Supply bound: 122 + 0.05 x 122 x steps.
    cells = 122
    return cells + 0.05 * cells * STEPS


if __name__ == "__main__":
    g, cells, coins = gift()
    p1, p2 = pd_arena()
    rows = [
        ("clean_up (7)", clean_up(), "apple supply: 122 at reset + 0.05 x 122 per step (minus the cleaning start-up)"),
        ("mushrooms (5)", mushrooms(), "digestion: 3 team reward per blue, one blue per 20 steps per agent"),
        ("gift (5)", g, f"coin supply: {cells} cells x 0.0002 x 1000 = {coins:.0f} coins, x 9 each"),
        ("pd_arena (4)", p2, f"respawn: 6 per zap for ~66 agent-steps ({p1:.0f}); mutual zaps {p2:.0f}"),
    ]
    print(f"{'env':<15} {'upper bound':>12}  rule")
    for env, b, why in rows:
        print(f"{env:<15} {b:>12.0f}  {why}")
