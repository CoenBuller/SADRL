"""[SADRL fix 2026-11-17] Round 17 (E.4): per-env "role" action for the MAPPO / CC-MAPPO diagnostics (logging only).

Each env's characteristic action, counted per agent in the logs ({role}_agent{k}, {role}_rank{k}, credit_on_{role},
{role}_fraction). For clean_up it is the cleaning beam, so the Clean Up log names stay "clean_*" as before.
"""

STAY_ACTION = 6  # "stay" is action 6 in every env below

ROLE_ACTIONS = {
    "clean_up": (8, "clean"),                # Actions.zap_clean
    "mushrooms": (7, "zap"),                 # Actions.zap_forward
    "gift": (7, "give"),                     # Actions.zap_forward (the gift beam; info["give_actions"])
    "pd_arena": (7, "interact"),             # Actions.interact
    "harvest_common_open": (7, "zap"),       # Actions.zap_forward
}


def role_action(env_name):
    """(action id, log name) of env_name's role action."""
    if env_name not in ROLE_ACTIONS:
        raise ValueError(f"no role action defined for env {env_name!r}; add it to algorithms/utils/env_roles.py")
    return ROLE_ACTIONS[env_name]
