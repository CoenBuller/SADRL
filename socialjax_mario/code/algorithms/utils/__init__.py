"""Shared utilities for all algorithms."""

from algorithms.utils.networks import (
    CNN,
    ActorCritic,
    Actor,
    Critic,
    SmallCNN,
    SmallActor,
    SmallCritic,
    mappo_networks,  # [SADRL fix 2026-11-03]
    add_agent_id,  # [SADRL fix 2026-11-08] AGENT_ID (round 16, A.1)
    agent_id_dim,  # [SADRL fix 2026-11-08]
    mappo_network_sizes,
)

from algorithms.utils.data_utils import (
    batchify,
    batchify_dict,
    batchify_numpy,
    unbatchify
)

from algorithms.utils.vdn_networks import (
    QNetwork
)

from algorithms.utils.io_utils import (
    save_params,
    load_params
)

from algorithms.utils.eval_utils import (
    evaluate_ippo,
    evaluate_mappo_style
)

from algorithms.utils.types import (
    Transition,
    MAPPOTransition,
    IRATTransition,
)

from algorithms.utils.transfer_utils import (
    s_from_ratio,
)

__all__ = [
    # Network architectures
    "CNN",
    "ActorCritic",
    "Actor",
    "Critic",
    "SmallCNN",
    "SmallActor",
    "SmallCritic",
    # VDN-specific networks
    "QNetwork",
    # Data manipulation utilities
    "batchify",
    "batchify_dict",
    "batchify_numpy",
    "unbatchify",
    # IO utilities
    "save_params",
    "load_params",
    # Evaluation utilities
    "evaluate_ippo",
    "evaluate_mappo_style",
    # Shared types
    "Transition",
    "MAPPOTransition",
    "IRATTransition",
    # TRANSFER utilities
    "s_from_ratio",
]
