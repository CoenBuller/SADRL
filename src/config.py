from dataclasses import dataclass

@dataclass
class Config:
    # Environment
    ENV_NAME: str = "gift"
    N_AGENTS: int = 5
    N_ENVS: int = 256
    TOTAL_TIMESTEPS: int = int(3e8)
    NUM_STEPS: int = 1000
    SEED: int = 30

    # Storage
    WANDB: bool = True
    GIF_FRAMES: int = 1000
    OUT_DIR: str = "results"

    # Reputation 
    OWN_POS: tuple[int, int] = (9, 5)
    REP_DECAY: float = 0.95

    # Algorithm
    ALGO: str = "IPPO"
    COMMON_REWARD: bool = True
    ACTIVATION: str = "relu"
    REPUTATION: bool = True

    # Optimization loop
    UPDATE_EPOCHS: int = 2
    NUM_MINIBATCHES: int = 500

    # Optimizer
    MAX_GRAD_NORM: float = 0.5
    ANNEAL_LR: bool = True
    LR: float = 1e-4

    # Advantage & Loss
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    VF_COEF: float = 0.5 
    ENT_COEF: float = 0.01
