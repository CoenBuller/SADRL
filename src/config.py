from dataclasses import dataclass

@dataclass
class Config:
    # Environment
    ENV_NAME: str = "gift"
    N_AGENTS: int = 5
    N_ENVS: int = 100
    TOTAL_TIMESTEPS: int = int(1e8)
    NUM_STEPS: int = 100
    SEED: int = 30

    # Storage
    WANDB: bool = True
    GIF_FRAMES: int = 1000
    OUT_DIR: str = "results"

    # Reputation 
    OWN_POS: tuple[int, int] = (9, 5)
    REP_DECAY: float = 0.95

    # Algorithm
    ALGO: str = "RecurrentCOMA"
    COMMON_REWARD: bool = True
    ACTIVATION: str = "relu"
    REPUTATION: bool = False
    SMALL: bool = True     # Use the small actor criti or not.
    HIDDEN_DIM: int = 32

    # RNN settings
    RNN_WARMUP: int = 10    # If using RNN network and prediction features over the recurrent steps, we let it warm up for 's' steps 
    WINDOW: int = 30        # RNN looks over a window of 30 steps

    # Optimization loop
    UPDATE_EPOCHS: int = 10
    NUM_MINIBATCHES: int = 100

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
