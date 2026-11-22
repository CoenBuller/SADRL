from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    ENV_NAME: str = "gift"
    N_AGENTS: int = 5
    N_ENVS: int = 100
    TOTAL_TIMESTEPS: int = int(1e8)
    NUM_STEPS: int = 100
    INNER_STEPS: int | None = None
    SEED: int = 30
    MAP_H: int | None = None
    MAP_W: int | None = None

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
    SMALL: bool = True  # Use the small actor critic or not.
    HIDDEN_DIM: int = 32

    # RNN & auxiliary prediction settings
    RNN_WARMUP: int = 10  # Warm up for 's' steps before auxiliary prediction features
    WINDOW: int = 30  # RNN looks over a window of 30 steps
    PREDICT_POS: bool = False  # Auxiliary absolute-position prediction from RNN output
    POS_COEF: float = 0.1
    PREDICT_NEXT_STATE: bool = False  # Auxiliary next-embedding prediction from RNN output
    NEXT_STATE_COEF: float = 0.05  # Weight on next-embedding loss
    NEXT_VAR_MIN_STD: float = 0.1  # Minimum per-dim std floor to prevent predictor collapse
    NEXT_VAR_COEF: float = 0.1  # Weight on variance floor penalty
    LSTM: bool = False  # If True, use an LSTM. Default to GRU otherwise.

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