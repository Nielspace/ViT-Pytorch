class Config:
    # Image configuration
    IMG_SIZE = 160
    PATCH_SIZE = 14
    CROP_SIZE = 160
    BATCH_SIZE = 1
    DATASET_SAMPLE = "full"

    # opimizer configuration
    LR = 0.003
    OPIMIZER = "Adam"

    # Model configuration
    NUM_CLASSES = 400
    IN_CHANNELS = 3
    HIDDEN_SIZE = 764
    NUM_ATTENTION_HEADS = 12

    LINEAR_DIM = 3072
    NUM_LAYERS = 12

    ATTENTION_DROPOUT_RATE = 0.1

    DROPOUT_RATE = 0.1
    STD_NORM = 1e-6
    EPS = 1e-6

    MPL_DIM = 128
    OUTPUT = "softmax"
    LOSS_FN = "nll_loss"

    # Device configuration
    DEVICE = ["cpu", "mps", "cuda"]

    # Training configuration
    N_EPOCHS = 1

