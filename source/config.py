class Config:
    #Image configuration
    IMG_SIZE = 128
    PATCH_SIZE = 10
    CROP_SIZE = 100
    BATCH_SIZE = 1
    N_SAMPLES = 400
    N_TRAIN = 400   
    N_VAL = 400
    N_TEST = 400

    #opimizer configuration
    LR = 0.003
    OPIMIZER = 'Adam'

    #Model configuration
    N_CLASSES = 400
    IN_CHANNELS = 3
    HIDDEN_SIZE = 768
    NUM_ATTENTION_HEADS = 12
    
    ATTENTION_DROPOUT_RATE = 0.1
 
    DEPTH = 400
    HEADS = 8
    MPL_DIM = 128
    OUTPUT = 'softmax'
    LOSS_FN = 'nll_loss'


    #Device configuration
    DEVICE = ["cpu","mps","cuda"]

    #Training configuration
    N_EPOCHS = 100
    TRAIN_LOSS_HISTORY = []
    VAL_LOSS_HISTORY = []