MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
RANDOM_SEED = 42
MODEL_NAME = "allenai/scibert_scivocab_uncased"
LABELS_MAP = {
    "human_medicine" : 0,
    "veterinary_medicine" : 1
    }