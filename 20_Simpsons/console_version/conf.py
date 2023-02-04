from pathlib import Path

CONFIG = {

    'IMG_SIZE': 380,
    'TRAIN_DIR' : '../data/train/',
    'TEST_DIR' : '../data/testset/',
    'LOG_DIR' :'logdir',
    'TRAINED_MODEL_PATH' :'saved_models',
    'SEED': 2022,
    'BATCH_SIZE': 64,
    'EPOCHS' :10,
    'DATA_MODES' :['train' ,'val' ,'test']
}