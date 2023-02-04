import numpy as np
import platform
from pathlib import Path
from torch import manual_seed
from sklearn.preprocessing import LabelEncoder
import torch


def choose_platform(platform):
    if platform == 'Darwin':  # for macos m1 gpu acceleration
        device = 'mps'
    elif platform == 'Linux' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device


label_encoder = LabelEncoder()

CONFIG = {
    'DEBUG':False,
    # wandb
    'LOG_ENABLE': True,
    'PROJECT_NAME': 'Simpsons_Pytorch',
    'DESC': '(weighted_sampler)',
    # model
    'MODEL_NAME': 'efficientnet_b4',
    'WEIGHTS': 'DEFAULT',
    'RESCALE_SIZE': 224,
    # feature extract
    'FEATURE_EXTRACT_LR': 0.01,
    'FEATURE_EXTRACT_EPOCHS': 10,
    'FEATURE_EXTRACT_BATCH_SIZE': 256,
    # fine tune
    'LR':0.01,
    'EPOCHS': 30,
    'BATCH_SIZE': 256,
    
    'SEED': 2023,
    'DEVICE': choose_platform(platform.system()),
    'DATA_MODES': ['train', 'val', 'test'],
    # Paths
    'TRAIN_DIR': Path('data/train/'),
    'TEST_DIR': Path('data/testset/'),
    'SAVE_PATH': Path('./saved_models'),
    'CHECKPOINT_INTERVAL': 10,

    # early stopping
    'PATIENCE': 5,
    'EARLY_STOPPING': True

}

# CONFIG['CHECKPOINT_INTERVAL'] = CONFIG['EPOCHS'] // 3
CONFIG['NUM_CLASSES'] = np.unique([p.parent.name for p in CONFIG['TRAIN_DIR'].glob('**/*.jpg')]).shape[0]

