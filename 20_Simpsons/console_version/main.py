import pandas as pd
import numpy as np

import albumentations as A

import torch
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2 

from conf import *

from dataset import *

if __name__ == '__main__':
    if not CONFIG['logdir'].exists():
        CONFIG['logdir'].mkdir()
        print('logdir created.')
    if not CONFIG['TRAINED_MODEL_PATH'].exists():
        CONFIG['TRAINED_MODEL_PATH'].mkdir()
        print(f"{CONFIG['TRAINED_MODEL_PATH']} folder created")

    train_files = Path(CONFIG['TRAIN_DIR'].rglob('*/*.jpg'))


