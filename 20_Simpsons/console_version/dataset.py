from conf import *
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision import transforms
import albumentations as A

class SimpsonsDataset(Dataset):
    """
    Pytorch dataset
    files - path to train/test files
    mode - one of: 'train' 'val' 'test'
    """
    def __init__(self, files, mode):
        super().__init__()
        # files to load
        self.files = sorted(files)
        # mode
        self.mode = mode

        if self.mode not in CONFIG['DATA_MODES']:
            print(f"{self.mode} is not correct; correct modes: {CONFIG['DATA_MODES']}")
            raise NameError

        self.len_ = len(self.files)
     
        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)
                      
    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        image = Image.open(file)
        image = image.convert("RGB")
        image.load()
        return image
  
    def __getitem__(self, index):

        if self.mode == 'train':

            transform = A.Compose([
                A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], always_apply=True),
                A.Rotate(p=0.3),
                A.CoarseDropout(p=0.3),
                A.ShiftScaleRotate(p=0.3),
                A.Normalize(),
                ToTensorV2()
            ])

        else:
            transform = A.Compose([
                A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], always_apply=True),
                A.Normalize(),
                ToTensorV2()])
        x = self.load_sample(self.files[index])
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y
        
