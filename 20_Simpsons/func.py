from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from config import CONFIG
import pickle
from collections import Counter
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm, trange
import wandb
from sklearn.metrics import f1_score
import pandas as pd
import datetime
import os
from natsort import natsorted

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

class SimpsonsDataset(Dataset):
    """
    Load and prepare images
    """
    def __init__(self, files, mode):
        super().__init__()

        self.files = natsorted(files)

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
        # transformations
          transform = transforms.Compose([
              transforms.RandomRotation(degrees=30),
              transforms.RandomHorizontalFlip(p=0.5),
              transforms.ColorJitter(hue=.10, saturation=.10),
              transforms.RandomGrayscale(p=0.1),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
        else:
        # val and test transforms
          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y
        
    def _prepare_sample(self, image):
        image = image.resize((CONFIG['RESCALE_SIZE'], CONFIG['RESCALE_SIZE']))
        return image
    
def set_seed(SEED=CONFIG['SEED']):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow takes Tensors and return images"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)
    

def persons_distribution(labels, plot=False):
    """
    Simple function to see class distribution in dataset and plot it
    """
    persons_count = dict(Counter(labels).most_common())
    
    if plot:
        names = list(persons_count.keys())
        ctl = list(persons_count.values())
        
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(y=names, x=ctl, errwidth=0)
        ax.bar_label(ax.containers[0])
    return persons_count

                
def fit_epoch(model, train_loader, criterion, optimizer, wandb_freq=30, device=CONFIG['DEVICE']):
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0
    # f1 score
    running_preds = []
    running_true = []
    running_f1 = 0
    # for i, (inputs, labels) in enumerate(tqdm_notebook(train_loader)):
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='train')):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)
        
        #f1 score
        running_true.extend(labels.cpu().tolist())
        running_preds.extend(preds.cpu().tolist())
        
        if i % wandb_freq == 0 and CONFIG['LOG_ENABLE']:
            wandb.log({"run_loss": running_loss / processed_data})
    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    # f1 score
    train_f1 = f1_score(running_true, running_preds, average='weighted')
    if CONFIG['LOG_ENABLE']:
        wandb.log({
            "train_loss":train_loss,
            "train_acc":train_acc,
            "train_f1":train_f1
        })
#     return train_loss, train_acc
    return train_loss, train_f1, train_acc


def make_report(true, preds , label_encoder, verbose=False):

    
    true_names = label_encoder.inverse_transform(true)
    preds_names = label_encoder.inverse_transform(preds)
    
    report = classification_report(true_names, preds_names,
                                   output_dict=True,
                                   zero_division=1)
#     print(type(report))
    if type(report) == dict and verbose:
        report_ = pd.DataFrame(report).T
        report_ = report_.drop(['accuracy', 'weighted avg', 'macro avg']).sort_values(by='f1-score', ascending=False)
        report_.index.map(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
        display(report_)

    return report


def eval_epoch(model, val_loader, criterion, label_encoder, device=CONFIG['DEVICE'], verbose=False):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    epoch_answers_tmp = {'true':[], 'preds':[]}

    # for inputs, labels in val_loader:
    for i, (inputs, labels) in enumerate(tqdm(val_loader, desc='val')):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
        epoch_answers_tmp['true'].extend(labels.cpu().tolist())
        epoch_answers_tmp['preds'].extend(preds.cpu().tolist())
        
#     epoch_report = make_report(epoch_answers_tmp['true'], epoch_answers_tmp['preds'])
    make_report(epoch_answers_tmp['true'], epoch_answers_tmp['preds'],label_encoder, verbose=verbose)
        
        
    val_loss = running_loss / processed_size
    val_acc = running_corrects / processed_size
#     print(val_acc, running_correctsm processed_size)
    val_f1 = f1_score(epoch_answers_tmp['true'],epoch_answers_tmp['preds'], average='weighted')

    if CONFIG['LOG_ENABLE']:
        wandb.log({"val_loss":val_loss, "val_acc":val_acc, "val_f1":val_f1})
#     return val_loss, val_acc, epoch_report
    return val_loss, val_f1 ,val_acc, epoch_answers_tmp

def save_checkpoint(model,
                   epoch,
                   optimizer,
                   val_loss,
                   model_name:str=CONFIG['MODEL_NAME'],
                   desc:str=CONFIG['DESC'],
                   save_path:Path()=CONFIG['SAVE_PATH'],
                    ):
    if not save_path.exists():
        save_path.mkdir()
    filename =  datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M_{model_name}__epoch{epoch}_{val_loss:.3f}_{desc}.pt")
    save_path = save_path.joinpath(filename)
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': ":.2f".format(val_loss),
    }, save_path)
    print(f">>>> saved checkpoint to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_dir=CONFIG['SAVE_PATH'], checkpoint_file=False):
    """
    return model, optimizer, epoch, loss
    """
    if not checkpoint_file:
        models_checkpoints = list(checkpoint_dir.glob('**/*.pt'))
        last_checkpoint = max(models_checkpoints, key=os.path.getctime)

        checkpoint = torch.load(last_checkpoint)
    else:
        checkpoint = torch.load(checkpoint_file)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     model.train()
    print(f"load checkpoint from {last_checkpoint}")
    return model, optimizer#, checkpoint['epoch'], checkpoint['loss']


def train(
          train_loader,
          val_loader,
          model,
          epochs,
          batch_size,
          opt,
          scheduler,
          criterion,
          verbose=False,
          patience=CONFIG['PATIENCE'],
          checkpoint_interval=CONFIG['CHECKPOINT_INTERVAL'],
          early_stopping=CONFIG['EARLY_STOPPING']
         ):

    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

    total_true_preds = {'true':[],
                       'preds':[]}

    history = {'train_loss':[],
              'train_acc':[],
               'train_f1':[],
              'val_loss':[],
              'val_acc':[],
               'val_f1':[]
              }
    best_model_weights = {}
    last_loss = None
    epochs_since_best = 0

    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f} \
    train_f1 {t_f1:0.4f} val_f1 {v_f1:0.4f}"

    for epoch in trange(epochs, desc='epoch'):

        epoch_to_save = epoch + 1
        
        print('LR:', scheduler.get_last_lr())

        train_loss, train_f1, train_acc = fit_epoch(model, train_loader, criterion, opt)
        print("loss", train_loss, "f1", train_f1)
        
        val_loss, val_f1, val_acc,  tmp_true_preds = eval_epoch(model, val_loader, criterion, label_encoder, verbose=verbose)
        total_true_preds['true'].extend(tmp_true_preds['true'])
        total_true_preds['preds'].extend(tmp_true_preds['preds'])

        # save train history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        scheduler.step()

        if CONFIG['LOG_ENABLE']:
            wandb.log({'train_epoch_loss':train_loss,
                        'train_epoch_acc':train_acc,
                       'train_f1_weighted':train_f1,
                        'val_epoch_loss':val_loss,
                        'val_acc':val_acc,
                       'val_f1_weighted':val_f1
                        })

        tqdm.write(log_template.format(ep=epoch+1,
                                       t_loss=train_loss,
                                        v_loss=val_loss,
                                       t_acc=train_acc,
                                       v_acc=val_acc,
                                       t_f1=train_f1,
                                       v_f1=val_f1
                                      ))
        if checkpoint_interval != 0 and epoch_to_save % checkpoint_interval == 0:
            save_checkpoint(model, epoch_to_save, opt, val_loss)

        # early stopping
        if early_stopping:
            if last_loss is None:
                last_loss = val_loss

            elif last_loss < val_loss:
                epochs_since_best += 1
                save_checkpoint(model, epoch_to_save, opt, val_loss)
                print(f"Early stopping counter: {epochs_since_best}")
                if epochs_since_best > patience:
                    print(f"Stop training. Validation metric not improved for {patience} epochs.")

                    return history, total_true_preds
            else:
                last_loss = val_loss
                epochs_since_best = 0
    save_checkpoint(model, epoch_to_save, opt, val_loss)
    return history, total_true_preds


def predict(model, test_loader, device=CONFIG['DEVICE']):
    with torch.no_grad():
        logits = []
    
        for inputs in tqdm(test_loader):
            inputs = inputs.to(device)
            # model = model.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
            
    probs = torch.nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs


def set_parameter_requires_grad(model, freeze=False):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

        
# layers, where weights will updated
def show_unfreezed_layers(model):
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)


