# Import libraries
import torch, torchvision, os
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn; from PIL import Image
from torchvision import transforms as T; from glob import glob
# Set manual seed
torch.manual_seed(2023)

class CustomDataset(Dataset):

    """

    This class gets several parameters and returns dataset to train an AI model.

    Parameters:

        root             - path to data, str;
        transformations  - transformations to be applied, torchvision object.    
    
    """    
    
    def __init__(self, root, data, transformations = None):
        
        self.transformations = transformations
        self.im_paths = [im_path for im_path in sorted(glob(f"{root}/{data}/*/*")) if "jpg" in im_path]
        
        self.cls_names, self.cls_counts, count, data_count = {}, {}, 0, 0
        for idx, im_path in enumerate(self.im_paths):
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names: self.cls_names[class_name] = count; self.cls_counts[class_name] = 1; count += 1
            else: self.cls_counts[class_name] += 1
        
    def get_class(self, path): return os.path.dirname(path).split("/")[-1]
    
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        
        im_path = self.im_paths[idx]
        im = Image.open(im_path)
        gt = self.cls_names[self.get_class(im_path)]
        
        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt
    
def get_dls(root, transformations, bs, split = [0.9, 0.05], ns = 4):
    
    tr_ds = CustomDataset(root = root, data = "train", transformations = transformations)
    ts_ds = CustomDataset(root = root, data = "test", transformations = transformations)
    
    all_len = len(tr_ds); tr_len = int(all_len * split[0]); val_len = all_len - tr_len
    tr_ds, val_ds = random_split(dataset = tr_ds, lengths = [tr_len, val_len])
    
    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(val_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, ts_ds.cls_names
