import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
        except Exception:
            print(f"Open Exception: {path}")
            img = Image.new(mode='RGB', size=(512,512), color='#000000')
    
        try:
            img = img.convert('RGB')
        except Exception as e:
            print(f'{e}: {path}')
        finally:
            return img

class HardDataset(Dataset):

    def __init__(self, root, json_file, transform=None, s_transform=None, loader=pil_loader):
        super().__init__()
        self.root = root
        
        fp = open(json_file, 'r')
        self.hardsamples = json.load(fp)
        fp.close()

        self.samples = self.hardsamples['easy'] + self.hardsamples['medium']

        self.transform = transform
        self.loader = loader

        
    def __getitem__(self, index):

        query_path = self.samples[index]['query']
        gallery_paths = self.samples[index]['gallery']

        pos_gallery_path = gallery_paths[-1]
        neg_gallery_path = gallery_paths[np.random.randint(len(gallery_paths) - 1)]

        paths = [query_path, pos_gallery_path, neg_gallery_path]
        paths = [os.path.join(self.root, p) for p in paths]

        imgs = [self.loader(p) for p in paths]
        q, pg, ng = [self.transform(img) for img in imgs]


        return q, pg, ng
    
    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    jsonfile = '/root/MLPN_workshop/hardsample/hardsample_eval.json'
    fp = open(jsonfile, 'r')
    hardsample = json.load(fp)
    print(len(hardsample['easy']), len(hardsample['medium']), len(hardsample['hard']))