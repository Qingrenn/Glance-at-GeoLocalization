import torch.utils.data as Data
import os
from PIL import Image
import numpy as np

def PIL_loader(path):
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
        except Exception as e:
            print(f'{e}: {path}')
            img = Image.new(mode='RGB', size=(512,512), color='#000000')
        
        try:
            img = img.convert('RGB')
        except Exception as e:
            print(f'{e}: {path}')
        finally:
            return img

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_pair_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, target, class_to_idx[target])
                    images.append(item)
    return images

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class PairDataset(Data.Dataset):
    def __init__(self, root, view1, view2,
                transform1=None,
                transform2=None, 
                loader=PIL_loader):
        
        view1_dir = os.path.join(root, view1)
        classes, class_to_idx = find_classes(view1_dir)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(view1_dir, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            if len(imgs) == 0:
                raise RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS))
        
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform1 = transform1
        self.transform2 = transform2
        self.loader = loader
        self.view1 = view1
        self.view2 = view2
        self.view2_dir = os.path.join(root, view2)

    def _get_pair_sample(self, _cls):
        folder_root = os.path.join(self.view2_dir, str(_cls))
        
        assert os.path.isdir(folder_root), 'no pair drone image'
        
        img_path = []
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        
        return result_path
    

    def __getitem__(self, index):
        path, _cls, target = self.imgs[index]
        img1 = self.loader(path)
        if self.transform1 is not None:
            img1 = self.transform1(img1)

        # positive pair
        path = self._get_pair_sample(_cls)
        img2 = self.loader(path)
        if self.transform2 is not None:
            img2 = self.transform2(img2)

        return img1, img2, target

    def __len__(self):
        return len(self.imgs)