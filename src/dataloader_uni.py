from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from pairdataset import PairDataset
import numpy as np

def get_transforms(args) -> None:
    transform_train_list = [
        transforms.Resize((args.h, args.w), interpolation=3),
        transforms.Pad(args.pad, padding_mode='edge'),
        transforms.RandomCrop((args.h, args.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_sat_list = [
        transforms.Resize((args.h, args.w), interpolation=3),
        transforms.Pad(args.pad, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((args.h, args.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    
    transform_val_list = [
        transforms.Resize(size=(args.h, args.w), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if args.color_jitter:
        transform_train_list = [transforms.ColorJitter(
            brightness=0.1, contrast=0,saturation=0.1,hue=0)] + transform_train_list
        transform_sat_list = [transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1,hue=0)] + transform_sat_list

    transform_train = transforms.Compose(transform_train_list)
    transform_sat = transforms.Compose(transform_sat_list)
    transform_val = transforms.Compose(transform_val_list)

    return transform_train, transform_sat, transform_val


def get_train_dataloaders(args):
    def _init_fn(worker_id):
        np.random.seed(int(args.seed) + worker_id)

    transform_train, transform_sat, _ = get_transforms(args)

    # dataset_train_drone = datasets.ImageFolder(
    #         os.path.join(args.dataset_dir, 'drone'),
    #         transform_train)
    dataset_train_drone = PairDataset(
        root=args.dataset_dir,
        view1='drone',
        view2='satellite',
        transform1=transform_train,
        transform2=transform_sat
    )
    dataloader_train_drone = DataLoader(
        dataset_train_drone,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        worker_init_fn=_init_fn
    )

    # dataset_train_sat = datasets.ImageFolder(
    #     os.path.join(args.dataset_dir, 'satellite'),
    #     transform_sat)
    dataset_train_sat = PairDataset(
        root=args.dataset_dir,
        view1='satellite',
        view2='drone',
        transform1=transform_sat,
        transform2=transform_train
    )
    print(len(dataset_train_sat))
    dataloader_train_sat = DataLoader(
        dataset_train_sat,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        worker_init_fn=_init_fn
    )

    return dataloader_train_drone, dataloader_train_sat


def get_val_dataloaders(args):
    _, _, transform_val = get_transforms(args)

    dataset_val_gallery_sat = datasets.ImageFolder(
        os.path.join(args.test_dir, 'gallery_satellite'),
        transform_val)
    dataloader_val_gallery_sat = DataLoader(
        dataset_val_gallery_sat,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=8)
    
    dataset_val_gallery_drone = datasets.ImageFolder(
        os.path.join(args.test_dir, 'gallery_drone'),
        transform_val)
    dataloader_val_gallery_drone = DataLoader(
        dataset_val_gallery_drone,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=8)
    
    dataset_val_query_sat = datasets.ImageFolder(
        os.path.join(args.test_dir, 'query_satellite'),
        transform_val)
    dataloader_val_query_sat = DataLoader(
        dataset_val_query_sat,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=8)
    
    dataset_val_query_drone = datasets.ImageFolder(
        os.path.join(args.test_dir, 'query_drone'),
        transform_val)
    dataloader_val_query_drone = DataLoader(
        dataset_val_query_drone,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=8)
    
    return dataloader_val_gallery_sat, dataloader_val_gallery_drone, \
            dataloader_val_query_sat, dataloader_val_query_drone


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["University"], default="CVUSA",
                        help="Which downstream task.")
    parser.add_argument("--dataset_dir", default="../dataset/University-Release/train", type=str,
                        help="The dataset path.")
    parser.add_argument("--test_dir", default="../dataset/University-Release/test", type=str,
                        help="The test dataset path.")
    parser.add_argument('--batchsize', default=8, type=int, 
                        help='batchsize')
    parser.add_argument("--val_batchsize", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--pad', default=10, type=int, 
                        help='padding')
    parser.add_argument('--h', default=256, type=int, 
                        help='height')
    parser.add_argument('--w', default=256, type=int, 
                        help='width')
    parser.add_argument('--seed', default=1234, type=int, 
                        help='random seed')
    parser.add_argument('--color_jitter', action='store_true', 
                        help='use color jitter in training')
    
    args = parser.parse_args()

    dataloader_train, dataloader_sat = get_train_dataloaders(args)
    data_train, data_train_pair, label_train = next(iter(dataloader_train))
    data_sat, data_sat_pair, label_sat = next(iter(dataloader_sat))
    print(data_train.shape, data_train_pair.shape, label_train, len(dataloader_train))
    print(data_sat.shape, data_sat_pair.shape, label_sat, len(dataloader_sat))

    dataloader_gs, dataloader_gd, dataloader_qs, dataloader_qd = get_val_dataloaders(args)
    data_gs, label_gs = next(iter(dataloader_gs))
    data_gd, label_gd = next(iter(dataloader_gd))
    data_qs, label_qs = next(iter(dataloader_qs))
    data_qd, label_qd = next(iter(dataloader_qd))

    print(data_gs.shape, label_gs, len(dataloader_gs))
    print(data_gd.shape, label_gd, len(dataloader_gd))
    print(data_qs.shape, label_qs, len(dataloader_qs))
    print(data_qd.shape, label_qd, len(dataloader_qd))

