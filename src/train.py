import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from dataloader_uni import get_train_dataloaders, get_val_dataloaders
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule, ConstantLRSchedule
from loss import one_LPN_output, nceloss, decouple_loss
from functools import partial
from utils.SAM import SAM
import yaml
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy
import copy
from mlpn import CSWinTransv2_threeIn

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', default='0, 1', type=str, 
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument("--dataset", choices=["University"], default="University",
                    help="Which downstream task.")
parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
parser.add_argument("--dataset_dir", default="../autodl-tmp/University-Release/train", type=str,
                    help="The dataset path.")
parser.add_argument("--test_dir", default="../autodl-tmp/University-Release/test", type=str,
                    help="The test dataset path.")

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')

parser.add_argument("--total_epoch", default=300, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument('--learning_rate', default=0.005, type=float, 
                    help='learning rate')
parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
parser.add_argument('--batchsize', default=8, type=int, 
                    help='batchsize')
parser.add_argument("--val_batchsize", default=8, type=int,
                    help="Total batch size for eval.")

parser.add_argument('--droprate', default=0.75, type=float, 
                    help='drop rate')
parser.add_argument('--pad', default=10, type=int, 
                    help='padding')
parser.add_argument('--h', default=256, type=int, 
                    help='height')
parser.add_argument('--w', default=256, type=int, 
                    help='width')
parser.add_argument('--block', default=4, type=int, 
                    help='the num of block')

parser.add_argument('--seed', default=1234, type=int, 
                    help='random seed')
parser.add_argument('--color_jitter', action='store_true', 
                    help='use color jitter in training')

args = parser.parse_args()

# Gloabal
lpnloss = partial(one_LPN_output, criterion=nn.CrossEntropyLoss(), block=args.block)

tensorboard_writer = SummaryWriter(os.path.join(args.output_dir, 'run'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()
args.device = device

global_step = 0

# gpu
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

if len(gpu_ids) > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.enabled = True
    cudnn.benchmark = True
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    cudnn.benchmark = True

def save_model(args, model, optimizer, epoch, best_acc):

    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"model_checkpoint_${epoch}.pth")
    checkpoint = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc
    }
    torch.save(checkpoint, model_checkpoint)
    
    print("Saved model checkpoint to [DIR: %s]", args.output_dir)


def train(model, dataloader_drone, dataloader_sat, optimizer, scheduler, epoch):
    global global_step
    
    epoch_accd = 0
    epoch_accs = 0
    epoch_loss = 0
    epoch_nceloss = 0
    epoch_deloss = 0

    pbar = tqdm(total=len(dataloader_sat))
    samplesize = len(dataloader_sat.dataset)
    for data_drone, data_sat in zip(dataloader_drone, dataloader_sat):
        inputs_d, inputs_ds, labels_d = data_drone
        inputs_s, inputs_sd, labels_s = data_sat

        now_batch_size, c, h, w = inputs_d.shape
        if now_batch_size < args.batchsize:  # skip the last batch
            continue

        inputs_d = Variable(inputs_d.cuda().detach())
        inputs_ds = Variable(inputs_ds.cuda().detach())
        labels_d = Variable(labels_d.cuda().detach())

        inputs_s = Variable(inputs_s.cuda().detach())
        inputs_sd = Variable(inputs_sd.cuda().detach())
        labels_s = Variable(labels_s.cuda().detach())

        optimizer.zero_grad()

        ######################### inference 1
        logit_d, embedding_d = model(inputs_d)
        logit_ds, embedding_ds,  = model(inputs_ds)
        logit_s, embedding_s  = model(inputs_s)
        logit_sd, embedding_sd  = model(inputs_sd)

        # LPN Loss    
        pred_d, loss_d = lpnloss(logit_d, labels_d)
        _, loss_ds = lpnloss(logit_ds, labels_d)
        pred_s, loss_s = lpnloss(logit_s, labels_s)
        _, loss_sd = lpnloss(logit_sd, labels_s)
        csl_loss = (loss_d + loss_ds + loss_s + loss_sd) / 2

        # infonce loss
        nce_loss = nceloss(embedding_d, embedding_ds, embedding_s, embedding_sd,
                        labels_d, labels_s)
            
        # decouple loss
        dwdr_loss_d, _, _ = decouple_loss(embedding_d, embedding_ds)
        dwdr_loss_s, _, _ = decouple_loss(embedding_s, embedding_sd)
        dwdr_loss = (dwdr_loss_d + dwdr_loss_s) / 2
            
        loss = csl_loss + nce_loss
        loss = loss * 0.9 + dwdr_loss * 0.1
            
        loss.backward()
        optimizer.first_step(zero_grad=True)

        ######################### inference 2
        logit_d, embedding_d = model(inputs_d)
        logit_ds, embedding_ds,  = model(inputs_ds)
        logit_s, embedding_s  = model(inputs_s)
        logit_sd, embedding_sd  = model(inputs_sd)

        # LPN Loss    
        pred_d, loss_d = lpnloss(logit_d, labels_d)
        _, loss_ds = lpnloss(logit_ds, labels_d)
        pred_s, loss_s = lpnloss(logit_s, labels_s)
        _, loss_sd = lpnloss(logit_sd, labels_s)
        csl_loss = (loss_d + loss_ds + loss_s + loss_sd) / 2

        # infonce loss
        nce_loss = nceloss(embedding_d, embedding_ds, embedding_s, embedding_sd,
                    labels_d, labels_s)
            
        # decouple loss
        dwdr_loss_d, _, _ = decouple_loss(embedding_d, embedding_ds)
        dwdr_loss_s, _, _ = decouple_loss(embedding_s, embedding_sd)
        dwdr_loss = (dwdr_loss_d + dwdr_loss_s) / 2
            
        loss = csl_loss + nce_loss
        loss = loss * 0.9 + dwdr_loss * 0.1
        
        loss.backward()
        optimizer.second_step(zero_grad=True)

        scheduler.step() 

        global_step += 1

        epoch_loss += loss.item() * now_batch_size
        epoch_nceloss += nce_loss.item() * now_batch_size
        epoch_deloss += dwdr_loss.item() * now_batch_size
        epoch_accd += float(torch.sum(pred_d == labels_d.data))
        epoch_accs += float(torch.sum(pred_s == labels_s.data))
        pbar.update(1)

    epoch_loss /= samplesize
    epoch_accd /= samplesize
    epoch_accs /= samplesize
    epoch_nceloss /= samplesize
    epoch_deloss /= samplesize

    tensorboard_writer.add_scalar('train/loss', epoch_loss, epoch)
    tensorboard_writer.add_scalar('train/Infonce_loss', epoch_nceloss, epoch)
    tensorboard_writer.add_scalar('train/deloss', epoch_deloss, epoch)
    tensorboard_writer.add_scalar('train/Satellite_Acc', epoch_accs, epoch)
    tensorboard_writer.add_scalar('train/Drone_Acc', epoch_accd, epoch)
    print(f'[Epoch ${epoch}]:\n\
            train/loss: {epoch_loss}\n\
            train/Infonce_loss: {epoch_nceloss}\n\
            train/deloss: {epoch_deloss}\n\
            train/Satellite_Acc: {epoch_accs}\n\
            train/Drone_Acc: {epoch_accd}')


def valid(model, dataloader_query, dataloader_gallery, epoch):
    
    def extract_feature(model, dataloader):
        def fliplr(img):
            '''flip horizontal'''
            inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
            img_flip = img.index_select(3, inv_idx)
            return img_flip
        
        features = torch.FloatTensor()
        print('valid inference ...')
        for data in tqdm(dataloader):
            img, label = data
            n, c, h, w = img.size()
                
            ff = torch.FloatTensor(n, 512, args.block).zero_().cuda()

            for i in range(2):
                if (i == 1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                logits, embeddings = model(input_img)
                ff += logits
        
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(args.block)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            features = torch.cat((features, ff.data.cpu()), 0)

        return features
    
    def get_id(img_path):
        labels = []
        paths = []
        for path, v in img_path:
            # print(path, v)
            folder_name = os.path.basename(os.path.dirname(path))
            labels.append(int(folder_name))
            paths.append(path)
        return labels, paths
    
    def compute_mAP(index, good_index, junk_index):
        ap = 0
        cmc = torch.IntTensor(len(index)).zero_()
        if good_index.size == 0:  # if empty
            cmc[0] = -1
            return ap, cmc

        # remove junk_index
        mask = np.in1d(index, junk_index, invert=True)
        index = index[mask]

        # find good_index index
        ngood = len(good_index)
        mask = np.in1d(index, good_index)
        rows_good = np.argwhere(mask == True)
        rows_good = rows_good.flatten()

        cmc[rows_good[0]:] = 1
        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) * 1.0 / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i * 1.0 / rows_good[i]
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2

        return ap, cmc
    
    def evaluate(qf, ql, gf, gl):
        query = qf.view(-1, 1)

        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        # predict index
        index = np.argsort(score)  # from small to large
        index = index[::-1]
    
        query_index = np.argwhere(gl == ql)
        good_index = query_index
        
        junk_index = np.argwhere(gl == -1)

        CMC_tmp = compute_mAP(index, good_index, junk_index)
        return CMC_tmp

    # feature extraction
    model_test = copy.deepcopy(model)
    if len(gpu_ids) > 1:
        for i in range(args.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model_test.module, cls_name)
            c.classifier = nn.Sequential()
    else:
        for i in range(args.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model_test, cls_name)
            c.classifier = nn.Sequential()

    model_test.eval()
    query_features = extract_feature(model_test, dataloader_query)
    gallery_features = extract_feature(model_test, dataloader_gallery)

    query_imgs = dataloader_query.dataset.imgs
    gallery_imgs = dataloader_gallery.dataset.imgs
    query_labels, query_paths = get_id(query_imgs)
    gallery_labels, gallery_paths = get_id(gallery_imgs)

    result = {
        'gallery_f': gallery_features.numpy(),
        'gallery_label': gallery_labels, 
        'gallery_path': gallery_paths,
        'query_f': query_features.numpy(), 
        'query_label': query_labels, 
        'query_path': query_paths
    }
    
    mid_result_path = os.path.join(args.output_dir, 'val_mid_result.mat')
    scipy.io.savemat(mid_result_path, result)

    # evaluate
    result = scipy.io.loadmat(mid_result_path)
    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    acc1 = CMC[0] * 100
    ap1 = ap / len(query_label) * 100

    tensorboard_writer.add_scalar('val/Recall@1', acc1, epoch + 1)
    tensorboard_writer.add_scalar('val/AP', ap1, epoch + 1)

    return acc1

def main():
    global global_step

    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    # data preparation
    dataloader_drone, dataloader_sat = get_train_dataloaders(args)
    dataloader_val_gallery_sat, _, _, dataloader_val_query_drone = get_val_dataloaders(args)
    
    loader_length = min(len(dataloader_drone), len(dataloader_sat))
    class_names = dataloader_sat.dataset.classes

    print(f'loader length: {loader_length}\n', 
          f'class number: {len(class_names)}\n') 

    # model initialization
    # model = getModel(args).to(args.device)
    model = CSWinTransv2_threeIn(
        class_num = len(class_names),
        droprate = args.droprate,
        decouple = 0,
        infonce = 1,
    )

    # ignore parameters
    if len(gpu_ids) == 1:
        model = model.cuda()
        ignored_params = list()
        for i in range(args.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        
        optim_params = [{'params': base_params, 'lr': 0.1 * args.learning_rate}]

        for i in range(args.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': args.learning_rate})
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
        
        ignored_params = list()
        for i in range(args.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model.module, cls_name)
            ignored_params += list(map(id, c.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        
        optim_params = [{'params': base_params, 'lr': 0.1 * args.learning_rate}]
        
        for i in range(args.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model.module, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': args.learning_rate})


    # optimizer
    optimizer = SAM(optim_params,
                    torch.optim.SGD, 
                    lr=args.learning_rate,
                    weight_decay=5e-4,
                    momentum=0.9,
                    nesterov=True)
    
    # optimizer = torch.optim.AdamW(model.parameters(),
    #                     lr=args.learning_rate,
    #                     weight_decay=args.weight_decay)
    
    # load pretrained model
    if args.resume:
        print(args.resume)
        state_dict = torch.load(os.path.join(args.resume, 'model_checkpoint.pth'), map_location='cpu')
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
        global_step = start_epoch * loader_length
        best_acc = state_dict['best_acc']
        best_epoch = -1
        print('model start from ' + str(start_epoch) + ' epoch')
        print('best acc: ' + str(best_acc))
    else:
        start_epoch = 0
        global_step, best_acc = 0, 0
        best_epoch = -1
    
    # scheduler 
    t_total = args.total_epoch * loader_length
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=0, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)


    for epoch in range(start_epoch, args.total_epoch):
        model.train()

        train(model, dataloader_drone, dataloader_sat, optimizer, scheduler, epoch)
        
        if epoch > 100 and epoch % 10 == 0:
            accuracy = valid(model, dataloader_val_query_drone, dataloader_val_gallery_sat, epoch)

            if best_acc < accuracy:
                save_model(args, model, optimizer, epoch, accuracy)
                best_acc = accuracy
                best_epoch = epoch
            print('BestRecall@1:%.2f, bestepoch:%.0f' % (best_acc, best_epoch))
    
    save_model(args, model, optimizer, 999, 999)
        
if __name__ == '__main__':
    main()


