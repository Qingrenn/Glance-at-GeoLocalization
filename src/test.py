import torch
import scipy
import numpy as np
from tqdm import tqdm

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

mid_result_path = '/home/qingren/qr_wkspace/GeoLocalization/output/val_mid_result.mat'
result = scipy.io.loadmat(mid_result_path)
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0

for i in tqdm(range(len(query_label))):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC
acc1 = CMC[0] * 100
ap1 = ap / len(query_label) * 100

print(acc1, ap1)

