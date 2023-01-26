import torch 
import torch.nn.functional as F
import numpy as np
from skimage.measure import label as sklabel

L1_loss = torch.nn.L1Loss(reduction='none')

def prototype_loss(trg_prob_cf, trg_prob_propagated, trg_pred_cf_feature, model, i_iter, device, 
    src_cls_ptr, src_cls_features, src_ins_ptr, src_ins_features, num_prototype, num_ins, BG_LABEL, FG_LABEL):
    loss_cls = torch.zeros(1).to(device)
    loss_ins = torch.zeros(1).to(device)

    pred_target_idx = torch.argmax(trg_prob_cf, dim=1)
    pred_target_kf_idx = torch.argmax(trg_prob_propagated, dim=1).to(device)
    _pred_diff = pred_target_kf_idx!=pred_target_idx
    pred_target_idx[_pred_diff] = 255

    if i_iter > 0:
        for ii in range(len(BG_LABEL)):
            cls_idx = BG_LABEL[ii]
            if src_cls_ptr[ii] / num_prototype <= 1:
                continue

            mask = pred_target_idx==cls_idx
            feature = global_avg_pool(trg_pred_cf_feature, mask.float())
            if cls_idx != torch.argmax(torch.squeeze(model.layer6(feature.float()).float())).item():
                continue
            ext_feature = feature.squeeze().expand(num_prototype, 2048)
            loss_cls += torch.min(torch.sum(L1_loss(ext_feature, src_cls_features[ii,:,:]),dim=1) / 2048.)

        seg_ins = seg_label(pred_target_idx.squeeze(), FG_LABEL)
        for ii in range(len(FG_LABEL)):
            cls_idx = FG_LABEL[ii]
            if src_ins_ptr[ii] / num_ins <= 1:
                continue
            segmask, pixelnum = seg_ins[ii]
            if len(pixelnum) == 0:
                continue
            sortmax = np.argsort(pixelnum)[::-1]
            for i in range(min(10, len(sortmax))):
                mask = segmask==(sortmax[i]+1)
                feature = global_avg_pool(trg_pred_cf_feature, mask.float())
                feature = feature.squeeze().expand(num_ins, 2048)
                loss_ins += torch.min(torch.sum(L1_loss(feature, src_ins_features[ii,:,:]),dim=1) / 2048.) / min(10, len(sortmax))
    return loss_cls, loss_ins

def global_avg_pool(inputs, weight):
    b,c,h,w = inputs.shape[-4], inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]
    weight_new = weight.detach().clone()
    weight_sum = torch.sum(weight_new)
    weight_new = weight_new.view(h,w)
    weight_new = weight_new.expand(b,c,h,w)
    weight_sum = max(weight_sum, 1e-12)
    return torch.sum(inputs*weight_new,dim=(-1,-2),keepdim=True) / weight_sum

def seg_label(label, FG_LABEL):
    segs = []
    for fg in FG_LABEL:
        mask = label==fg
        if torch.sum(mask)>0:
            masknp = mask.cpu().numpy().astype(int)
            seg, forenum = sklabel(masknp, background=0, return_num=True, connectivity=2)
            seg = torch.LongTensor(seg).cuda()
            pixelnum = np.zeros(forenum, dtype=int)
            for i in range(forenum):
                pixelnum[i] = torch.sum(seg==(i+1)).item()
            segs.append([seg, pixelnum])
        else:
            segs.append([mask.long(), np.zeros(0)])
    return segs
