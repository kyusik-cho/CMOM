import copyreg
import os
import sys
from pathlib import Path
import os.path as osp
from tkinter import N
from turtle import Turtle
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from advent.utils.func import adjust_learning_rate
from advent.utils.func import loss_calc

from skimage.measure import label as sklabel
from cmom.utils.mix import mix_operation
from cmom.utils.prototype_utils import prototype_loss
from PIL import Image


def train_domain_adaptation(model, source_loader, target_loader, cfg):
    if cfg.TRAIN.DA_METHOD == 'CMOM':
        train_CMOM(model, source_loader, target_loader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")

def train_CMOM(model, source_loader, target_loader, cfg):
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True    

    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    
    # interpolate output segmaps
    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)                         
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)    
    interp_target_label = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='nearest')    

    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)


    # Init Prototype. borrowed from SIM (Differential Treatment for Stuff and Things)
    num_prototype = 50 
    num_ins = num_prototype * 10
    SOURCE = cfg.SOURCE
    if SOURCE == 'Viper':
        BG_LABEL = [0, 1, 2, 3, 6, 7, 8]
        FG_LABEL = [4, 5, 9, 10, 11, 12, 13, 14]
    elif SOURCE == 'SynthiaSeq':
        BG_LABEL = [0, 1, 2, 3, 7, 8]  # Road, Sidewalk, Building, Fence, Veg, Sky  
        FG_LABEL = [4, 5, 6, 9, 10, 11] # Pole, Light, Sign, Pedestrain, Bike, Car
    else:
        raise NotImplementedError(f"BG/FG {cfg.SOURCE}")

    src_cls_features = torch.zeros([len(BG_LABEL),num_prototype,2048], dtype=torch.float32).to(device)
    src_cls_ptr = np.zeros(len(BG_LABEL), dtype=np.uint64)
    src_ins_features = torch.zeros([len(FG_LABEL),num_ins,2048], dtype=torch.float32).to(device)
    src_ins_ptr = np.zeros(len(FG_LABEL), dtype=np.uint64)


    LAMBDA_CLS = cfg.TRAIN.lamda_FA
    LAMBDA_INS = cfg.TRAIN.lamda_FA

    #######################
    #### Training Iter ####
    #######################
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        # reset optimizers
        optimizer.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # Source
        _, source_batch = source_loader_iter.__next__()
        src_img_cf, src_label, src_img_kf, _, src_img_name, src_label_kf = source_batch
        if src_label.dim() == 4:
            src_label = src_label.squeeze(-1)
        file_name = src_img_name[0].split('/')[-1]
        if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')

        flow_int16_x10_src = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        src_flow = torch.from_numpy(flow_int16_x10_src / 10.0).permute(2, 0, 1).unsqueeze(0)

        # Target       
        _, target_batch = target_loader_iter.__next__()
        trg_img_cf, _, image_trg_kf, _, name = target_batch 
        file_name_trg = name[0].split('/')[-1]
        frame_trg = int(file_name_trg.replace('_leftImg8bit.png', '')[-6:])
        frame1_trg = frame_trg - 1
        flow_int16_x10_name_trg = file_name_trg.replace('leftImg8bit.png', str(frame1_trg).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)

        # for mix
        src2_img_cf, src2_img_kf, src2_flow = interp_target(src_img_cf.clone().detach()), interp_target(src_img_kf.clone().detach()), interp_target(src_flow.clone().detach())
        src2_label, src2_label_kf = interp_target_label(src_label.clone().detach().unsqueeze(0)).squeeze(0), interp_target_label(src_label_kf.clone().detach().unsqueeze(0)).squeeze(0)
        
        # get PL 
        label_name = ('%s/%s'%(cfg.TRAIN.pseudo_label_path, file_name_trg.replace('_leftImg8bit.png', '_pseudolabel.png')))  
        label_pseudo_ = get_labels(label_name, labels_size=(input_size_target[1], input_size_target[0]))
        
        label_prev_name = label_name.replace('%s_pseudolabel'%str(frame_trg).zfill(6), '%s_pseudolabel_prev'%str(frame1_trg).zfill(6))
        label_pseudo_prev = get_labels(label_prev_name, labels_size=(input_size_target[1], input_size_target[0])).squeeze(0)
                      

        ############################
        ### Source2 x Target Mix ###
        ############################     
        if cfg.TRAIN.MIX > 0.:   
            image_cf_mixed, image_kf_mixed, label_cf_mixed, label_kf_mixed, mixed_flow, classes = \
                mix_operation(trg_img_cf, src2_img_cf, image_trg_kf, src2_img_kf, label_pseudo_, src2_label,
                            src2_flow, trg_flow, cfg.TRAIN.MIX, label_kf_T = np.expand_dims(label_pseudo_prev, axis=0), label_kf_S = src2_label_kf)

        ######### Source-domain supervised training
        src_pred_aux, src_pred, src_pred_cf_aux, src_pred_cf, src_pred_cf_feature, src_pred_kf_aux, src_pred_kf, src_pred_kf_feature = \
            model(src_img_cf.cuda(device), src_img_kf.cuda(device), src_flow, device)
        src_pred_srcsize = interp_source(src_pred)
        loss_seg_src_main = loss_calc(src_pred_srcsize, src_label, device)
        if cfg.TRAIN.MULTI_LEVEL:
            src_pred_aux_srcsize = interp_source(src_pred_aux)
            loss_seg_src_aux = loss_calc(src_pred_aux_srcsize, src_label, device)
        else:
            loss_seg_src_aux = 0
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # Prototype generation.
        if cfg.TRAIN.lamda_FA > 0:
            pred_idx = torch.argmax(F.softmax(src_pred_cf), dim=1)
            right_label = F.interpolate(src_label.unsqueeze(0).float(), (pred_idx.size(1),pred_idx.size(2)), mode='nearest').squeeze(0).long().to(device)
            right_label[right_label!=pred_idx] = 255
            for ii in range(len(BG_LABEL)):
                cls_idx = BG_LABEL[ii]
                mask = right_label==cls_idx
                if torch.sum(mask) == 0:
                    continue
                feature = global_avg_pool(src_pred_cf_feature, mask.float())
                if cls_idx != torch.argmax(torch.squeeze(model.layer6(feature.float()).float())).item():
                    continue
                src_cls_features[ii,int(src_cls_ptr[ii]%num_prototype),:] = torch.squeeze(feature).clone().detach()
                src_cls_ptr[ii] += 1

            seg_ins = seg_label(right_label.squeeze(), FG_LABEL)
            for ii in range(len(FG_LABEL)):
                cls_idx = FG_LABEL[ii]
                segmask, pixelnum = seg_ins[ii]
                if len(pixelnum) == 0:
                    continue
                sortmax = np.argsort(pixelnum)[::-1]
                for i in range(min(10, len(sortmax))):
                    mask = segmask==(sortmax[i]+1)
                    feature = global_avg_pool(src_pred_cf_feature, mask.float())
                    if cls_idx != torch.argmax(torch.squeeze(model.layer6(feature.float()).float())).item():
                        continue
                    src_ins_features[ii,int(src_ins_ptr[ii]%num_ins),:] = torch.squeeze(feature).clone().detach()
                    src_ins_ptr[ii] += 1                

        ########################################
        ### Self Training with mixed samples ###
        ########################################
        if cfg.TRAIN.MIX > 0.:
            image_cf_mixed = torch.from_numpy(image_cf_mixed).unsqueeze(0)
            image_kf_mixed = torch.from_numpy(image_kf_mixed).unsqueeze(0)
            label_cf_mixed = torch.from_numpy(label_cf_mixed)
            label_kf_mixed = torch.from_numpy(label_kf_mixed)
            
            mix_pred_aux, mix_pred, mix_pred_cf_aux, mix_pred_cf, mix_pred_cf_feature, mix_pred_kf_aux, mix_pred_kf, mix_pred_kf_feature =\
                model(image_cf_mixed.cuda(device), image_kf_mixed.cuda(device),  torch.from_numpy(mixed_flow).unsqueeze(0), device)    
            
            ### Use Mixed Samples as target
            trg_pred_aux, trg_pred, trg_pred_cf_aux, trg_pred_cf, trg_pred_kf_aux, trg_pred_kf = \
                mix_pred_aux, mix_pred, mix_pred_cf_aux, mix_pred_cf, mix_pred_kf_aux, mix_pred_kf
            trg_pred_cf_feature, trg_pred_kf_feature = mix_pred_cf_feature, mix_pred_kf_feature
        else:
            trg_pred_aux, trg_pred, trg_pred_cf_aux, trg_pred_cf, trg_pred_cf_feature, trg_pred_kf_aux, trg_pred_kf, trg_pred_kf_feature =\
                model(trg_img_cf.cuda(device), image_trg_kf.cuda(device), trg_flow, device)
        loss = 0

        ### Warp     
        trg_prob_cf = F.softmax(trg_pred_cf)
        trg_prob_cf_aux = F.softmax(trg_pred_cf_aux)
        trg_prob_kf = F.softmax(trg_pred_kf).cpu().numpy()
        trg_prob_aux_kf = F.softmax(trg_pred_kf_aux).cpu().numpy()
        interp_flow2trg = nn.Upsample(size=(trg_prob_cf.shape[-2], trg_prob_cf.shape[-1]), mode='bilinear', align_corners=True)
        interp_flow2trg_ratio = trg_prob_cf.shape[-2] / trg_flow.shape[-2]
        trg_flow_interp = interp_flow2trg(trg_flow) * interp_flow2trg_ratio
        trg_flow_interp = trg_flow_interp.cpu().numpy()
        trg_prob_propagated = np.zeros(trg_prob_cf.shape)
        trg_prob_propagated_aux = np.zeros(trg_prob_cf_aux.shape) #
        for x in range(trg_prob_kf.shape[-1]):
            for y in range(trg_prob_kf.shape[-2]):
                x_flow = int(round(x - trg_flow_interp[:, 0, y, x][0]))
                y_flow = int(round(y - trg_flow_interp[:, 1, y, x][0]))
                if x_flow >= 0 and x_flow < trg_prob_kf.shape[-1] and y_flow >= 0 and y_flow < trg_prob_kf.shape[-2]:
                    trg_prob_propagated[:, :, y_flow, x_flow] = trg_prob_kf[:, :, y, x]
                    trg_prob_propagated_aux[:, :, y_flow, x_flow] = trg_prob_aux_kf[:, :, y, x]
        trg_prob_propagated = torch.from_numpy(trg_prob_propagated)
        trg_prob_propagated_aux = torch.from_numpy(trg_prob_propagated_aux)     
       

        ########################################
        ### Self Training with mixed samples ###
        ########################################
        trg_pred_trgsize = interp_target(trg_pred)
        if cfg.TRAIN.MIX > 0.:
            loss_seg_mix_main = loss_calc(trg_pred_trgsize, label_cf_mixed, device)
        else:    
            loss_seg_mix_main = loss_calc(trg_pred_trgsize, torch.from_numpy(label_pseudo_),device)
        if cfg.TRAIN.MULTI_LEVEL: 
            trg_pred_aux_trgsize = interp_target(trg_pred_aux)
            trg_pred_kf_aux_trgsize = interp_target(trg_pred_kf_aux)
            if cfg.TRAIN.MIX > 0.:
                loss_seg_mix_aux = loss_calc(trg_pred_aux_trgsize, label_cf_mixed,device)
            else:
                loss_seg_mix_aux = loss_calc(trg_pred_aux_trgsize, torch.from_numpy(label_pseudo_),device)
        else:
            loss_seg_mix_aux = 0
        loss = loss + (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_mix_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_mix_aux)


        ###########################
        ### Prototype Alignment ###
        ###########################
        # ref: https://github.com/SHI-Labs/Unsupervised-Domain-Adaptation-with-Differential-Treatment/blob/master/train_sim.py
        loss_cls = torch.zeros(1).to(device)
        loss_ins = torch.zeros(1).to(device)
        if cfg.TRAIN.lamda_FA > 0:
            loss_cls, loss_ins = prototype_loss(trg_prob_cf, trg_prob_propagated, trg_pred_cf_feature, model, i_iter, device, 
                    src_cls_ptr, src_cls_features, src_ins_ptr, src_ins_features, num_prototype, num_ins, BG_LABEL, FG_LABEL)            
            loss = loss + (LAMBDA_CLS * loss_cls + LAMBDA_INS * loss_ins)

        loss.backward()
        loss_cls = LAMBDA_CLS * loss_cls.item()
        loss_ins = LAMBDA_INS * loss_ins.item()        

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()        

        current_losses = {'loss_src_aux': loss_seg_src_aux,
                          'loss_src': loss_seg_src_main,
                          'loss_seg_mix_main:': loss_seg_mix_main,
                          'loss_seg_mix_aux': loss_seg_mix_aux,
                          'loss_cls': loss_cls,
                          'loss_ins': loss_ins,}
        print_losses(current_losses, i_iter)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)

            os.makedirs(snapshot_dir, exist_ok=True)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

def weighted_l1_loss(input, target, weights):
    loss = weights * torch.abs(input - target)
    loss = torch.mean(loss)
    return loss

def get_labels(file, labels_size):
    return to_numpy(F.interpolate(torch.from_numpy(_load_img(file, labels_size, Image.NEAREST, rgb=False)).unsqueeze(0).unsqueeze(0).float(),
              labels_size, mode='nearest').squeeze(0).long())

def _load_img(file, labels_size, _, rgb=False):
    img = Image.open(file)
    return np.asarray(img, np.uint8)

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

# From SIM
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