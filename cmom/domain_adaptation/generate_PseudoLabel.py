import copyreg
import os
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from tqdm import tqdm
from PIL import Image

def save_pseudo(model_pseudo, target_loader, cfg):
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    cudnn.enabled = True    

    target_loader_iter = enumerate(target_loader)    

    alpha=0.2
    gamma=8
    PL_save_dir = 'saved_pseudo_label'
    os.makedirs(PL_save_dir, exist_ok=True)

    cls_thresh = np.ones(num_classes)*0.9
    cls_thresh_prev = np.ones(num_classes)*0.9
    
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        print('\n\n# Generating Pseudo Label ')
        _, target_batch = target_loader_iter.__next__()
        trg_img_cf, trg_label_cf, image_trg_kf, _, name = target_batch
        file_name_trg = name[0].split('/')[-1]
        frame = int(file_name_trg.replace('_leftImg8bit.png', '')[-6:])
        frame1 = frame - 1
        flow_int16_x10_name_trg = file_name_trg.replace('leftImg8bit.png', str(frame1).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)

        file_name_prev = file_name_trg.replace('%s_leftImg8bit.png'%str(frame).zfill(6), '%s_pseudolabel.png'%(str(frame1).zfill(6)))
        save_name_trg = file_name_trg.replace('%s_leftImg8bit.png'%str(frame).zfill(6), '%s_pseudolabel.png'%(str(frame).zfill(6))).split('.')[0]
        save_name_prev = file_name_prev.split('.')[0]
        print(save_name_trg)
        print(save_name_prev)

        model_pseudo.eval()
        model_pseudo.to(device)
        trg_pred_aux, trg_pred, trg_pred_cf_aux, trg_pred_cf, _, trg_pred_kf_aux, trg_pred_kf, _, trg_prev_pred, trg_prev_pred_aux = \
                    model_pseudo(trg_img_cf.cuda(device), image_trg_kf.cuda(device), trg_flow, device, need_prev_pred = True)

        # Pseudo label threshold generation. borrowed from IAST
        # https://github.com/Raykoooo/IAST/blob/master/code/sseg/workflow/sl_trainer.py        
        interp_to_trg = nn.Upsample(size=(image_trg_kf.shape[-2], image_trg_kf.shape[-1]), mode='bilinear', align_corners=True)

        trg_pred = interp_to_trg(trg_pred)
        trg_prev_pred = interp_to_trg(trg_prev_pred)
        trg_pred_cf = interp_to_trg(trg_pred_cf)
        trg_pred_kf = interp_to_trg(trg_pred_kf) 

        logits_ = nn.Softmax(dim=1)(trg_pred)
        logits_prev = nn.Softmax(dim=1)(trg_prev_pred)
        
        
        # prev
        # cls_thresh_prev = np.copy(cls_thresh)
        max_pred_ = logits_.max(dim=1)
        label_pred_ = max_pred_[1].data.cpu().numpy() 
        logits_pred_ = max_pred_[0].data.cpu().numpy()
        logits_cls_dict = {c: [cls_thresh[c]] for c in range(num_classes)}

        # for prev
        max_pred_prev = logits_prev.max(dim=1)
        label_pred_prev = max_pred_prev[1].data.cpu().numpy()
        logits_pred_prev = max_pred_prev[0].data.cpu().numpy()
        logits_cls_dict_prev = {c: [cls_thresh[c]] for c in range(num_classes)}
        for cls in range(num_classes):
                logits_cls_dict[cls].extend(logits_pred_[label_pred_ == cls].astype(np.float16))
                logits_cls_dict_prev[cls].extend(logits_pred_prev[label_pred_prev == cls].astype(np.float16))


        tmp_cls_thresh_prev = ias_thresh(logits_cls_dict_prev, alpha=alpha, cfg=cfg, w=cls_thresh, gamma=gamma, num_classes=num_classes)
        beta = 0.9
        cls_thresh_prev = beta*cls_thresh_prev + (1-beta)*tmp_cls_thresh_prev 
        cls_thresh_prev[cls_thresh_prev>=0.999] = 0.999

        # current
        tmp_cls_thresh = ias_thresh(logits_cls_dict, alpha=alpha, cfg=cfg, w=cls_thresh, gamma=gamma, num_classes=num_classes)
        beta = 0.9
        cls_thresh = beta*cls_thresh + (1-beta)*tmp_cls_thresh 
        cls_thresh[cls_thresh>=0.999] = 0.999


        np_logits_ = logits_.data.cpu().numpy()
        logit = np_logits_[0].transpose(1,2,0)
        label_pseudo_ = np.argmax(logit, axis=2)
        pred_image_current = np.argmax(logit, axis=2)
        logit_amax = np.amax(logit, axis=2)
        label_cls_thresh = np.apply_along_axis(lambda x: [cls_thresh[e] for e in x], 1, label_pseudo_)
        ignore_index = logit_amax < label_cls_thresh
        label_pseudo_[ignore_index] = 255

        # prev
        np_logits_prev = logits_prev.data.cpu().numpy()
        logit = np_logits_prev[0].transpose(1,2,0)
        label_pseudo_prev = np.argmax(logit, axis=2)
        pred_image_prev = np.argmax(logit, axis=2)
        logit_amax_prev = np.amax(logit, axis=2)
        label_cls_thresh = np.apply_along_axis(lambda x: [cls_thresh_prev[e] for e in x], 1, label_pseudo_prev)
        ignore_index = logit_amax_prev < label_cls_thresh
        label_pseudo_prev[ignore_index] = 255
            
        label_pseudo_ = np.expand_dims(label_pseudo_, axis=0) # .unsqueeze(0)
        
        logit_amax = logit_amax # pred map
        logit_amax_prev = logit_amax_prev
        
        save_name_trg = save_name_trg 
        save_name_prev = save_name_prev 
        pseudolabel_asimage = Image.fromarray(label_pseudo_.squeeze(0).astype(np.uint8)) 
        pseudolabel_asimage.save("%s/%s.png"%(PL_save_dir, save_name_trg))
        pseudolabel_prev_asimage = Image.fromarray(label_pseudo_prev.astype(np.uint8))
        pseudolabel_prev_asimage.save("%s/%s_prev.png"%(PL_save_dir, save_name_prev))        
        
        if i_iter == 9000:
            exit()
        sys.stdout.flush()

def print_ent_savingPL(image, save_name = "tmp", interp = None, norm = True):    
    channels, h, w = image.shape
    if interp:
        image = interp(image.unsqueeze(0)).cpu().data[0].numpy()
    else:
        image = (image.unsqueeze(0)).cpu().data[0].numpy()
    if norm:
        norm = np.max(image)
    else:
        norm = 1.
    img = image / norm * 255

    # Per-Channel result  
    output = img.transpose(1, 2, 0)  
    for c in range(channels):
      img_c = Image.fromarray(np.asarray(output[:,:,c], dtype=np.uint8).astype(np.uint8))
      img_c.save(save_name)
    return norm

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

# ref: https://github.com/Raykoooo/IAST/blob/master/code/sseg/workflow/sl_trainer.py
def ias_thresh(conf_dict, cfg, alpha, w=None, gamma=1.0, num_classes = 15):
    if w is None:
        w = np.ones(num_classes)
    # threshold 
    cls_thresh = np.ones(num_classes,dtype = np.float32)
    for idx_cls in np.arange(0, num_classes):
        if conf_dict[idx_cls] != None:
            arr = np.array(conf_dict[idx_cls])
            cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
    return cls_thresh
