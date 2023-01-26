import numpy as np
import torch
from torch import nn

from cmom.utils.DACS_utils import transformmasks
from cmom.utils.DACS_utils import transformsgpu

def mix_operation(image_cf_T, image_cf_S, image_kf_T, image_kf_S, label_cf_T, label_cf_S, Optical_flow_S, Optical_flow_T, mix_ratio, label_kf_T = None, label_kf_S = None):
    classes = torch.unique(torch.Tensor(label_cf_S)) 
    ignore_label = [255]
    classes=torch.tensor([x for x in classes if x not in ignore_label])
 
    nclasses = classes.shape[0]
    classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses*mix_ratio),replace=False)).long()])#.cuda()

    MixMask_cf = transformmasks.generate_class_mask(torch.Tensor(label_cf_S.squeeze(0)), classes).unsqueeze(0)#.cuda()
    MixMask_kf = transformmasks.generate_class_mask(torch.Tensor(label_kf_S.squeeze(0)), classes).unsqueeze(0)#.cuda() 

    strong_parameters = {"Mix": MixMask_cf}
    image_cf_mixed, _ = strongTransform(strong_parameters, data = torch.cat((image_cf_S.clone().detach(),image_cf_T.clone().detach())))
    _, label_cf_mixed = strongTransform(strong_parameters, target = torch.cat((label_cf_S.clone().detach().unsqueeze(0),torch.Tensor(label_cf_T.copy()).unsqueeze(0))))

    strong_parameters["Mix"] = MixMask_kf
    image_kf_mixed, _ = strongTransform(strong_parameters, data = torch.cat((image_kf_S.clone().detach(),image_kf_T.clone().detach())))
    _, label_kf_mixed = strongTransform(strong_parameters, target = torch.cat((label_kf_S.clone().detach().unsqueeze(0),torch.Tensor(label_kf_T.copy()).unsqueeze(0))))

    interp_flow2trg = nn.Upsample(size=(label_cf_S.shape[-2], label_cf_S.shape[-1]), mode='bilinear', align_corners=True)
    interp_flow2trg_ratio = label_cf_S.shape[-2] / Optical_flow_S.shape[-2]
    Optical_flow_S = interp_flow2trg(Optical_flow_S) * interp_flow2trg_ratio
    Optical_flow_T = interp_flow2trg(Optical_flow_T) * interp_flow2trg_ratio
    mixed_flow, _ = strongTransform(strong_parameters, data = torch.cat((Optical_flow_S.clone().detach(),Optical_flow_T.clone().detach())))

    image_cf_mixed = image_cf_mixed.squeeze(0).numpy()
    image_kf_mixed = image_kf_mixed.squeeze(0).numpy()
    label_cf_mixed = label_cf_mixed.squeeze(0).numpy()
    label_kf_mixed = label_kf_mixed.squeeze(0).numpy()
    mixed_flow = mixed_flow.squeeze(0).numpy()

    return image_cf_mixed, image_kf_mixed, label_cf_mixed, label_kf_mixed, mixed_flow, classes

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    return data, target
