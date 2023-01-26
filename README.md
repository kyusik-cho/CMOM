# Domain-Adaptive-Video-Semantic-Segmentation-via-Cross-Domain-Moving-Object-Mixing
Code for **&lt;Domain Adaptive Video Semantic Segmentation via Cross-Domain Moving Object Mixing>** in WACV 2023    
[[paper]](https://openaccess.thecvf.com/content/WACV2023/html/Cho_Domain_Adaptive_Video_Semantic_Segmentation_via_Cross-Domain_Moving_Object_Mixing_WACV_2023_paper.html) [[demo]](https://www.youtube.com/watch?v=xrfe21mNQh0)

## Prerequisites
### Installation:
1. Conda enviroment
```bash
conda create -n CMOM python=3.6
conda activate CMOM
conda install -c menpo opencv
pip install kornia
pip install importlib-metadata
```

2. Clone [ADVENT](https://github.com/valeoai/ADVENT)
```bash
git clone https://github.com/valeoai/ADVENT.git
pip install -e ./ADVENT
```

3. Clone the repo
```bash
git clone https://github.com/kyusik-cho/CMOM.git
pip install -e ./CMOM
```

### Data preparation:
Download [Cityscapes](https://www.cityscapes-dataset.com/), [VIPER](https://playing-for-benchmarks.org/download/), [SYNTHIA-Seq](http://synthia-dataset.cvc.uab.cat/SYNTHIA_SEQS/SYNTHIA-SEQS-04-DAWN.rar).    
Ensure the file structure is as follows.

- Cityscapes-Seq
```
<data_dir>/Cityscapes/
<data_dir>/Cityscapes/leftImg8bit_sequence
<data_dir>/Cityscapes/gtFine
```

- VIPER
```
<data_dir>/Viper/
<data_dir>/Viper/train/img
<data_dir>/Viper/train/cls
```

- SYNTHIA-Seq
```
<data_dir>/SynthiaSeq/
<data_dir>/SynthiaSeq/SEQS-04-DAWN
```

### Optical Flow Estimation:
We followed [DA-VSN](https://github.com/Dayan-Guan/DA-VSN) to get optical flow.    
Please follow their policy to get estimated optical flow.

### Pseudo labels
Download the pseudo labels [here](https://drive.google.com/drive/folders/1pomtz6zUJwmj5Lhrjyy1l-K6vYhVfCaM?usp=sharing) and put them under `<root_dir>/cmom`.    
Or run `make_pseudolabel.py` with [DA-VSN pretrained model](https://github.com/Dayan-Guan/DA-VSN).

### Pre-trained model:
Download the [pre-trained models](https://drive.google.com/drive/folders/1BepeA09R9a5M2qKnCtiVCy_4dela68JV?usp=sharing) and put them under `<root_dir>/pretrained_models`.    
When training a model, you can start with either [DA-VSN pretrained model](https://github.com/Dayan-Guan/DA-VSN) or DeepLab ImageNet pretrained models.


## Train
```bash
python train.py --cfg configs/cmom_viper2city.yml --tensorboard 
python train.py --cfg configs/cmom_syn2city.yml --tensorboard 
```


## Test
```bash
python test.py --cfg configs/cmom_viper2city.yml
python test.py --cfg configs/cmom_syn2city.yml
```


## Acknowledgement
This code is based on the following open-source projects. 
- [ADVENT](https://github.com/valeoai/ADVENT)
- [DA-VSN](https://github.com/Dayan-Guan/DA-VSN)
- [DACS](https://github.com/vikolss/DACS)
- [SIM](https://github.com/SHI-Labs/Unsupervised-Domain-Adaptation-with-Differential-Treatment)
 
