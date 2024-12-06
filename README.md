# Exo2EgoDVC 

Official implementation and data release for "Exo2EgoDVC: Dense Video Captioning of Egocentric Human Activities Using Web Instructional Videos (WACV 2025)". See the corresponding paper info below:
> Exo2EgoDVC: Dense Video Captioning of Egocentric Human Activities Using Web Instructional Videos \
> Takehiko Ohkawa, Takuma Yagi, Taichi Nishimura, Ryosuke Furuta, Atsushi Hashimoto, Yoshitaka Ushiku, and Yoichi Sato \
> IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025 \
> [[paper]](https://arxiv.org/abs/2311.16444) [[project]](https://tkhkaeio.github.io/projects/25-egodvc)

![](assets/teaser_exo2ego.jpg)

## Release notes
[Dec 6th, 2024]: Open repository


## Dependency
```
Python>=3.7
PyTorch>=1.5.1
```
Make a vitual conda environment following [[PDVC-GitHub]](https://github.com/ttengwang/PDVC.git). Our code is run on the ``PDVC`` enviornment.


## EgoYC2 Dataset
Our work proposes a newly captured egocentric dense video captioning dataset, dubbed EgoYC2.
Download from [[EgoYC2-download]](https://drive.google.com/drive/folders/1UIUktsdJ1MRGfQoFRq_RM8hHCNJmv9wz?usp=sharing) and place `ho_feats`, `crop_feats`, and `face_track` as the following structure.
```
data/
├── egoyc2
│   ├── captiondata
│   │   ├── egoyc2_eval_wacv25.json
│   │   ├── egoyc2_train_wacv25.json
│   │   └── para
│   │       └── para_egoyc2_eval_wacv25.json
│   ├── features
│   │   ├── ho_feats
│   │   │   └── resnet_bn
│   └── └── crop_feats
│           └── resnet_bn
└── yc2
    ├── captiondata
    │   ├── para
    │   │   └── para_yc2_eval_wacv25.json
    │   ├── yc2_eval_wacv25.json
    │   └── yc2_train_wacv25.json
    ├── face_track
    ├── features
    │   ├── download_yc2_tsn_features.sh
    │   └── resnet_bn
    └── vocabulary_youcook2.json
```
- Features `crop_feats` and `ho_feats` contain frame-wise features from cropped images showing the hand interactions and features from the cropped images and interacted hand-object regions, respectively.
- The YC2 data download (`yc2/features/resnet_bn`) should be referred to [[PDVC-GitHub]](https://github.com/ttengwang/PDVC.git).
- The data split of YC2 (``captiondata``) is not compartible with the original split to align with the split of EgoYC2. The split used in our experiments is named as `train/eval_wacv25`.

## Training and Validation
The running scripts are found under `scripts`.
For the pre-training on source data (YC2),  use `src_pt_yc2.sh` and `src_vipt_yc2.sh` as a naive baseline and our view-invariant pre-training, respectively.

For the fine-tuning on target data (EgoYC2), use `trg_ft_egoyc2.sh` and `trg_vift_yc2+egoyc2.sh` as a naive baseline and our view-invariant fine-tuning, respectively.
Please replace the path of the pre-trained checkpoint (`PRE_PATH`) according to the experiment conditions.

## Acknowledgement
The implementation of captioning models is based on [[PDVC-GitHub]](https://github.com/ttengwang/PDVC.git).

## Citation
Please cite the following article if our code helps you.
```
@inproceedings{ohkawa:wacv25,
  author      = {Takehiko Ohkawa and Takuma Yagi and　Taichi Nishimura and　Ryosuke Furuta and　Atsushi Hashimoto and　Yoshitaka Ushiku and　Yoichi Sato},
  title       = {{Exo2EgoDVC}: Dense Video Captioning of Egocentric Procedural Activities Using Web Instructional Videos},
  booktitle   = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year        = {2025},
}
```