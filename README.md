# Unsupervised Microscopy image registration

# Install 

## Python enviroment for data analysis

Install python 3.6 <br>

https://www.python.org/downloads/release/python-3614/

Create a virtualenviroment, and activate:

Run 

```
pip install -r requirments.txt

```

Matlab 2019b install version is required for annotation.
<br>
#### Note
Landmark_application under development for python annotation.
This application is designed to replace matlab's control point selection tool.
Check it out on landmark_feature branch


For /SuperPoint, /SuperCUT, /U-Net and /ContrastiveUnpairedTranslation

It is suggested to create a different virtial enviroment using the requirements in their folder.
These subrepositories has their own README.md

## Unaligned datset preprocessing for training

1. Data should be partitioned partition_paired_dataset.py. This will split the data  into train and test (80-20%  of the images).
2. With matlab scale_annotated.m script to generate warped images that will be on the same scale as the fixed images.
3. create_unaligned_datapairs.py will center crop the warped and the fixed images. 
4. The generated images can be forwared to Contrastive Unparied training.

## Aligned datset preprocessing for training

For evaluation and Supervised training 
1. Data should be partitioned partition_paired_dataset.py. If it is done before not necesarry to repeat.
2. With matlab align_annotated.m script to generate warped images.
3. align_annotated_script will align and the image pairs and remove black parts coming from warping.
4. The generated images can be forwared to U-Net training.
See /U-Net

## Annotation

Use code from matlab_scripts/imreg_annotation.m
There will be simple image reading procedures,
example of loading imagepairs into CP select tools,
example of saving the landmark annotations,
visualization codes to ovelay the aligned images for validation
<br>
<br>
Example landmark annotation:
<br>
```
% Moving image from the screening microscope
M = imread("./datasets/HeLA/registration/p1_wA1_t1_m123_c0_z0_l1_o0.png");

% Fixed image from Leica LMD6 63x
F = imread("./datasets/HeLA/lmd/p1_wA1_t1_m23_c1_z0_l1_o0_1.BMP");

cpselect(M,F)
```
<br>
<img src='images/annitation_example.png'>
<br>
<br>
File/export controlpoints to workspace

```
% estimate the transformation using the control points
tform = fitgeotrans(movingPoints,fixedPoints,'NonreflectiveSimilarity')
% warp
Jregistered = imwarp(M,tform,'OutputView',imreFd(size(F)));
% visualize the aligment
imshowpair(F,Jregistered, 'blend')
```

## Inference and Evaluation

/ContrastiveUnpariedTranslation folder Cut_{dataset} jupyter notebook training scripts contains test examples.
The output folders of the test script should be added to SuperPoint/superpoint/calc_sp_sift_corr_{dataset}.py
This script will try to align images, outputs result transformation into SuperPoint/superpoint/result/{experiment}
The output of the scale_annotated.m mat files contains the ground truth transformations in 2x3 format
eval_all_data.py will read result and ground truth transformations and compare them.

## Inference with SuperCUT

Registrationpipelineinference folder contains the SuperCUT packaged into a single repository.
Images can be forwarded with CLI to the pipeline which will return with a transformation matrix.
(see readme in the folder)


## Models


https://zenodo.org/records/10108327



## Citation

{placeholder}


This repository is based on these methods:

@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}

@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={224--236},
  year={2018}
}
