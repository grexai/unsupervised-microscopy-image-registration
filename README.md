#Unsupervised Microscopy image registration

##Unaligned datset preprocessing for training
  
1. partition dataset with: partition_paired_dataset, this will split paired data into train and test folders

2 run create unaligned_datapairs.py This will generate training images for unsupervised CUT training and images will be on the same scale.
The generated images can used for training CUT model

For evaluation and Supervised training 
1. Data should be partitioned partition_paired_dataset. If it is done before not necesarry to repeat.
2. With matlab scale annotated script will generate warped images using the annotations.
3. align_annotated_script will center crop the warped and the fixed images. 
4. The generated images can be forwared to U-Net training.
See U-Net.

##Annotation

Use code from matlab_scripts/imreg_annotation.m
There will be simple image reading procedures,
example of loading imagepairs into CP select tools,
example of saving the landmark annotations,
visualization codes to ovelay the aligned images for validation

##Inference with SuperCUT

Registrationpipelineinference folder contains the SuperCUT packaged into a single repository.
Images can be forwarded with CLI to the pipeline which will return with a transformation matrix.
(see readme in the folder)

##Models
Models for CUT, SuperPoint, and U-Net can be downloaded at:

https://zenodo.org/records/10108327


##Citation
If you use this repository please cite:

{placeholder}


This repository is based on these methods,

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