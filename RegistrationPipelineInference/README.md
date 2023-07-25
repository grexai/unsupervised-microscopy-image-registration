# Readme for Unsupervised image registration inference

This repository is for multimodal microscopy image registration




<details>
<summary>Install instructions </summary>

## Setup a python enviroment

Create a virutal enviroment with python version 3.6

Activate the virtualenv, and  run 

```
pip install  -r requirements.txt
```

## Download models
Download models for cut and SuperPoint, and place them into "./models/cut" and ".models/sp/"
 respectively

## Images

Use the example images in the Images/A and Images/B folder

or download the full datasets from
https://zenodo.org/record/8162985
and preprocess them

</details>

<details>
<summary>Run</summary>


## Run

Start the pipeline with arguments:
Cut model path: where latest_net_G.pb is located
SuperPoint model path, where saved_model.pb is located
Image A path: in the article referred as modality 1
Image B path: in the article reffered as modality 2



### Example

```

python3 run_pipeline.py "./models/cut/cut_unaligned_resize/" "./models/sp/sp_v6/" "./Images/A/p1_wA1_t1_m9_c1_z0_l1_o0_1.png" "Images/B/p1_wA1_t1_m9_c1_z0_l1_o0_1.png"

```

</details>