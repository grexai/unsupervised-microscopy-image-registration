# Unet-torch

wandb login
weights and biases are implmented to track the losses
Usage
create a virtual enviroment and activate it
pip install -r requrements.txt

create your own config file using config.json template or use existing one


run the following script for image to image translation training.
```
python3 train.py "config_HeLa.json"

```


To inference  use
```
python3 test.py config_HeLa.json
```
