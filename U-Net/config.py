import json


class ConfigParams:
    def __init__(self, fname: str= "config_mitotic_bb.json"):
        self.model_name = 'unet_model.pth.tar'
        self.train_image_dir = ''
        self.train_mask_dir = ''
        self.val_image_dir = ''
        self.val_mask_dir = ''
        self.test_image_dir = ''
        self.test_mask_dir = ''
        self.features = [64, 128, 256, 512]
        self.batch_size = 4
        self.image_size = [256, 256]
        self.lr = 5e-4
        self.epochs = 1000
        self.load_pretrained = False
        self.load_json(fname)

    def load_json(self, name: str = "config_mitotic_bb.json"):
        f = open(name)
        data = json.load(f)
        # print(data)
        self.train_image_dir = data['train_images']
        self.train_mask_dir = data['train_mask']
        self.val_image_dir = data['val_images']
        self.val_mask_dir = data['val_mask']
        self.test_image_dir = data['test_images']
        self.test_mask_dir = data['test_mask']
        self.features = data['features']
        self.batch_size = data['batch_size']
        self.image_size = data['image_size']
        self.lr = data['lr']
        self.epochs = data['n_epochs']
        self.model_name = data['model_name']
        self.load_pretrained = data['load_pretrained']
