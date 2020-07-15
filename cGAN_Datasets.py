from PIL import Image

import torch
import torch.utils.data as data

from torchvision import transforms
from glob import glob


def make_horse2zebra_datapath_list():
    """学習用のウマからシマウマに変えるデータセットパスを作る"""

    img_extend = '.jpg'

    train_img_list_A = glob('./data/horse2zebra/trainA/*' + img_extend)
    train_img_list_B = glob('./data/horse2zebra/trainB/*' + img_extend)

    return train_img_list_A, train_img_list_B


class ImageTransform():
    """画像の前処理クラス"""

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class GAN_Img_Dataset(data.Dataset):
    """画像のDatasetクラス"""

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img)

        return img_transformed
