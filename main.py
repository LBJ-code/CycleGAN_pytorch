# cell[1]
import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

from DCGAN_Img_Dataset import GAN_Img_Dataset, make_datapath_list, ImageTransform
from DCGAN_nets import Discriminator, Generator

# cell[2]
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# ネットワークの初期化
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2DとConv2DTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G = Generator(z_dim=20, image_size=64)
D = Discriminator(z_dims=20, image_size=64)
G.apply(weight_init)
D.apply(weight_init)

print("ネットワークの初期化完了")


# モデルを学習させる関数の作成
def train_model(G, D, dataloader, num_epochs):

    # GPUが使えるのかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス : ", device)

    # 最適化手法の設定
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, (beta1, beta2))
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, (beta1, beta2))

    # 誤差関数の定義
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # パラメータをハードコーディング
    z_dim = 20
    mini_batch_size = 64

    # ネットワークをGPUに
    G.to(device)
    D.to(device)

    G.train()
    D.train() # 各ネットワークを訓練モードに

    # ネットワークが静的であれば高速化
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print('------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('------------')
        print('(train)')

        # データローダからminibatchずる取り出すループ
        for imgs in dataloader:

            # ----------
            # 1. Discriminatorの学習
            # ----------
            # ミニバッチサイズが1だと，バッチノーマライゼーションでエラーになるらしい
            if imgs.size()[0] == 1:
                continue

            # GPUが使えるならGPUにデータを送る
            imgs = imgs.to(device)

            # 正解ラベルと偽ラベルを作成
            # epochの最後のイテレーションはミニバッチ数が少なくなる
            mini_batch_size = imgs.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            # 真の画像を判定
            d_out_real = D(imgs)

            # 偽の画像を作成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # 誤差を計算
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # ----------
            # 2. Generatorの学習
            # ----------
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            # 誤差を計算
            g_loss = criterion(d_out_fake.view(-1), label_real)

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # ----------
            # 3. 記録
            # ----------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        # epochのphase毎のlossと正解率
        t_epoch_finish = time.time()
        print('----------')
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f}'.format(
            epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size
        ))
        print('timer: {:.4f} sec'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    return G, D


num_epoch = 200
train_img_list = make_datapath_list()
mean = (0.5,)
std = (0.5,)
train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
G_update, D_update = train_model(G, D, dataloader=train_dataloader, num_epochs=num_epoch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 入力の乱数生成
batch_size = 8
z_dim = 20
fixed_z = torch.randn(batch_size, z_dim)
fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

# 画像生成
fake_images = G_update(fixed_z.to(device))

# 訓練データ
batch_iterator = iter(train_dataloader) # イテレータに変換
imgs = next(batch_iterator)

# 出力
import matplotlib.pyplot as plt
for i in range(0, 5):
    # 上段に訓練データを
    plt.imshow(imgs[i][0].cpu().detach().numpy(), 'gray')
    plt.show()

    # 下段に生成データを表示する
    plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')
    plt.show()
