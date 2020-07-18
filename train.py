# 現状ではcycleganの全結合層が定数を入れないとバグる

import random
import time
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import io as skio

import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision

from cGAN_Datasets import GAN_Img_Dataset, make_horse2zebra_datapath_list, ImageTransform
from cGAN_nets import NLayerDiscriminator, ResnetGenerator

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

G_horse2zebra = ResnetGenerator(input_nc=3, output_fc=3)
G_zebra2horse = ResnetGenerator(input_nc=3, output_fc=3)
D_for_horse = NLayerDiscriminator(input_nc=3, n_layers=5)
D_for_zebra = NLayerDiscriminator(input_nc=3, n_layers=5)

def train_model(G_horse2zebra, G_zebra2horse, D_for_horse, D_for_zebra, dataloader_horse, dataloader_zebra, num_epochs,
                train_horse_img_list, lambda_for_cycLoss=0.5, save_freq=100):

    # GPUが使えるか確認
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("使用デバイス : ", device)

    # 最適化手法の設定
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_horse2zebra_optimizer = torch.optim.Adam(G_horse2zebra.parameters(), g_lr, (beta1, beta2))
    g_zebra2horse_optimizer = torch.optim.Adam(G_zebra2horse.parameters(), g_lr, (beta1, beta2))
    d_horse_optimizer = torch.optim.Adam(D_for_horse.parameters(), d_lr, (beta1, beta2))
    d_zebra_optimizer = torch.optim.Adam(D_for_zebra.parameters(), d_lr, (beta1, beta2))

    # パラメータをハードコーディング
    h2z_mini_batch_size = 4
    z2h_mini_batch_size = 4

    # ネットワークをGPUに
    G_horse2zebra = G_horse2zebra.to(device)
    G_zebra2horse = G_zebra2horse.to(device)
    D_for_horse = D_for_horse.to(device)
    D_for_zebra = D_for_zebra.to(device)

    G_horse2zebra.train()
    G_zebra2horse.train()
    D_for_horse.train()
    D_for_zebra.train()

    # ネットワークがある程度固定であれば高速化
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_horse_imgs = len(dataloader_horse.dataset)
    horse_batch_size = dataloader_horse.batch_size
    num_zebra_imgs = len(dataloader_zebra.dataset)
    zebra_batch_size = dataloader_zebra.batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        epoch_g_h2z_loss = 0.0
        epoch_g_z2h_loss = 0.0
        epoch_d_h_loss = 0.0
        epoch_d_z_loss = 0.0
        epoch_reconst_loss = 0.0

        print('----------')
        print('Epoch {} / {}'.format(epoch, num_epochs))
        print('----------')
        print('(train)')

        # データローダーからminibatchずつ取り出すループ
        for horse_imgs, zebra_imgs in zip(dataloader_horse, dataloader_zebra):

            # ----------
            # 1. Discriminatorの学習
            # -----------
            # ミニバッチサイズが1だと，バッチノーマライゼーションでエラーになるのでさける
            if horse_imgs.size()[0] == 1 or zebra_imgs.size()[0] == 1:
                continue

            # GPUが使えるなら
            horse_imgs = horse_imgs.to(device)
            zebra_imgs = zebra_imgs.to(device)

            # 正解ラベルと偽ラベルを作成
            # epochの最後のイテレーションはミニバッチ数が少なくなる
            h2z_mini_batch_size = horse_imgs.size()[0]
            z2h_mini_batch_size = zebra_imgs.size()[0]

            # 真の画像を判定
            d_out_real_horse = D_for_horse(horse_imgs)
            d_out_real_zebra = D_for_zebra(zebra_imgs)

            # 偽の画像を生成して判定
            fake_horse_images = G_zebra2horse(zebra_imgs)
            fake_zebra_images = G_horse2zebra(horse_imgs)
            d_out_fake_horse = D_for_horse(fake_horse_images)
            d_out_fake_zebra = D_for_zebra(fake_zebra_images)

            # 誤差をhinge version of the adversarial lossを利用
            h_d_loss_real = torch.nn.ReLU()(1.0 - d_out_real_horse).mean()
            z_d_loss_real = torch.nn.ReLU()(1.0 - d_out_real_zebra).mean()

            h_d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake_horse).mean()
            z_d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake_zebra).mean()

            h_d_loss = h_d_loss_real + h_d_loss_fake
            z_d_loss = z_d_loss_real + z_d_loss_fake

            # バックプロパゲーション
            g_horse2zebra_optimizer.zero_grad()
            g_zebra2horse_optimizer.zero_grad()
            d_horse_optimizer.zero_grad()
            d_zebra_optimizer.zero_grad()

            h_d_loss.backward()
            z_d_loss.backward()
            d_horse_optimizer.step()
            d_zebra_optimizer.step()

            epoch_d_h_loss += h_d_loss.item()
            epoch_d_z_loss += z_d_loss.item()

            # ----------
            # 2. Generatorの学習
            # ----------
            # 偽の画像を生成して判定するのとサイクル損失を
            fake_horse_images = G_zebra2horse(zebra_imgs)
            fake_zebra_images = G_horse2zebra(horse_imgs)
            d_out_fake_horse = D_for_horse(fake_horse_images)
            d_out_fake_zebra = D_for_zebra(fake_zebra_images)

            # GAN lossを計算
            h_g_GAN_loss = - d_out_fake_horse.mean()
            z_g_GAN_loss = - d_out_fake_zebra.mean()

            # 再構成画像を生成
            reconst_horse_images = G_zebra2horse(fake_zebra_images)
            reconst_zebra_images = G_horse2zebra(fake_horse_images)

            # L1lossを計算
            h_g_L1_loss = nn.L1Loss()(reconst_horse_images, horse_imgs)
            z_g_L1_loss = nn.L1Loss()(reconst_zebra_images, zebra_imgs)
            g_total_loss = h_g_GAN_loss + z_g_GAN_loss + lambda_for_cycLoss * (h_g_L1_loss + z_g_L1_loss)

            epoch_g_z2h_loss += h_g_GAN_loss.item()
            epoch_g_h2z_loss += z_g_GAN_loss.item()
            epoch_reconst_loss += lambda_for_cycLoss * (h_g_L1_loss.item() + z_g_L1_loss.item())


            # バックプロパゲーション
            g_zebra2horse_optimizer.zero_grad()
            g_horse2zebra_optimizer.zero_grad()
            d_horse_optimizer.zero_grad()
            d_zebra_optimizer.zero_grad()
            g_total_loss.backward()
            g_zebra2horse_optimizer.step()
            g_horse2zebra_optimizer.step()

            # ----------
            # 3. 記録（メモリの都合からいろいろ手を加えている途中）
            # ---------
            '''
            print('have looked : ', iter_num * batch_size)
            sample_fake_zebra_img = torchvision.utils.make_grid(fake_zebra_images.cpu()[0].unsqueeze(0))
            sample_horse_img = torchvision.utils.make_grid(horse_imgs.cpu()[0].unsqueeze(0))
            img_transformed = np.transpose(sample_horse_img.detach().numpy(), (1, 2, 0))
            plt.imshow(img_transformed)
            plt.pause(.01)
            img_transformed = np.transpose(sample_fake_zebra_img.detach().numpy(), (1, 2, 0))
            plt.imshow(img_transformed)
            plt.pause(.01)
            '''

            if (iteration - 1) % save_freq == 0:
                torch.save(G_horse2zebra.state_dict(),
                           './models/G_horse2zebra.pth')
                torch.save(G_zebra2horse.state_dict(),
                           './models/G_zebra2horse.pth')
                torch.save(D_for_horse.state_dict(),
                           './models/D_horse.pth')
                torch.save(D_for_zebra.state_dict(),
                           './models/D_zebra.pth')
                input_horse_for_save = np.clip(horse_imgs[0].cpu().numpy().transpose((1, 2, 0)), 0.0, 1.0)
                input_zebra_for_save = np.clip(zebra_imgs[0].cpu().numpy().transpose((1, 2, 0)), 0.0, 1.0)
                output_horse_for_save = np.clip(fake_horse_images.cpu()[0].detach().numpy().transpose((1, 2, 0)), 0.0, 1.0)
                output_zebra_for_save = np.clip(fake_zebra_images.cpu()[0].detach().numpy().transpose((1, 2, 0)), 0.0, 1.0)
                plt.imsave('./save_img/in_horse_{}.jpg'.format(iteration - 1), input_horse_for_save)
                plt.imsave('./save_img/in_zebra_{}.jpg'.format(iteration - 1), input_zebra_for_save)
                plt.imsave('./save_img/out_horse_{}.jpg'.format(iteration - 1), output_horse_for_save)
                plt.imsave('./save_img/out_zebra_{}.jpg'.format(iteration - 1), output_zebra_for_save)
            iteration += 1


        t_epoch_finish = time.time()
        print('----------')
        print('epoch {}'.format(epoch))
        print('Epoch_horseD_Loss : {:.4f}  || Epoch_zebraD_Loss : {:.4f}'.format(epoch_d_h_loss/horse_batch_size,
                                                                                 epoch_d_z_loss/zebra_batch_size))
        print('Epoch_horseG_Loss : {:.4f}  || Epoch_zebraG_Loss : {:.4f}'.format(epoch_g_z2h_loss/horse_batch_size,
                                                                                 epoch_g_h2z_loss/zebra_batch_size))
        print('Epoch_reconstruction_Loss : {:.4f}'.format(epoch_reconst_loss/horse_batch_size))
        print('timer: {:.4f} sec'.format(t_epoch_finish - t_epoch_start))

        t_epoch_start = time.time()

    return G_zebra2horse, G_horse2zebra, D_for_horse, D_for_zebra

num_epoch = 1000
train_horse_img_list, train_zebra_img_list = make_horse2zebra_datapath_list()
train_horse_img_list = train_horse_img_list[0:1064]
train_zebra_img_list = train_zebra_img_list[0:1064]
mean = (0.5,)
std = (0.5,)
os.makedirs('./save_img', exist_ok=True)
os.makedirs('./models', exist_ok=True)
train_horse_dataset = GAN_Img_Dataset(file_list=train_horse_img_list, transform=ImageTransform(mean, std))
train_zebra_dataset = GAN_Img_Dataset(file_list=train_zebra_img_list, transform=ImageTransform(mean, std))
batch_size = 8
train_horse_dataloader = torch.utils.data.DataLoader(train_horse_dataset, batch_size=batch_size, shuffle=True)
train_zebra_dataloader = torch.utils.data.DataLoader(train_zebra_dataset, batch_size=batch_size, shuffle=True)
G_A, G_B, D_A, D_B = train_model(G_horse2zebra, G_zebra2horse, D_for_horse, D_for_zebra,
                                 train_horse_dataloader, train_zebra_dataloader, num_epoch,
                                 train_horse_img_list, lambda_for_cycLoss=10)
