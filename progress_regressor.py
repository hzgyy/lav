import os
import numpy as np

import utils
import align_dataset_test as align_dataset
from config import CONFIG

from evals.phase_classification import evaluate_phase_classification, compute_ap
from evals.phase_classification import fit_svm,evaluate_svm
from evals.kendalls_tau import evaluate_kendalls_tau
from evals.phase_progression import evaluate_phase_progression

from train import AlignNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import random
import argparse
import glob
from natsort import natsorted
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from align_dataset import get_steps_with_context

def make_video_with_captions(imgs, captions, save_path="output.mp4", fps=10, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    """
    将图像序列和对应文字合成为视频
    imgs: list of PIL.Image
    captions: list of str
    save_path: 输出视频路径
    fps: 帧率
    font_path: 字体路径 (Linux默认路径，可换成别的)
    """
    assert len(imgs) == len(captions), "imgs 和 captions 长度不一致！"
    
    # 转为统一尺寸
    w, h = imgs[0].size
    size = (w, h)
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, size)
    
    # 字体
    try:
        font = ImageFont.truetype(font_path, size=24)
    except:
        font = ImageFont.load_default()

    for img, cap in zip(imgs, captions):
        # 确保是 RGB 格式
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # 在底部绘制半透明背景
        text_w, text_h = draw.textsize(cap, font=font)
        margin = 10
        overlay_height = text_h + 2 * margin
        draw.rectangle(
            [(0, h - overlay_height), (w, h)],
            fill=(0, 0, 0, 128)
        )

        # 写入文字（白色）
        draw.text((margin, h - text_h - margin), cap, fill=(255, 255, 255), font=font)
        
        # 转换为 OpenCV 格式
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"✅ 视频已保存至 {save_path}")

# -------------------------------
# Dataset & DataLoader
# -------------------------------
class ProgDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------
# 小型 MLP 回归模型
# -------------------------------
class ProgressRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main(ckpt,args):
    # get trained model
    # _, ckpt_step = ckpt.split('.')[0].split('_')[-2:]
    # ckpt_step = int(ckpt_step.split('=')[1])
    # DEST = os.path.join(args.dest, 'eval_step_{}'.format(ckpt_step))

    device = f"cuda:{args.device}"
    # device = "cpu"
    model = AlignNet.load_from_checkpoint(ckpt, map_location=device)
    # model.hparams.config.DATA.NUM_CONTEXT = 1
    model.to(device)
    model.eval()
    # grad off
    # torch.set_grad_enabled(False)
    
    # read in images:
    if args.data_path == None:
        data_path = '/root/autodl-tmp/LAV-CVPR21/test/pouring2'
    else:
        data_path = args.data_path
    num_train_trajs = 5
    #num_val_trajs = 10-num_train_trajs
    #preprarations
    _transforms = utils.get_transforms(augment=False)
    labels_all = []
    embs_all = []
    lengths = []
    demo_paths = natsorted(glob.glob(os.path.join(args.demo_path, "vid*")))[:args.N]
    trajs = demo_paths
    trajs.append(data_path)
    with torch.no_grad():
        for i,traj in enumerate(trajs):
            # read images
            print(traj)
            img_paths = natsorted(glob.glob(os.path.join(traj, '*.jpg')))
            imgs = utils.get_pil_images(img_paths)
            imgs = _transforms(imgs)
            n = imgs.shape[0]
            steps = np.arange(n)
            steps = get_steps_with_context(steps, num_context=2,context_stride=5)
            imgs = imgs[steps]
            # get embeddings
            a_X = imgs.to(device).unsqueeze(0)
            original = a_X.shape[1]//2
            a_emb = model(a_X)
            print(f"emb shape:{a_emb.shape},original:{original}")
            a_emb = a_emb[:, :original,:]

            a_emb_reduced = a_emb.squeeze(0).detach().cpu().numpy()
            if i == len(trajs)-1:
                test_emb = a_emb_reduced
                continue
            embs_all.append(a_emb_reduced)
            labels = np.arange(a_emb_reduced.shape[0])/a_emb_reduced.shape[0]
            labels_all.append(labels)
            lengths.append(labels.shape[0])
    # 拼接数据
    X = np.concatenate(embs_all, axis=0).astype(np.float32)   # (N, D)
    y = np.concatenate(labels_all, axis=0).astype(np.float32) # (N,)

    # 转成 tensor
    X = torch.from_numpy(X)
    y = torch.from_numpy(y).unsqueeze(1)   # (N,1)
    
    # do training
    dataset = ProgDataset(X, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    input_dim = X.shape[1]
    regressor = ProgressRegressor(input_dim).cuda()

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(regressor.parameters(), lr=1e-3)

    EPOCHS = 20
    for epoch in range(EPOCHS):
        regressor.train()
        total_loss = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()

            pred = regressor(batch_X)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] loss = {total_loss / len(loader):.6f}")
        
    # -------------------------------
    # 推理示例
    # -------------------------------
    regressor.eval()
    with torch.no_grad():
        example_emb = torch.from_numpy(test_emb).float().cuda()
        pred_progress = regressor(example_emb).cpu().numpy().squeeze()
        print("Predicted progress:", pred_progress)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--demo_path', type=str, default=None)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--description', type=str, default=None)

    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='Cuda device to be used')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_frames', type=int, default=None, help='Path to dataset')

    args = parser.parse_args()

    # if os.path.isdir(args.model_path):
    #     ckpts = natsorted(glob.glob(os.path.join(args.model_path, '*')))
    # else:
    #     ckpts = [args.model_path]
    ckpt = args.model_path
    
    ckpt_mul = args.device
    main(ckpt, args)