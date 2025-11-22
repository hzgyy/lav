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

def get_embeddings(model, data, labels_npy, args):

    embeddings = []
    labels = []
    frame_paths = []
    names = []

    device = f"cuda:{args.device}"
    
    for act_iter in iter(data):
        for i, seq_iter in enumerate(act_iter):
            seq_embs = []
            seq_fpaths = []
            original = 0
            for _, batch in enumerate(seq_iter):
                a_X, a_name, a_frames = batch
                a_X = a_X.to(device).unsqueeze(0)
                original = a_X.shape[1]//2
                
#                 a_X = a_X[:,:a_X.shape[1]//2,:,:,:]
#                 original = a_X.shape[1]//2
                
#                 if ((args.num_frames*2)-a_X.shape[1]) > 0:
#                     b = a_X[:, -1].clone()
#                     b = torch.stack([b]*((args.num_frames*2)-a_X.shape[1]),axis=1).to(device)
#                     a_X = torch.concat([a_X,b], axis=1)
                
                b =  a_X[:, -1].clone()
                try:
                    b = torch.stack([b]*((args.num_frames*2)-a_X.shape[1]),axis=1).to(device)
                except:
                    b = torch.from_numpy(np.array([])).float().to(device)
                a_X = torch.concat([a_X,b], axis=1)
                a_emb = model(a_X)[:, :original,:]
                
                if args.verbose:
                    print(f'Seq: {i}, ', a_emb.shape)

                seq_embs.append(a_emb.squeeze(0).detach().cpu().numpy())
                seq_fpaths.extend(a_frames)
            
            seq_embs = np.concatenate(seq_embs, axis=0)
            
            name = str(a_name).split('/')[-1]
            # name = name[:8] + '/' + name[8:10] + '/' + name[10:]
            lab = labels_npy[name]['labels']
            end = min(seq_embs.shape[0], len(lab))
            lab = lab[:end]#.T
            seq_embs = seq_embs[:end]
            print(seq_embs.shape, len(lab))
            embeddings.append(seq_embs[:end])
            frame_paths.append(seq_fpaths)
            names.append(a_name)
            labels.append(lab)

    return embeddings, names, labels

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
    torch.set_grad_enabled(False)
    
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
    # for i in tqdm(range(5)):
    for i,traj in enumerate(trajs):
        # read images
        print(traj)
        img_paths = natsorted(glob.glob(os.path.join(traj, '*.jpg')))
        imgs = utils.get_pil_images(img_paths)
        print(len(img_paths))
        imgs = _transforms(imgs)
        print(imgs.shape)
        # get labels
        # labels = np.load(os.path.join(trajs_path, 'labels.npy'), allow_pickle=True).item()
        # labels = np.load(os.path.join(trajs_path, 'labels.npy'), allow_pickle=True)
        # assert labels.shape[0] == len(imgs),f"imgs and labels length unpaired:{labels.shape[0]}vs{len(imgs)}"
        n = imgs.shape[0]
        steps = np.arange(n)
        steps = get_steps_with_context(steps, num_context=2,context_stride=5)
        imgs = imgs[steps]
        # get embeddings
        a_X = imgs.to(device).unsqueeze(0)
        original = a_X.shape[1]//2
        # b =  a_X[:, -1].clone()
        # try:
        #     b = torch.stack([b]*((args.num_frames*2)-a_X.shape[1]),axis=1).to(device)
        # except:
        #     b = torch.from_numpy(np.array([])).float().to(device)
        # a_X = torch.concat([a_X,b], axis=1)
        # a_emb = model(a_X)[:, :original,:]
        a_emb = model(a_X)
        print(f"emb shape:{a_emb.shape},original:{original}")
        a_emb = a_emb[:, :original,:]

        # do PCA to reduce dimension to 50
        # pca = PCA(n_components=50)
        # a_emb_reduced = pca.fit_transform(a_emb.squeeze(0).detach().cpu().numpy())
        # print(f"PCA reduced shape:{a_emb_reduced.shape}")
        a_emb_reduced = a_emb.squeeze(0).detach().cpu().numpy()
        embs_all.append(a_emb_reduced)
        labels = np.zeros((a_emb_reduced.shape[0],),dtype=int)
        labels[::] = i
        labels_all.append(labels)
        lengths.append(labels.shape[0])
        # a_emb = model(a_X)
        # labels = labels[:-1:2]
        # assert False, a_emb[0].shape
        # # print(labels)
        # assert labels.shape[0] == a_emb.shape[1],f"labels and embs length unpaired:{labels.shape[0]}vs{a_emb.shape[1]}"
        # labels_all.append(labels)
        # if args.verbose:
        #     print(f'Seq: {i}, ', a_emb.shape)
        # embs_all.append(a_emb.squeeze(0).detach().cpu().numpy())
    # print(embs_all[0].shape,labels_all[0].shape)
    embs_all = np.concatenate(embs_all,axis=0)
    # do TSNE
    tsne = TSNE(n_components=2)
    embs_all_2d = tsne.fit_transform(embs_all)
    print(f"TSNE reduced shape:{embs_all_2d.shape}")
    colors_all = np.concatenate(labels_all,axis=0)
    assert embs_all_2d.shape[0] == colors_all.shape[0],f"embs and labels length unpaired:{embs_all_2d.shape[0]}vs{colors_all.shape[0]}"
    # plot and save all trajectories together
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(
        embs_all_2d[:, 0],
        embs_all_2d[:, 1],
        c=colors_all,
        cmap='viridis',   # "viridis" 或 "plasma" 比 "jet" 更平滑自然
        alpha=0.7
    )
    offset = 0
    for i, L in enumerate(lengths):
        start = offset
        end = offset + L - 1

        # 起点：用大星形标记
        plt.scatter(
            embs_all_2d[start, 0],
            embs_all_2d[start, 1],
            marker='*',
            s=200,  # 点的大小
            edgecolors='black',
            linewidths=1.5,
            c=[colors_all[start]]
        )

        # 终点：用大X标记
        plt.scatter(
            embs_all_2d[end, 0],
            embs_all_2d[end, 1],
            marker='X',
            s=200,
            edgecolors='black',
            linewidths=1.5,
            c=[colors_all[end]]
        )

        step = 25

        for i in range(start, end, step):
            plt.text(
                embs_all_2d[i, 0],
                embs_all_2d[i, 1],
                str(i-start),
                fontsize=10,
                ha='center',
                va='center',
                color='black',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
            )

        offset += L
    # Add legend
    classes = np.unique(colors_all)
    for cls in classes:
        plt.scatter([], [], c=scatter.cmap(scatter.norm(cls)), label=f'Class {cls}')
    plt.legend(title="Class Label")
    plt.title(f'All Trajectories Embeddings Visualization')
    plt.savefig(os.path.join(args.root,f"{args.description}.png"))
    plt.close()
    #some extra calculation
    query_emb = a_emb_reduced
    demo_emb = embs_all[:-query_emb.shape[0]]
    query_emb = query_emb[:,np.newaxis,:]
    demo_emb = demo_emb[np.newaxis,:,:]
    dist = np.sum((query_emb - demo_emb)**2,axis=-1)
    dist_mean = np.min(dist,axis=-1)
    print(dist_mean)
    assert False, "stop here"
    # do training
    # train_embs = embs_all[:num_train_trajs]
    # train_labels = labels_all[:num_train_trajs]
    # val_embs = embs_all[num_train_trajs:]
    # val_labels = labels_all[num_train_trajs:]
    # svm_model, train_acc = fit_svm(train_embs, train_labels)
    # val_acc, conf_mat,val_preds = evaluate_svm(svm_model, val_embs, val_labels)

    # #generate videos
    # val_traj_path = os.path.join(data_path,f'vid{num_train_trajs}')
    # val_img_paths = natsorted(glob.glob(os.path.join(val_traj_path, '*.jpg')))
    # val_imgs = utils.get_pil_images(val_img_paths)
    # val_imgs = val_imgs[:-1:2]
    # # str_labels = ["reach bottle","move to pour","pour","move back","withdraw hand"]
    # str_labels = ["grasp","move above","move down","release"]
    # captions = [str_labels[i] for i in val_preds]
    # assert type(captions[0]) == str,f"caption type error{type(captions[0])}"
    # assert len(val_imgs) == len(captions),f"imgs and captions length unpaired:{len(val_imgs)}vs{len(captions)}"
    # make_video_with_captions(val_imgs,captions,save_path=os.path.join("./test","output.mp4"),fps=5)
    # print('\n-----------------------------')
    # print('Train-Acc: ', train_acc)
    # print('Val-Acc: ', val_acc)
    # print('Conf-Mat: ', conf_mat)

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