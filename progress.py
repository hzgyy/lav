import os
import numpy as np

import utils

from train import AlignNet
import torch

# import random
import argparse
import glob
from natsort import natsorted
from torchvision import transforms
# from tqdm import tqdm
# import cv2
# from PIL import Image, ImageDraw, ImageFont

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
import preprocessing.transforms_video as tv
from PIL import Image
from align_dataset import get_steps_with_context
# from config import CONFIG

class ProgressEstimator():
    def __init__(self,ckpt,device,k=3):
        # device = f"cuda:{args.device}"
        self.device = device
        self.model = AlignNet.load_from_checkpoint(ckpt, map_location=device)
        self.model.to(device)
        self.model.eval()
        # grad off
        # torch.set_grad_enabled(False)
        self.all_transform = transforms.Compose([tv.ToTensorVideo(),tv.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
        self.norm_transform = tv.NormalizeVideo(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.k = k
        self.demo_flag = False
        self.demo_embs = []
        self.demo_labels = []
        self.demo_len = 0
        self.similarity_type = 'l2'

    def set_group_demonstration(self,demo_embs,demo_labels):
        demo_len = 0
        for emb, label in zip(demo_embs,demo_labels):
            assert emb.shape[0] == label.shape[0]
            self.demo_embs.append(emb)
            self.demo_labels.append(label)
            demo_len += 1
        self.demo_len = demo_len
        if not self.demo_flag:
            self.demo_flag = True

    def set_demonstration(self,demo_embs,demo_labels):
        assert demo_embs.shape[0] == demo_labels.shape[0]
        self.demo_embs = demo_embs                 #np.array (N,128)
        self.demo_labels = demo_labels             #np.array (N,)
        if not self.demo_flag:
            self.demo_flag = True

    def get_embs(self,imgs,batch_size=128):
        # turn imgs into correct tensor
        # if type(imgs) == torch.Tensor:
        #     imgs = self.norm_transform(imgs)
        # else:
        #     imgs = self.all_transform(imgs)
            # trans = tv.ToTensorVideo()
            # imgs = trans(imgs)
        n = imgs.shape[0]
        steps = np.arange(n)
        steps = get_steps_with_context(steps, num_context=2,context_stride=5)
        # idx = torch.arange(n)
        # idx = torch.cat([idx[:1], torch.repeat_interleave(idx[1:], 2),idx[-1:]])
        imgs = imgs[steps]
        embs_all = np.empty((imgs.shape[0]//2,128))
        ptr = 0
        batch_size = imgs.shape[0]
        with torch.no_grad():
            for i in range(0, imgs.shape[0], batch_size):
                end = min(imgs.shape[0],i+batch_size)
                #get batch
                batch = imgs[i:end].to(self.device)
                # get embeddings
                # a_X = imgs.to(self.device).unsqueeze(0)
                a_X = batch.to(self.device).unsqueeze(0)
                original = a_X.shape[1]//2
                a_emb = self.model(a_X)
                #print(f"emb shape:{a_emb.shape},original:{original}")
                a_emb = a_emb[:, :original,:]

                a_emb_reduced = a_emb.squeeze(0).detach().cpu().numpy()
                # a_emb_reduced = a_emb.squeeze(0).detach()
                # a_emb_reduced = a_emb_reduced / np.linalg.norm(a_emb_reduced, axis=1, keepdims=True)  # (N,128)
                # embs.append(a_emb_reduced)
                embs_all[ptr:ptr+a_emb_reduced.shape[0],:] = a_emb_reduced
                ptr += a_emb_reduced.shape[0]
            # a_emb_reduced = a_emb_reduced / torch.linalg.norm(a_emb_reduced,dim=-1,keepdim=True)
        # embs_all = np.concatenate(embs,axis=0)
        return embs_all
    
    def get_embs_gpu(self,imgs,batch_size=128):
        # turn imgs into correct tensor
        if type(imgs) == torch.Tensor:
            imgs = self.norm_transform(imgs)
        else:
            imgs = self.all_transform(imgs)
            # trans = tv.ToTensorVideo()
            # imgs = trans(imgs)
        n = imgs.shape[0]
        idx = torch.arange(n)
        idx = torch.cat([idx[:1], torch.repeat_interleave(idx[1:], 2),idx[-1:]])
        imgs = imgs[idx]
        embs_all = torch.empty((imgs.shape[0]//2,128))
        ptr = 0
        with torch.no_grad():
            for i in range(0, imgs.shape[0], batch_size):
                end = min(imgs.shape[0],i+batch_size)
                #get batch
                batch = imgs[i:end].to(self.device)
                # get embeddings
                # a_X = imgs.to(self.device).unsqueeze(0)
                a_X = batch.to(self.device).unsqueeze(0)
                original = a_X.shape[1]//2
                a_emb = self.model(a_X)
                #print(f"emb shape:{a_emb.shape},original:{original}")
                a_emb = a_emb[:, :original,:]

                # a_emb_reduced = a_emb.squeeze(0).detach().cpu().numpy()
                a_emb_reduced = a_emb.squeeze(0).detach()
                a_emb_reduced = a_emb_reduced / torch.linalg.norm(a_emb_reduced,dim=-1,keepdim=True)
                # a_emb_reduced = a_emb_reduced / np.linalg.norm(a_emb_reduced, axis=1, keepdims=True)  # (N,128)
                # embs.append(a_emb_reduced)
                embs_all[ptr:ptr+a_emb_reduced.shape[0],:] = a_emb_reduced
                ptr += a_emb_reduced.shape[0]
        # embs_all = np.concatenate(embs,axis=0)
        return embs_all
    
    def get_sim(self,x,y):
        x = x[:,np.newaxis,:]
        y = y[np.newaxis,:,:]
        if self.similarity_type == 'cosine':
            # x = F.normalize(x, p=2, dim=2)
            # y = F.normalize(y, p=2, dim=2)
            dist = 1.- x @ y.T
        else:  # l2
            dist = np.sum((x - y)**2,axis=-1)
        return dist

    def get_group_estimate(self,imgs):
        query_embs = self.get_embs(imgs) #(M,128)
        M = query_embs.shape[0]
        D = self.demo_len
        progress_all = np.empty((M, D))
        dist_all = np.empty((M, D))
        for idx,(demo_embs, demo_labels) in enumerate(zip(self.demo_embs,self.demo_labels)):
            # sim = 1 - query_embs @ demo_embs.T # (M,N_i) 
            sim = self.get_sim(query_embs,demo_embs)
            min_indices = np.argmin(sim,axis=-1)    #(M,1)
            progress = demo_labels[min_indices]     #(M,)
            # dist = np.take_along_axis(sim,min_indices,axis=-1)   #(M,)
            # progress_all.append(progress)
            # dist_all.append(dist)
            progress_all[:,idx] = progress
            dist_all[:,idx] = sim[np.arange(M), min_indices]
        # progress_all = np.concatenate(progress_all,axis=-1) #(M,N_demo)
        # dist_all = np.concatenate(dist_all,axis=-1)
        # get the best k
        # knn_indices = np.argpartition(dist_all, self.k, axis=1)[:, :self.k] # (M,K)
        # threshold = 0.1
        # outlier_mask = dist_all > threshold
        # with np.printoptions(threshold=np.inf):
        #     print(dist_all)
        # estimated = np.mean(np.take_along_axis(progress_all,knn_indices,axis=-1),axis=-1)
        estimated = np.mean(progress_all,axis=-1)
        dist_mean = np.mean(dist_all,axis=-1)
        print(f"dist:{dist_mean}")
        threshold = 0.1
        outlier_mask = dist_mean > threshold
        most_progress = 0.0
        for i,mask in enumerate(outlier_mask):
            if not mask:
                most_progress = estimated[i]
                continue
            estimated[i] = most_progress
        # print(f"dist:{dist_mean}")
        return estimated

    def get_estimate(self,imgs): 
        query_embs = self.get_embs(imgs) #(M,128) 
        # I assume all the embs are normalized 
        # sim = 1 - query_embs @ self.demo_embs.T # (M,N)
        sim = self.get_sim(query_embs,self.demo_embs)
        knn_indices = np.argpartition(sim, self.k, axis=1)[:, :self.k] # (M,K)
        # with np.printoptions(threshold=np.inf):
        #     print(f"idxs:{knn_indices}")
        estimated = np.mean(self.demo_labels[knn_indices], axis=1) # (M,)
        knn_sims = np.take_along_axis(sim, knn_indices, axis=1)  # (M, K)
        dist_k = np.mean(knn_sims,axis=-1)   #(M,)
        print(f"dist:{np.min(sim,axis=-1)}")
        threshold = 0.1
        outlier_mask = dist_k > threshold
        most_progress = 0.0
        for i,mask in enumerate(outlier_mask):
            if not mask:
                most_progress = estimated[i]
                continue
            estimated[i] = most_progress
        # print(f"sim_k:{dist_k}")
        return estimated
    
    def get_distance(self,imgs):
        query_embs = self.get_embs(imgs) #(M,128) 
        # I assume all the embs are normalized query_embs = self.get_embs(imgs) #(M,128) 
        # I assume all the embs are normalized 
        sim = 1-query_embs @ self.demo_embs.T # (M,N) 
        min_dist = np.min(sim,axis=-1)
        print(min_dist)
        return min_dist

    def get_estimate_gpu(self,imgs):
        with torch.no_grad():
            query_embs = self.get_embs(imgs)    #(M,128)
            # I assume all the embs are normalized
            # 计算相似度 (GPU)
            sim = 1- query_embs @ self.demo_embs.T  # (M, N)

            # 取前K个最近邻
            sim_k, knn_indices = torch.topk(sim, self.k, dim=1)

            # 平均标签
            estimated = self.demo_labels[knn_indices].mean(dim=1)
        return estimated.cpu().numpy()

def set_demo_embs(demo_path,estimator,device):
    trajs_path = natsorted(glob.glob(os.path.join(demo_path,'vid*')))
    embs_all = []
    labels_all = []
    _transforms = utils.get_transforms(augment=False)
    for traj_path in trajs_path:
        print(traj_path)
        img_paths = natsorted(glob.glob(os.path.join(traj_path, '*.jpg')))
        # imgs = utils.get_pil_images(img_paths)
        imgs = utils.get_pil_images(img_paths)
        imgs = _transforms(imgs)
        print(type(imgs),imgs.shape)
        # assert False
        embs = estimator.get_embs(imgs)
        embs_all.append(embs)
        length = embs.shape[0]
        # labels = torch.arange(length,device=device)/length
        labels = np.arange(length)/length
        assert embs.shape[0] == labels.shape[0]
        labels_all.append(labels)
    embs_all = np.concatenate(embs_all,axis=0)
    labels_all = np.concatenate(labels_all,axis=0)
    # embs_all = torch.cat(embs_all,dim=0)
    # labels_all = torch.cat(labels_all,dim=0)
    print(f"get demo embs shape:{embs_all.shape},get embs labels shape:{labels_all.shape}")
    estimator.set_demonstration(embs_all,labels_all)
    return

def set_group_demo_embs(demo_path,estimator,device):
    trajs_path = glob.glob(os.path.join(demo_path,'vid*'))
    embs_all = []
    labels_all = []
    for traj_path in trajs_path:
        img_paths = natsorted(glob.glob(os.path.join(traj_path, '*.jpg')))
        # imgs = utils.get_pil_images(img_paths)
        imgs = utils.get_images(img_paths)
        # assert False
        embs = estimator.get_embs(imgs)
        embs_all.append(embs)
        length = embs.shape[0]
        # labels = torch.arange(length,device=device)/length
        labels = np.arange(length)/length
        assert embs.shape[0] == labels.shape[0]
        labels_all.append(labels)
    # embs_all = np.concatenate(embs_all,axis=0)
    # labels_all = np.concatenate(labels_all,axis=0)
    # # embs_all = torch.cat(embs_all,dim=0)
    # # labels_all = torch.cat(labels_all,dim=0)
    # print(f"get demo embs shape:{embs_all.shape},get embs labels shape:{labels_all.shape}")
    estimator.set_group_demonstration(embs_all,labels_all)
    return

class RewardTransformer():
    def __init__(self,estimator,capacity,device="cpu",gamma=0.99):
        self.estimator = estimator
        self.img_buffer = torch.empty((capacity,3,224,224),device =device)
        self.reward_buffer = np.zeros(capacity, dtype=np.float64)
        self.counter = 0
        self.gamma = gamma
        self.episode = 0
    
    def push(self,reward,img):
        i = self.counter
        # if img.max() > 1:
        #     img = img/255.0
        self.img_buffer[i].copy_(img)
        self.reward_buffer[i] = reward
        self.counter += 1

    def flush(self,reward_type = 0):
        if reward_type == 0:
            rewards = self.cal_reward()
        else:
            rewards = self.cal_dist_reward()
        # imgs = self.img_buffer[:self.counter]
        # imgs = imgs[:,[2,1,0],:,:]
        # print(imgs.max(),imgs.min())
        # tensor_to_images(imgs,f"/media/mani/Data/gyy_workspace/IBRL/ibrl/test/vid_{self.episode}")
        self.episode += 1
        self.counter = 0
        return rewards

    def cal_reward(self):
        N = self.counter
        # interval = 5
        # imgs = self.img_buffer[:N:interval]
        imgs = self.img_buffer[:N]
        rewards = self.reward_buffer[:N]
        assert self.estimator.demo_flag == True
        est_progress = self.estimator.get_estimate(imgs)    #np (N,)
        print(f"est progress:{est_progress}")
        # print(est_progress)
        # rewards[::interval][:-1] += (est_progress[1:]*self.gamma - est_progress[:-1])
        # rewards[:-1] += (est_progress[1:]*self.gamma - est_progress[:-1])
        # rewards = est_progress +rewards*5
        # rewards[:-1] += (est_progress[1:] - est_progress[:-1])
        return rewards  #np(N,)
    
    def cal_dist_reward(self):
        N = self.counter
        imgs = self.img_buffer[:N]
        rewards = self.reward_buffer[:N]
        assert self.estimator.demo_flag == True
        min_dist = self.estimator.get_distance(imgs)    #np (N,)
        # print(est_progress)
        rewards += np.exp(-min_dist*2)*0.1
        return rewards  #np(N,)

def tensor_to_images(tensor, save_dir=None):
    """
    把一个 (N, 3, H, W) 的 Tensor 转为 N 张图片（PIL.Image）。
    若 save_dir 不为空，则保存为文件。
    """
    assert tensor.ndim == 4 and tensor.shape[1] == 3, "输入必须是 (N, 3, H, W)"
    
    # 把 tensor 从 [-1,1] 或 [0,1] 转为 [0,255]
    tensor = tensor.detach().cpu().clamp(0, 1)

    to_pil = transforms.ToPILImage()
    images = []
    for i, img_tensor in enumerate(tensor):
        img = to_pil(img_tensor)
        images.append(img)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            img.save(os.path.join(save_dir, f"img_{i}.jpg"))
    return images


def main(ckpt,args):
    device = f"cuda:{0}"
    # ckpt_path = "/media/mani/Data/gyy_workspace/IBRL/ibrl/lav/trained_model/final_square_ft.pth"
    ckpt_path = ckpt
    img_path = args.demo_path
    estimator = ProgressEstimator(ckpt_path,device,k=5)
    # set_group_demo_embs(img_path,estimator,device)
    set_demo_embs(img_path,estimator,device)
    test_path = args.test_path
    img_paths = natsorted(glob.glob(os.path.join(test_path, '*jpg')))
    # img_paths = natsorted(glob.glob(os.path.join("/media/mani/Data/gyy_workspace/IBRL/ibrl/test/test", '*png')))
    print(len(img_paths))
    imgs = utils.get_pil_images(img_paths)
    # estimate = estimator.get_group_estimate(imgs)
    # print(f"progress:{estimate}")
    # sub_transform = tv.ToTensorVideo()
    _transforms = utils.get_transforms(augment=False)
    print(_transforms)
    imgs = _transforms(imgs)
    reward = np.zeros(imgs.shape[0])
    # print(reward)
    reward_trans = RewardTransformer(estimator,300)
    for r,img in zip(reward,imgs):
        reward_trans.push(r,img)
    tran_rewards = reward_trans.flush(reward_type=0)
    print(tran_rewards,type(tran_rewards))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--demo_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)

    args = parser.parse_args()

    # if os.path.isdir(args.model_path):
    #     ckpts = natsorted(glob.glob(os.path.join(args.model_path, '*')))
    # else:
    #     ckpts = [args.model_path]
    ckpt = args.model_path
    main(ckpt, args)