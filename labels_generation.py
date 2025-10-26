import os
import numpy as np
import glob

if __name__ == '__main__':
    idx = 5
    vid_path = f'/root/autodl-tmp/LAV-CVPR21/test/pouring2/vid{idx}'
    num_traj = len(glob.glob(os.path.join(vid_path, '*.jpg')))
    labels = np.zeros(num_traj,dtype=int)
    square_labels = ['grasp','reach1','reach2','release']
    #reach
    bound1 = 55
    #move
    bound2 = 80
    #pour
    bound3 = 175
    #move back
    bound4 = 210
    #withdraw
    labels[bound1:bound2] = 1
    labels[bound2:bound3] = 2
    labels[bound3:bound4] = 3
    labels[bound4:] = 4
    # labels[bound5:] = 5
    print(labels)
    #save
    np.save(os.path.join(vid_path,'labels.npy'),labels)
    # idx = 1
    # vid_path = f'/root/autodl-tmp/LAV-CVPR21/test/vid{idx}'
    # num_traj = len(glob.glob(os.path.join(vid_path, '*.jpg')))
    # labels = np.zeros(num_traj,dtype=int)
    # ori_labels = np.load(os.path.join(vid_path, 'labels.npy'), allow_pickle=True)
    # labels[:-1] = ori_labels
    # labels[-1] = labels[-2]
    # print(labels,len(labels))
    # np.save(os.path.join(vid_path,'labels.npy'),labels)