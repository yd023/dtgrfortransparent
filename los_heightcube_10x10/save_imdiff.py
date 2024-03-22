import cv2
import numpy as np
import matplotlib.pyplot  as plt
import os

setting = "iterations_lr1_spp_dinamic_20_tv_0.001_firstspp32_changerateofspp_2"
data = np.genfromtxt(f"results_standard/{setting}/iter_loss.log", delimiter=',')

if not os.path.exists(f'diff_timg'):
    os.mkdir(f'diff_timg')
if not os.path.exists(os.path.join('diff_timg', setting)):
    os.mkdir(os.path.join('diff_timg', setting))
if not os.path.exists(os.path.join('diff_timg', setting, 'images_diff')):
    os.mkdir(os.path.join('diff_timg', setting, 'images_diff'))
if not os.path.exists(os.path.join('diff_timg', setting, 'images_iter')):
    os.mkdir(os.path.join('diff_timg', setting, 'images_iter'))
if not os.path.exists(os.path.join('diff_timg', setting, 'images_difiter')):
    os.mkdir(os.path.join('diff_timg', setting, 'images_difiter'))

n=len(os.listdir(f"results_standard/{setting}imgs"))
print(n)

for iter in range(n):

    images_ref = []
    images_iter = []
    images_diff = []

    for i in range(160, 275, 10):
        img_ref = cv2.imread("results_standard/images_ref/los_heightcube_10x10_standard_0_tau_" + str(i) + ".png")
        images_ref.append(img_ref)
        img_iter = cv2.imread(f"results_standard/{setting}imgs/imgs_iter_{iter}/los_heightcube_10x10_standard_0_tau_" + str(i) + ".png")
        images_iter.append(img_iter)
        im_diff = img_ref.astype(int) - img_iter.astype(int)
        im_diff_abs = np.abs(im_diff)
        images_diff.append(im_diff_abs)

    fig = plt.figure()

    for i, im in enumerate(images_diff):
        fig.add_subplot(3,4,i+1).set_title(str(i), fontsize=8)
        plt.imshow(im)
        plt.axis('off')
    
    plt.savefig("diff_timg/{}/images_diff/iter{:0=4}.png".format(setting,iter))

    fig2 = plt.figure()

    for i, im in enumerate(images_iter):
        fig2.add_subplot(3,4,i+1).set_title(str(i), fontsize=8)
        plt.imshow(im)
        plt.axis('off')
    
    fig2.suptitle(f"iter = {iter}\n spp = {(int)(data[iter,-1])}")
    plt.savefig("diff_timg/{}/images_iter/iter{:0=4}.png".format(setting,iter))

for iter in range(n):
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(wspace=0)

    image_diff = cv2.imread("diff_timg/{}/images_diff/iter{:0=4}.png".format(setting,iter))
    fig.add_subplot(1,2,1).set_title("difference",fontsize=15)
    plt.imshow(image_diff)
    plt.axis('off')
    
    image_iter = cv2.imread("diff_timg/{}/images_iter/iter{:0=4}.png".format(setting,iter))
    fig.add_subplot(1,2,2)#.set_title(f"iter{iter}",fontsize=15)
    plt.imshow(image_iter)
    plt.axis('off')
    
    plt.savefig("diff_timg/{}/images_difiter/iter{:0=4}.png".format(setting,iter))


