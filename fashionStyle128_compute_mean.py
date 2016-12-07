import os, pickle, csv
import numpy as np
from scipy import misc
import pdb


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def load_image(path):
    img = misc.imread(path)
    if img.ndim == 2:
        img = to_rgb(img)
    elif img.ndim == 3 and img.shape[2]>3:
        img = img[:,:,0:3]
    return img.astype(np.float32)

def load_images(image_indices, image_paths, im_height, im_width):
    images = np.zeros((len(image_indices), im_height, im_width, 3))
    for i,j in enumerate(image_indices):
        img = misc.imread(image_paths[j])
        if img.ndim == 2:
            img = to_rgb(img)
        elif img.ndim == 3 and img.shape[2]>3:
            img = img[:,:,0:3]
        images[i,:,:,:] = img
    images_float = images.astype(np.float32)
    return images_float

def save_mean(sum_img, mean_root, count):
    mean_file = os.path.join(mean_root, "out_{0}.npy".format(count))
    mean_img = sum_img / count
    np.save(mean_file, mean_img)


def save_stats(save_path, X):
    save = {}
    save['mean'] = X.mean(axis=0)
    x0 = X[:,:,:,0].flatten()
    x1 = X[:,:,:,1].flatten()
    x2 = X[:,:,:,2].flatten()
    #X.sum(axis=0).sum(axis=0).sum(axis=0) / (X.shape[0]*X.shape[1]*X.shape[2])
    save['mean_channels'] = np.zeros(3)
    save['mean_channels'][0] = x0.mean()
    save['mean_channels'][1] = x1.mean()
    save['mean_channels'][2] = x2.mean()

    save['channel_std'] = np.zeros(3)
    save['channel_std'][0] = x0.std()
    save['channel_std'][1] = x1.std()
    save['channel_std'][2] = x2.std()
    np.savez(save_path, **save)


def main():
    dataset_path = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/"
    save_mean_folder = "mean/"
    im_height = 384
    im_width = 256

    with open(os.path.join(dataset_path,'photos.txt'), 'rb') as f:
        reader = csv.reader(f)
        image_paths = [os.path.join(dataset_path, p[0]) for p in reader]
    trainids = np.load(os.path.join(dataset_path,'trainids.npy'))

    mean_root = os.path.join(dataset_path, save_mean_folder)
    if not os.path.exists(mean_root):
    	os.mkdir(mean_root)

    sum_img  = np.zeros((im_height, im_width, 3))
    for e, i in enumerate(trainids):
        img = load_image(image_paths[i])
        sum_img += img
        if e % 100 == 0:
            print("[{0}/{1}] loaded".format(e,len(trainids)))
        if e % 10000 == 0 and e > 100:
             save_mean(sum_img, mean_root, e+1)
             print("saving mean {0}".format(e))

    save_mean(sum_img, mean_root, e+1) #len(trainids)

def main2():
    dataset_path = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/"
    with open(os.path.join(dataset_path,'photos.txt'), 'rb') as f:
        reader = csv.reader(f)
        image_paths = [os.path.join(dataset_path, p[0]) for p in reader]
    trainids = np.load(os.path.join(dataset_path,'trainids.npy'))
    im_height = 384
    im_width = 256
    C = int(len(trainids)/4.0)
    save_path_root = os.path.join(dataset_path, "stats2")
    if not os.path.exists(save_path_root):
        os.mkdir(save_path_root)
    save_path = os.path.join(save_path_root, "stats{0}".format(C))

    X = load_images(trainids[:C], image_paths, im_height, im_width)
    #pdb.set_trace()
    save_stats(save_path, X)

if __name__ == '__main__':
    main2()