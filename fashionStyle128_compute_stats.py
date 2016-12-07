import os, pickle, csv
import numpy as np
from scipy import misc
import pdb


class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        self.variance = self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


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


def save_stats(mean_root, count, sum_img, ov0, ov1, ov2):
    save_path = os.path.join(mean_root, "stats{0}".format(count))
    save = {}
    save['mean'] = sum_img / count
    save['mean_channels'] = sum_img.sum(axis=0).sum(axis=0) / (count*sum_img.shape[0]*sum_img.shape[1])
    
    save['channel_std'] = np.zeros(3)
    save['channel_std'][0] = ov0.std
    save['channel_std'][1] = ov1.std
    save['channel_std'][2] = ov2.std
    np.savez(save_path, **save)

def main():
    dataset_path = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/"
    save_mean_folder = "stats/"
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
    ov0 = OnlineVariance(ddof=0)
    ov1 = OnlineVariance(ddof=0)
    ov2 = OnlineVariance(ddof=0)

    for e, i in enumerate(trainids):
        img = load_image(image_paths[i])
        sum_img += img
        x0 = img[:,:,0].flatten()
        x1 = img[:,:,1].flatten()
        x2 = img[:,:,2].flatten()
        for xi,xj,xk in zip(x0,x1,x2):
            ov0.include(xi)
            ov1.include(xj)
            ov2.include(xk)

        if e % 10 == 0:
            print("[{0}/{1}] loaded".format(e,len(trainids)))
        if e % 200 == 0 and e > 100:
            save_stats(mean_root, e+1, sum_img, ov0, ov1, ov2)
            print("saving mean {0}".format(e))
    save_stats(mean_root, e+1, sum_img, ov0, ov1, ov2)

if __name__ == '__main__':
    main()