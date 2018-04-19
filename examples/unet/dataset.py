import io
import os
import zipfile

import numpy as np
from PIL import Image
from chainer.dataset import download

def get_facade():
    root = download.get_dataset_directory('study_chainer/facade')
    npz_path = os.path.join(root, 'base.npz')
    url = 'http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip'

    def creator(path):
        archive_path = download.cached_download(url)

        images = []
        labels = []

        with zipfile.ZipFile(archive_path, 'r') as archive:
            for i in range(1, 378+1):
                image_name = 'base/cmp_b{:04d}.jpg'.format(i)
                label_name = 'base/cmp_b{:04d}.png'.format(i)

                image = Image.open(io.BytesIO(archive.read(image_name)))
                image = np.asarray(image)
                images.append(image)
                label = Image.open(io.BytesIO(archive.read(label_name)))
                label = np.asarray(label)
                labels.append(label)

        np.savez_compressed(path, images=images, labels=labels)
        return {'images': images, 'labels': labels}

    raw = download.cache_or_load_file(npz_path, creator, np.load)
    return raw['images'], raw['labels']

from chainer.dataset import DatasetMixin

def image_label_crop(image, label, crop_size):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    w, h = image.shape[1:3]
    cw, ch = crop_size
    left = np.random.randint(w-cw)
    top = np.random.randint(h-ch)
    right = left+cw
    bottom = top+ch
    
    return image[:, left:right, top:bottom], label[left:right, top:bottom]

class FacadeDataset(DatasetMixin):
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
        self.crop_size = 192
        
    def __len__(self):
        return len(self.images)
    
    def get_example(self, index):
        return image_label_crop(
            self.images[index], 
            self.labels[index], 
            self.crop_size)
    
def transfrom_images(images):
    ret = []
    for i in range(len(images)):
        image = images[i]
        image = image.astype(np.float32)/255.0
        image = image.transpose(2, 0, 1)
        ret.append(image)
    return ret

def transform_labels(labels):
    ret = []
    for i in range(len(labels)):
        label = labels[i]
        #label = np.identity(13)[label][:,:,1:]
        label = label.astype(np.int32)-1
        #label = label.transpose(2, 0, 1)
        ret.append(label)
    return ret