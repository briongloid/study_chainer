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
