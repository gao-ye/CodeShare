import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import sampler
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import cv2
class lmdbDataset(Dataset):
# torch.utils.data 作用: (1) 创建数据集,有__getitem__(self, index)函数
# 来根据索引序号获取图片和标签, 有__len__(self)函数来获取数据集
    def __init__(self, test=False, root=None, transform=None, reverse=False, ifRotate90=False):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.dict ={'face':0, 'other':1}

        self.transform = transform
        self.reverse = reverse
        self.test = test
        self.root = root
        self.ifRotate90 = ifRotate90

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')  
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]            

            label_key = 'label-%09d' % index
            label = (txn.get(label_key.encode())).decode()
            label = self.dict[label]

            if self.transform is not None:
                img = self.transform(img,test=self.test)

        return (img, label)


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        # self.crop = transforms.RandomCrop((size[1],size[0]))        
        # self.pre_pro = transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05)
        self.toTensor = transforms.ToTensor()

    def __call__(self, img, test=False):

        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)