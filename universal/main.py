# -*- coding:utf-8 -*-

from __future__ import print_function
import argparse
import random
import math
import torch
import torch.backends.cudnn as cudnn
import torch.argsim as argsim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os,pdb
import tools.utils as utils
import tools.dataset as dataset
import time
from collections import OrderedDict
from models.model import MODEL
# from tools.utils import adjust_lr_exp
from PIL import Image


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_1', required=True, help='path to dataset')
	parser.add_argument('--train_2', required=False, help='path to dataset')
	parser.add_argument('--train_3', required=False, help='path to dataset')
	parser.add_argument('--test_1', required=True, help='path to dataset')
	parser.add_argument('--test_2', required=False, help='path to dataset')
	parser.add_argument('--test_3', required=False, help='path to dataset')
	parser.add_argument('--test_4', required=False, help='path to dataset')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
	parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
	parser.add_argument('--imgH', type=int, default=128, help='the height of the input image to network')
	parser.add_argument('--imgW', type=int, default=400, help='the width of the input image to network')
	parser.add_argument('--random_seed', type=int, default=0, help='value of random seed')
	parser.add_argument('--alphabet', type=str, default='0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " \' # $ % & ( ) * + , - . / : ; < = > ? @ [ \\ ] _ ` ~')

	args = parser.parse_args()
	return args

def Load_train_data(args):
	# Train data
	train_dataset_1 = dataset.lmdbDataset( args.alphabet,root=args.train_1, 
		transform=dataset.resizeNormalize((args.imgW, args.imgH)))
	assert train_dataset_1
	train_dataset = train_dataset_1
	
	if args.train_2!=None:
	train_dataset_2 = dataset.lmdbDataset( args.alphabet,root=args.train_2, 
		transform=dataset.resizeNormalize((args.imgW, args.imgH)))
	assert train_dataset_2
	train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_2])

	if args.train_3!=None:
	train_dataset_3 = dataset.lmdbDataset( args.alphabet,root=args.train_3, 
		transform=dataset.resizeNormalize((args.imgW, args.imgH)))
	assert train_dataset_3
	train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_3])
# 该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
	# train_loader = torch.utils.data.DataLoader(
	# 	train_dataset, batch_size=args.batchSize,
	# 	shuffle=False,sampler=dataset.randomSequentialSampler(train_dataset, args.batchSize),
	# 	num_workers=int(args.workers))
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batchSize,
		shuffle=False,
		num_workers=int(args.workers))

	return train_loader

def Load_test_data(dataset_name):
	dataset = dataset.lmdbDataset( args.alphabet1,test=True,root=dataset_name, 
		transform=dataset.resizeNormalize((args.imgW, args.imgH)))

	test_loader = torch.utils.data.DataLoader(
			dataset, shuffle=False, batch_size=args.batchSize, num_workers=int(args.workers))

	return test_loader


def set_random_seed(random_seed):
	random.seed(random_seed)
	np.random.seed(random_seed)
	torch.manual_seed(random_seed)
	# print(random.seed)


def train_batch():
    data = train_iter.next()
    cpu_images = data[0]
    cpu_labels = data[1]
    
    utils.loadData(image, cpu_images)
	utils.loadData(ori_label, cpu_labels)

    label = utils.label_convert(ori_label, nclass) ## 进行 one-hot 编码
	print("label size", label.shape)

    preds = MODEL(image)
    cost_pred = criterion(preds, label)
    cost = cost_pred
 
    loss_avg.add(cost)
    
    MODEL.zero_grad()
    cost.backward()
    optimizer.step()


def val(test_loader):
	test_iter = iter(test_loader)

	max_iter = len(data_loader)
	n_correct = 0
	n_total = 0

	for i in range(max_iter):
		data = test_iter.next()
		cpu_images = data[0]
		cpu_labels = data[1]
		[num] = cpu_labels.shape

		utils.loadData(image, cpu_images)
		preds = MODEL(image)

		arg_max = preds.argmax(1).cpu.numpy()
		labels = cpu_labels.numpy()
		correct = np.sum(arg_max == labels)

		n_correct += correct
		n_total   += num

	acc = n_correct / float(n_total)
	return acc



if __name__ =="__main__":
	args = parse_args()
	print(args)
	set_random_seed(args.random_seed)
	train_loader = Load_train_data(args)

	if args.test_1!=None:	
		test_loader1 = Load_train_data(args.test_1)
	if args.test_2!=None:	
		test_loader2 = Load_train_data(args.test_2)
	if args.test_3!=None:	
		test_loader3 = Load_train_data(args.test_3)

	nclass = len(args.alphabet.split(args.sep))
	criterion = torch.nn.CrossEntropyLoss()

	MODEL = MODEL(nclass)

	image = torch.FloatTensor(args.batchSize, 1, args.imgH, args.imgW)
	ori_label = torch.IntTensor(args.batchSize)
	label = torch.IntTensor(args.batchSize, nclass)

	if args.cuda:
		MODEL.cuda()
		MODEL = torch.nn.DataParallel(MODEL, device_ids=range(args.ngpu))
		image = image.cuda()
		ori_label = ori_label.cuda()
		label = label.cuda()
		criterion = criterion.cuda()
	# loss averager
	loss_pred = utils.averager()

	optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,MODEL.parameters()), lr=args.lr)

	for epoch in range(args.niter):

		train_iter = iter(train_loader)
		i = 0
		while i < len(train_loader):
			
			# adjust_lr_exp(argsimizer, args.lr, i+len(train_loader)*epoch, len(train_loader)*args.niter)
			

			# show the result on val dataset 
			# the result include acc
			if i % args.valInterval == 0 and epoch+float(i)/len(train_loader)>=args.val_start_epoch:
				for p in MODEL.parameters():
					p.requires_grad = False
				MODEL.eval()
				print('=============== Start Test ===============')
				acc_tmp1 = 0
				acc_tmp2 = 0
				acc_tmp3 = 0
				
				acc_tmp1 = val(test_loader1)
				if args.test_2!=None:
				acc_tmp2 = val(test_loader2)
				if args.test_3!=None:
				acc_tmp3 = val(test_loader3)

			for p in MODEL.parameters():
				p.requires_grad = True
			MODEL.train()

			train_batch()
			
			## show the result on train dataset  while after how many batch 
			#  the result include acc and loss
			if i % args.displayInterval == 0 and i!=0:
				t1 = time.time()            
				print ('Epoch: %d/%d; iter: %d/%d;  Loss: %.2f; time: %.2f s;' %
						(epoch, args.niter, i, len(train_loader),  loss_avg.val(), t1-t0)),

				loss_avg.reset()
				t0 = time.time()
				torch.save(MODEL.state_dict(), '{0}/latest.pth'.format(args.experiment))
			i += 1
		

	