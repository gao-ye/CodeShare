from __future__ import print_function
import argparse
import random
import math
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
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
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--test_1', required=True, help='path to dataset')
parser.add_argument('--test_2', required=False, help='path to dataset')
parser.add_argument('--test_3', required=False, help='path to dataset')
parser.add_argument('--test_4', required=False, help='path to dataset')
parser.add_argument('--test_5', required=False, help='path to dataset')
parser.add_argument('--test_6', required=False, help='path to dataset')
parser.add_argument('--test_7', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=128, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=400, help='the width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--MODEL', default='', help="path to model (to continue training)")
parser.add_argument('--n_bm', type=int, default=3, help='number of n_bm')
parser.add_argument('--testAngle', type=int, default=5, help='testAngle')
parser.add_argument('--alphabet', type=str, default='0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " \' # $ % & ( ) * + , - . / : ; < = > ? @ [ \\ ] _ ` ~')
parser.add_argument('--alphabet1', type=str, default='0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z')
parser.add_argument('--alphabet2', type=str, default='a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z')
parser.add_argument('--sep', type=str, default=' ')

opt = parser.parse_args()
print(opt)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

test_dataset1 = dataset.lmdbDataset( opt.alphabet1,test=True,root=opt.test_1, 
    transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
if opt.test_2!=None:
   test_dataset2 = dataset.lmdbDataset( opt.alphabet1,test=True,root=opt.test_2, 
       transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
if opt.test_3!=None:
   test_dataset3 = dataset.lmdbDataset( opt.alphabet1,test=True,root=opt.test_3, 
       transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
if opt.test_4!=None:
   test_dataset4 = dataset.lmdbDataset( opt.alphabet,test=True,root=opt.test_4, 
       transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),ifRotate90=True)
   test_dataset4_1 = dataset.lmdbDataset( opt.alphabet1,test=True,root=opt.test_4, 
       transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),ifRotate90=True)
if opt.test_5!=None:
   test_dataset5 = dataset.lmdbDataset( opt.alphabet1,test=True,root=opt.test_5, 
       transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),ifRotate90=True)
if opt.test_6!=None:
   test_dataset6 = dataset.lmdbDataset( opt.alphabet1,test=True,root=opt.test_6, 
       transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),ifRotate90=True)
if opt.test_7!=None:
   test_dataset7 = dataset.lmdbDataset( opt.alphabet1,test=True,root=opt.test_7, 
       transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),ifRotate90=True)

nclass = len(opt.alphabet.split(opt.sep))
converter = utils.strLabelConverterForAttention(opt.alphabet, opt.sep)
criterion = torch.nn.CrossEntropyLoss()

MODEL = MODEL(opt.n_bm, nclass)

if opt.MODEL != '':
    print('loading pretrained model from %s' % opt.MODEL)
    state_dict = torch.load(opt.MODEL)
    MODEL_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        MODEL_state_dict_rename[name] = v
    MODEL.load_state_dict(MODEL_state_dict_rename, strict=True)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgW)
text = torch.LongTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    MODEL.cuda()
    MODEL = torch.nn.DataParallel(MODEL, device_ids=range(1))
    image = image.cuda()
    text = text.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)


toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()


def adjust_data(cam,img,w,h):
    cam_h,cam_w = 4,13 #cam.shape

    im_mean = [0.486, 0.459, 0.408]
    im_std = [0.229, 0.224, 0.225]
    img = img * np.array(im_std).astype(float)
    img = img + np.array(im_mean)
    img = img * 255.
    img = np.clip(img,0,255)
    
    cam =  (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = np.uint8(255 * cam)
    cam = cam.reshape(cam_h,cam_w)
    heatmap = cv2.applyColorMap(cv2.resize(cam, (w, h)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.6
    return result

count = 0
def val_beam(dataset ,max_iter=1000, testAngle=0):
    rotate90 = dataset.ifRotate90
    
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers)) # opt.batchSize
    val_iter = iter(data_loader)
    max_iter = min(max_iter, len(data_loader))
    n_correct = 0
    n_total = 0

    n_head = 16

    for i in range(max_iter):
        data = val_iter.next()
        ori_cpu_images = data[0]
        cpu_texts = data[1]
        flag_rotate90 = data[2]
        
        t, l = converter.encode(cpu_texts, scanned=True)
        utils.loadData(text, t)
        utils.loadData(length, l)
        All_preds_add5EOS = []
        All_scores = []
        acc_index = []
        
        # # %%% Multi Angle %%%        
        # if testAngle==0:
        #    angles = [0]
        # else:
        #    angles = [-testAngle, 0, testAngle]

        angles = [0]
        cpu_images = ori_cpu_images
        utils.loadData(image, cpu_images)
        
        local_preds,local_scores = MODEL(image, length, text, test=True, cpu_texts=cpu_texts)
        
        All_preds_add5EOS.append(local_preds)
        All_scores.append(local_scores)

        # Start to decode
        preds_add5EOS = []
        for j in range(cpu_images.size(0)):
            text_begin = 0 if j == 0 else (length.data[:j].sum()+j*5)
            
            max_score = -99999
            max_index = 0
            for index in range(len(All_scores)):
                local_score = All_scores[index][j]
                if local_score > max_score:
                   max_score = local_score
                   max_index = index
                   
            preds_add5EOS.extend(All_preds_add5EOS[max_index][text_begin:text_begin+int(length[j].data)+5])
        preds_add5EOS = torch.stack(preds_add5EOS)
        
        sim_preds_add5eos = converter.decode(preds_add5EOS.data, length.data + 5)
                
        cnt = 0
        for pred, target in zip(sim_preds_add5eos, cpu_texts):
            pred = pred.split(opt.sep)[0]+opt.sep
            test_alphabet = dataset.alphabet.split(opt.sep)
            pred = ''.join(pred[i].lower() if pred[i].lower() in test_alphabet else '' for i in range(len(pred)))
            target = ''.join(target[i].lower() if target[i].lower() in test_alphabet else '' for i in range(len(target)))
            
            if pred.lower() == target.lower():
                n_correct += 1
                acc_index.append(cnt)
                print("acc_index.append", cnt)
            n_total += 1
            cnt += 1


        ## begin to get the right pic attention
        global count 
        break_num = 10000
        if count < break_num: 
            seq_stacked, slf_attns, enc_attns = MODEL(image, length, text, test=False, cpu_texts=cpu_texts, get_attention=True)
            enc_attns = enc_attns[0]

            N,_,H,W = cpu_images.shape

            enc_attns = enc_attns.cpu().numpy()
            for t in range(N):
                if count <= break_num and t in acc_index :
                    count += 1
                    # try:
                    for j in range(n_head):
                        if j == 0:
                            attns =  enc_attns[t*n_head + j]
                        else:
                            attns +=  enc_attns[t*n_head + j]
                
                    attns = attns / n_head
                    print("the attention map index is ", t)
                    # print(cpu_images[i][0].shape)
                    img = np.repeat(cpu_images[t][0].numpy(),3).reshape(H,W,3)


                    for char in range(length[t]):
                        att = attns[char]
                        result = adjust_data(att, img, W, H)
                        cv2.imwrite('imgs/img_%d_char_%d.jpg'%(count,char), result)
                    print('imgs/img_%d_.jpg'%(count), "write into disk success")
                    # except:
                    #     print("there is a erroe,  and skipping it")
    accuracy = n_correct / float(n_total)
    dataset_name = dataset.root.split('/')[-1]
    print(dataset_name+' ACCURACY -----> %f' % (accuracy))
    return accuracy



import os
try:
    os.system("rm -rf imgs")
except:
    print("there is no imgs")
os.system("mkdir imgs")


for p in MODEL.parameters():
    p.requires_grad = False
MODEL.eval()
print('================= Start val (beam size:'+str(opt.n_bm)+' + testAngle:'+str(opt.testAngle)+') ===================')

acc_tmp1 = val_beam(test_dataset1, testAngle=opt.testAngle)
# acc_tmp2 = val_beam(test_dataset2, testAngle=opt.testAngle)
# acc_tmp3 = val_beam(test_dataset3, testAngle=opt.testAngle)
# acc_tmp4 = val_beam(test_dataset4, testAngle=opt.testAngle)
# acc_tmp4_1 = val_beam(test_dataset4_1, testAngle=opt.testAngle)
# acc_tmp5 = val_beam(test_dataset5, testAngle=opt.testAngle)
# acc_tmp6 = val_beam(test_dataset6, testAngle=opt.testAngle)
# acc_tmp7 = val_beam(test_dataset7, testAngle=opt.testAngle)