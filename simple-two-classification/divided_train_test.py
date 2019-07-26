# -*- coding:utf-8 -*-
import numpy as np 
import cv2
import tqdm
import os
import shutil


def rename(other_path, others_path):

    print('pics begin to be renamed ')

    if os.path.exists(others_path):
        shutil.rmtree(others_path)  
    os.mkdir(others_path)  

    for _,_,files in os.walk(other_path):
        files = files

    cnt = 0
    length = len(files)
    for index,f in enumerate(files):
        if index%500 == 0:
            print(index, '/', length)
        # ss = ("{:0>8d}.jpg".format(cnt))
        ss = '%08d' % cnt
        cnt += 1

        shutil.copyfile(other_path + f, others_path+ss)

    print('pics have been renamed ')


if __name__ =="__main__":
    face_path = './data/face/'
    other_path = './data/other/'
    others_path  = './data/others/'
    data_path = './dataset/'
    train_rate = 0.8

    if os.path.exists(data_path):
        shutil.rmtree(data_path)  
    os.mkdir(data_path)  

    if os.path.exists(others_path):
        for _,_,files in os.walk(other_path):
            other_files = files
        for _,_,files in os.walk(others_path):
            others_files = files
        if len(other_files) == len(others_files):
            print('pics have been renamed ')
        else:
            rename(other_path, others_path)
    else:
        rename(other_path, others_path)


    f_train = open(data_path+'train_gt.txt', 'w', encoding ='utf-8')
    f_test  = open(data_path+'test_gt.txt',  'w', encoding ='utf-8') 


    label_dict = {'face': face_path, 'other':others_path}
    label_all = ['face', 'other']

    # label_dict = {'other':other_path}
    # label_all = ['other']

    cnt = 0
    for label in label_all:
        for _,_,files in os.walk(label_dict[label]):
            files = files
        print('the length of ',label,  ' is ', len(files))
        path = label_dict[label]
        np.random.seed(1234)
        length = len(files)
        for index, f in enumerate(files):
            if index%500 == 0:
                print(index, '/', length)

            if np.random.rand() < train_rate:
                f_train.write(path + f +','+label+ '\n')
            else:
                f_test.write(path + f +','+label+'\n')

    f_train.close()
    f_test.close()
    print('goto check the result')









