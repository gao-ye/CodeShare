import numpy as np
import lmdb
import os 
import shutil
import cv2
def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print("the number of pic is ", nSamples)
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)  

    # env = lmdb.open(outputPath, map_size=1099511627776)
    env = lmdb.open(outputPath,map_size=1000000000)
    print("open the mdb success")
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = ('image-%09d' % cnt).encode()
        labelKey = ('label-%09d' % cnt).encode()
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache[('num-samples').encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
    print("data written into ", outputPath)


if __name__ == "__main__":
    train_gt = './dataset/train_gt.txt'
    test_gt  = './dataset/test_gt.txt'
    Label_gt = []
    Label_gt.append(train_gt)
    Label_gt.append(test_gt)

    train_output_path = './dataset/train/'
    test_output_path = './dataset/test/'
    Output_path = []
    Output_path.append(train_output_path)
    Output_path.append(test_output_path)

    for label_gt, output_path in zip(Label_gt, Output_path):
        ImagepathList = []
        LabelList     = []

        with open(label_gt) as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                line = line.strip('\n')
                content = line.split(',')
                ImagepathList.append(content[0])
                LabelList.append(content[1])
        createDataset(output_path, ImagepathList,  LabelList)







