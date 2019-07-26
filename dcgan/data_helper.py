import cv2
import os
MAIN_PATH="./faces/"
def get_imgs():
    files = os.listdir(MAIN_PATH)
    imgs = []
    for file in files:
        imgs.append(cv2.imread(MAIN_PATH + file))
    print("get_imgs")
    return imgs
