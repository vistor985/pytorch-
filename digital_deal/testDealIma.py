import cv2
import os
import PIL.Image as PImage
from PIL import ImageFont, ImageDraw
import numpy as np
import random
import glob

from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import *
from tkinter.messagebox import *

from numpy.random import sample


def changeBackground(img, img_back, zoom_size, center):
    # 缩放
    img = cv2.resize(img, zoom_size)
    rows, cols, channels = img.shape

    # 转换hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 获取mask
    #lower_blue = np.array([78, 43, 46])
    #upper_blue = np.array([110, 255, 255])
    diff = [5, 30, 30]
    gb = hsv[0, 0]
    lower_blue = np.array(gb - diff)
    upper_blue = np.array(gb + diff)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow('Mask', mask)

    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    # 粘贴
    for i in range(rows):
        for j in range(cols):
            if dilate[i, j] == 0:  # 0代表黑色的点
                img_back[center[0] + i, center[1] +
                         j] = img[i, j]  # 此处替换颜色，为BGR通道
    return img_back


def makedataset(avatar, im):
    # 摳圖
    avatar = cv2.cvtColor(np.asarray(avatar), cv2.COLOR_RGBA2BGRA)
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
    im = cv2.resize(im, (2480, 1816))  # 统一背景图片大小
    avalong = np.random.randint(826, 2480)
    avaheight = int(avalong*0.7)
    centerrow = np.random.randint(0, 1816-avaheight)
    centercow = np.random.randint(0, 2480-avalong)
    im = changeBackground(avatar, im, (avalong, avaheight),
                          (centerrow, centercow))
    # 因为抠图掉边缘的缘故，图片显示时不会贴到边界
    im = PImage.fromarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
    return im


# im = PImage.open(askopenfilename(initialdir=os.getcwd(), title=u'选择背景'))
# # 读贴图
# print("im.size:", im.size)
# fname = askopenfilename(initialdir=os.getcwd(), title=u'选择头像')
# fname = "../faceDataset/Data_Collection_face_part_1/SCUT-FBP-%d.jpg" % (
# i+1)
# print(fname)
# avatar = PImage.open(fname)  # 500x670

all_im_path = glob.glob('../background/*')
all_avatar_path = glob.glob('./data/imagesup/*')
all_avatar_path_down = glob.glob('./data/imagesdown/*')
for i in range(100):
    im_path = random.sample(all_im_path, 1)
    im_path = ''.join(im_path)
    im = PImage.open(im_path)

    ava_path = random.sample(all_avatar_path,1)
    ava_path = ''.join(ava_path)
    avatar = PImage.open(ava_path)
    img = makedataset(avatar, im)
    img.save("./lastDataup/"+"nor"+str(i)+".png")

    ava_path = random.sample(all_avatar_path_down,1)
    ava_path = ''.join(ava_path)
    avatar = PImage.open(ava_path)
    img = makedataset(avatar, im)
    img.save("./lastDatadown/"+"nor"+str(i)+".png")
    print("finish:%d" % i)
# im = makedataset(im,avatar)
# im.save('test.png')
