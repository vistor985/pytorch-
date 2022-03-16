# coding:utf-8
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import cv2
import os
import PIL.Image as PImage
from PIL import ImageFont, ImageDraw
import numpy as np
import random
from dictionary import alphabet, nations
from address_set import province_set, city_set, couty_set
from dataAugmentation import augment

from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import *
from tkinter.messagebox import *


wordWidth = 70
wordHeight = 100
boxes = {'name': [400, 680, 1000, 780],
         'sex': [400, 820, 720, 920],
         'nation': [830, 820, 1380, 920],
         'sex_nation': [400, 820, 1380, 920],
         'birthday': [400, 960, 1330, 1060],
         'addr': [400, 1100, 1400, 1200],
         'idn': [400, 1450, 1800, 1550]}
images_output_path = './data/imagesup/'
txt_output_path = './data/annotations/'
dict_sum = len(alphabet) + 1

if getattr(sys, 'frozen', None):
    base_dir = os.path.join(sys._MEIPASS, 'usedres')
else:
    base_dir = os.path.join(os.path.dirname(__file__), 'usedres')


def IDcard_generator(amount):
    name_all = []
    sex_all = []
    nation_all = []
    addr_all = []
    year_all = []
    mon_all = []
    day_all = []
    id_all = []
    others = []

    numbers1 = '01'
    numbers2 = '012'
    numbers3 = '123456789'
    numbers = '0123456789'

    for i in range(amount):

        name_length = random.randint(2, 4)
        result = random.sample(alphabet, name_length)
        name_all.append(''.join(result))

        result = random.sample(u'女', 1)
        sex_all.append(''.join(result))

        result = random.sample(nations, 1)
        nation_all.append(''.join(result))

        result = random.sample(numbers, 1)
        result1 = random.sample('789', 1)
        result = list('19')+result1+result
        year_all.append(''.join(result))

        result1 = random.sample(numbers1, 1)
        if result1 == ['0']:
            result2 = random.sample(numbers3, 1)  # 01-09
        else:
            result2 = random.sample(numbers2, 1)  # 10-12
        result = result1+result2
        mon_all.append(''.join(result))

        result1 = random.sample(numbers2, 1)
        if result1 == ['0']:
            result2 = random.sample(numbers3, 1)
        else:
            result2 = random.sample(numbers, 1)
        result = result1+result2
        # result = random.sample(numbers, 2)
        day_all.append(''.join(result))

        id = []
        addr = []
        province = random.sample(province_set, 1)[0]
        # province = [u'天津市', 12]
        addr += province[0]
        if province[0] in city_set.keys():
            city = random.sample(city_set[province[0]], 1)[0]
            addr += city[0]
            if city[0][0] in couty_set.keys():
                couty = random.sample(couty_set[city[0]], 1)
                addr += couty[0]
                id += str(couty[1])
            else:
                id += str(city[1])
                if len(id) == 4:
                    id += random.sample(numbers, 2)
        else:
            id += str(province[1])
            if len(id) == 2:
                id += random.sample(numbers, 4)
        addr = ''.join(addr) # turn to str
        if len(addr)>11:
            addr = addr[:12]
        addr = addr + '公安局'
        # if len(addr) < 11:
        #     result = random.sample(alphabet, 11-len(addr))
        #     addr += ''.join(result)
        # elif len(addr) > 11:
        #     addr = addr[:12]
        id += year_all[i]
        id += mon_all[i]
        id += day_all[i]
        id += random.sample(numbers, 3)
        id += random.sample(u'0123456789X', 1)
        addr_all.append(''.join(addr))
        id_all.append(''.join(id))

    return name_all, sex_all, nation_all, year_all, mon_all, day_all, addr_all, id_all


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


def generator(num):
    global ename, esex, enation, eyear, emon, eday, eaddr, eidn
    images = []
    for i in range(num):
        name = ename[i]
        sex = esex[i]
        nation = enation[i]
        year = eyear[i]
        mon = emon[i]
        day = eday[i]
        addr = eaddr[i]
        idn = eidn[i]
        life = year+"."+mon+"."+day
        lifeend = str(int(year)+20)+"."+mon+"."+day
        life = life+"-"+lifeend
        im = PImage.open(os.path.join(base_dir, 'back.png'))
        # 读贴图
        # fname = askopenfilename(initialdir=os.getcwd(), title=u'选择头像')
        fname = "../faceDataset/Data_Collection_face_part_1/SCUT-FBP-%d.jpg" % (
            i+1)
        # print(fname)
        avatar = PImage.open(fname)  # 500x670
        # 字体样式
        name_font = ImageFont.truetype(os.path.join(base_dir, 'hei.ttf'), 72)
        other_font = ImageFont.truetype(os.path.join(base_dir, 'hei.ttf'), 60)
        bdate_font = ImageFont.truetype(
            os.path.join(base_dir, 'fzhei.ttf'), 60)
        id_font = ImageFont.truetype(
            os.path.join(base_dir, 'ocrb10bt.ttf'), 72)

        draw = ImageDraw.Draw(im)
        # draw.text((630, 690), name, fill=(0, 0, 0), font=name_font)
        # draw.text((630, 840), sex, fill=(0, 0, 0), font=other_font)
        # draw.text((1030, 840), nation, fill=(0, 0, 0), font=other_font)
        # draw.text((630, 980), year, fill=(0, 0, 0), font=bdate_font)
        # draw.text((950, 980), mon, fill=(0, 0, 0), font=bdate_font)
        # draw.text((1150, 980), day, fill=(0, 0, 0), font=bdate_font)
        start = 0
        # loc = 1120
        # while start + 11 < len(addr):
        #     draw.text((630, loc), addr[start:start + 11],
        #               fill=(0, 0, 0), font=other_font)
        #     start += 11
        #     loc += 100
        # draw.text((630, loc), addr[start:], fill=(0, 0, 0), font=other_font)
        # draw.text((950, 1475), idn, fill=(0, 0, 0), font=id_font)

        draw.text((1050, 960), addr[start:], fill=(0, 0, 0), font=other_font)
        draw.text((1050, 1100), life, fill=(0, 0, 0), font=other_font)
        # # 抠图
        # avatar = cv2.cvtColor(np.asarray(avatar), cv2.COLOR_RGBA2BGRA)
        # im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
        # im = changeBackground(avatar, im, (500, 670), (690, 1500))
        # im = PImage.fromarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
        # im.save('i' + 'color.png')
        # im.convert('L').save(output_path + 'bw.png')
        # images.append(im.convert('L')) #转灰度图片
        images.append(im)
        if (i+1) % 100 == 0:
            print('Generate images: {}/{}'.format(i+1, num))

    return images


def fragment_IDcard_save(images, augmented=False, batch_name=None):  # 碎片化存储
    global ename, esex, enation, eyear, emon, eday, eaddr, eidn
    txt_out = []
    num = len(images)
    if augmented:
        print('Output augmented data to {} and {}'.format(
            images_output_path, txt_output_path))
    else:
        print('Output data to {} and {}'.format(
            images_output_path, txt_output_path))
    for i in range(num):
        name = u'姓名' + ename[i]
        sex = u'性别' + esex[i]
        nation = u'民族' + enation[i]
        sex_nation = sex + nation
        birthday = u'出生' + eyear[i] + u'年' + emon[i] + u'月' + eday[i] + u'日'
        addr = u'住址' + eaddr[i]
        idn = u'公民身份证号码' + eidn[i]

        im = images[i]
        # name
        result = im.crop(boxes['name'])
        if augmented:
            result = augment(result)
        result.save(images_output_path + batch_name + str(i) + '_name.png')
        label = batch_name + str(i) + '_name.png ' + name + '\n'
        txt_out.append(label)
        if np.random.randint(0, 3) > 0:
            # sex
            result = im.crop(boxes['sex'])
            if augmented:
                result = augment(result)
            result.save(images_output_path + batch_name + str(i) + '_sex.png')
            label = batch_name + str(i) + '_sex.png ' + sex + '\n'
            txt_out.append(label)
            # nation
            result = im.crop(boxes['nation'])
            if augmented:
                result = augment(result)
            result.save(images_output_path + batch_name +
                        str(i) + '_nation.png')
            label = batch_name + str(i) + '_nation.png ' + nation + '\n'
            txt_out.append(label)
        else:
            # sex and nation
            result = im.crop(boxes['sex_nation'])
            if augmented:
                result = augment(result)
            result.save(images_output_path + batch_name +
                        str(i) + '_sex_nation.png')
            label = batch_name + \
                str(i) + '_sex_nation.png ' + sex_nation + '\n'
            txt_out.append(label)
        # birthday
        result = im.crop(boxes['birthday'])
        if augmented:
            result = augment(result)
        result.save(images_output_path + batch_name + str(i) + '_birthday.png')
        label = batch_name + str(i) + '_birthday.png ' + birthday + '\n'
        txt_out.append(label)
        # address
        result = im.crop(boxes['addr'])
        if augmented:
            result = augment(result)
        result.save(images_output_path + batch_name + str(i) + '_addr.png')
        label = batch_name + str(i) + '_addr.png ' + addr + '\n'
        txt_out.append(label)
        # ID number
        result = im.crop(boxes['idn'])
        if augmented:
            result = augment(result)
        result.save(images_output_path + batch_name + str(i) + '_idn.png')
        label = batch_name + str(i) + '_idn.png ' + idn + '\n'
        txt_out.append(label)

        if (i+1) % 100 == 0:
            print('Output images: {}/{}'.format(i+1, num))

    with open(txt_output_path + 'data.txt', 'w') as f:
        for line in txt_out:
            f.write(line)


def IDcard_save(images, batch_name=None):
    global ename, esex, enation, eyear, emon, eday, eaddr, eidn
    txt_out = []
    num = len(images)
    print('Output data to {} and {}'.format(
        images_output_path, txt_output_path))
    for i in range(num):
        # name = u'姓名' + ename[i] + '\n'
        # sex = u'性别' + esex[i]
        # nation = u'民族' + enation[i]
        # sex_nation = sex + nation + '\n'
        # birthday = u'出生' + eyear[i] + u'年' + \
        #     emon[i] + u'月' + eday[i] + u'日' + '\n'
        # addr = u'住址' + eaddr[i] + '\n'
        # idn = u'公民身份证号码' + eidn[i] + '\n'

        # txt_out.append(batch_name + str(i) + '.png' + '\n')
        # txt_out.append(name + sex_nation + birthday + addr + idn)
        images[i].save(images_output_path + batch_name + str(i) + '.png')

        if (i+1) % 100 == 0:
            print('Output images: {}/{}'.format(i+1, num))

    # with open(txt_output_path + 'data.txt', 'w', encoding='utf-8') as f:  # 加入了utf-8
    #     for line in txt_out:
    #         f.write(line)  # gbk error


if __name__ == '__main__':
    global ename, esex, enation, eyear, emon, eday, eaddr, eidn
    fragment_IDcard = False
    sample_sum = 50
    print('--- Randomly Generate Content ---')
    ename, esex, enation, eyear, emon, eday, eaddr, eidn = IDcard_generator(
        sample_sum)
    print('--- Generate ID Card ---')
    images = generator(num=sample_sum)
    if fragment_IDcard:  # 碎片化存儲
        print('--- Fragment ID card ---')
        fragment_IDcard_save(images, augmented=True, batch_name='0_')
    else:
        print('--- ID card ---')
        IDcard_save(images, batch_name='1_')
    print('--- Generate Database Successfully'.format(sample_sum))
