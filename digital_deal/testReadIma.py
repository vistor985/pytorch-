from torchvision import transforms
import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split  # 划分训练集和验证集

transform = transforms.Compose([
    # transforms.Resize(28),  # mnist为28*28,此处此句会因opencv(h,w,c)与tensor(c,w,h)图片存储格式不同对后续造成影响
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def path_to_data(all_ima_path):
    train_set = []
    for path in all_ima_path:
        img = cv2.imread(path)  # 读取图片
        img = cv2.resize(img, (64, 64))  # 将图片大小调整为64*64
        img = transform(img)
        mat = np.array(img)  # 将图片转为numpy
        train_set.append(mat)
    return train_set  # 返回存储所有像素数据的矩阵


def test_read_data():
    train_ima_path = glob.glob("image/ima_up/*")
    train_set_up = path_to_data(train_ima_path)
    print("len(train_set_up):", len(train_set_up))
    up_label = np.ones(len(train_set_up))
    train_ima_path = glob.glob("image/ima_down/*")
    train_set_down = path_to_data(train_ima_path)
    print("len(train_set_down):", len(train_set_down))
    down_label = np.zeros(len(train_set_down))
    train_set_up = np.array(train_set_up)
    train_set_down = np.array(train_set_down)
    dataset = np.vstack([train_set_up, train_set_down])
    label = np.hstack((up_label, down_label))
    return dataset, label
