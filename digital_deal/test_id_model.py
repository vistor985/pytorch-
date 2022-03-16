import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from testReadIma import *
from sklearn.model_selection import train_test_split
from testReadIma import *


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.fc = nn.Linear(5408, 2)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        # print(x.size())  # 用于确定卷积转全连接后有多少列
        x = self.fc(x)
        return x


# model = Net()
model = torch.load('model/allmodel4.pth')
# model.load_state_dict(torch.load("./idmodel.pth.tar"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


def stream_processing():
    # 获取VideoCapture类实例，读取视频文件
    fcap = cv2.VideoCapture(0)

    # 设置摄像头分辨率的高
    fcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # 设置摄像头分辨率的宽
    fcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # 跳到某一感兴趣帧并从此帧开始读取,如从第360帧开始读取
    # fcap.set(cv2.CAP_PROP_POS_FRAMES, 360)

    # 获取视频帧的宽
    w = fcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 获取视频帧的高
    h = fcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 获取视频帧的帧率
    fps = fcap.get(cv2.CAP_PROP_FPS)
    # 获取视频流的总帧数
    # fcount = fcap.get(cv2.CAP_PROP_FRAME_COUNT)

    # 获取VideoWriter类实例
    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
        'X', 'V', 'I', 'D'), int(fps), (int(w), int(h)))

    # 判断是否正确获取VideoCapture类实例
    while fcap.isOpened():
        # 获取帧画面
        success, frame = fcap.read()
        while success:
            cv2.imshow("demo", frame)  # 显示画面
            # 获取帧画面
            success, frame = fcap.read()

            train_set = []
            img = cv2.resize(frame, (64, 64))  # 将图片大小调整为64*64
            img = transform(img)
            mat = np.array(img)  # 将图片转为numpy
            train_set.append(mat)
            train_set = np.array(train_set)
            img = torch.from_numpy(train_set)
            with torch.no_grad():
                img = img.to(device)
                outfinal = model(img)
                _, predictedfinal = torch.max(outfinal.data, dim=1)
                # print("final:", predictedfinal)
            # 保存帧数据
            if predictedfinal == torch.tensor([0], device='cuda:0'):
                cv2.putText(frame, "face:down", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "emblem:up", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            writer.write(frame)

            if (cv2.waitKey(20) & 0xff) == ord('q'):  # 等待20ms并判断是按“q”退出，相当于帧率是50hz，注意waitKey只能传入整数，
                break
        # 释放VideoCapture资源
        fcap.release()
    # 释放VideoWriter资源
    writer.release()
    cv2.destroyAllWindows()  # 销毁所有opencv显示窗口


stream_processing()

# finaltestpath = glob.glob("image/finaltest/*")
# finaltest = path_to_data(finaltestpath)
# finaltest = np.array(finaltest)
# finaltest = torch.from_numpy(finaltest)
# with torch.no_grad():
#     finaltest = finaltest.to(device)
#     outfinal = model(finaltest)
#     _, predictedfinal = torch.max(outfinal.data, dim=1)
#     print("final:", predictedfinal)
