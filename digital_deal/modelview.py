# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''use opencv3 to capture video frame, show and save its stream.'''

import cv2

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
    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), int(fps), (int(w), int(h)))

    # 判断是否正确获取VideoCapture类实例
    while fcap.isOpened():
        # 获取帧画面
        success, frame = fcap.read()
        while success:
            cv2.imshow("demo", frame)  ## 显示画面
            # 获取帧画面
            success,frame = fcap.read()

            # 保存帧数据
            writer.write(frame)

            if (cv2.waitKey(20) & 0xff) == ord('q'):  ## 等待20ms并判断是按“q”退出，相当于帧率是50hz，注意waitKey只能传入整数，
                break
        # 释放VideoCapture资源
        fcap.release()
    # 释放VideoWriter资源
    writer.release()
    cv2.destroyAllWindows()  ## 销毁所有opencv显示窗口


if __name__ == "__main__":
    stream_processing()