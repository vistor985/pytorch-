from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


# 定义写字函数
def add_txt(image, size, draw_x, draw_y, txt):
    # 字体字号
    setFont = ImageFont.truetype('simhei.ttf', size)
    # 定义画板
    draw = ImageDraw.Draw(image)
    # 绘制
    draw.text((draw_x, draw_y), txt, font=setFont, fill=(0, 0, 0))
    return image


def make_fake_id_card():
    ori_image = cv2.imread('./make_card/IDCard.png')

    print('==ori_image.shape:', ori_image.shape)
    ori_image = cv2.resize(ori_image, (0, 0), fx=0.7, fy=0.7)
    print('==resize ori_image.shape:', ori_image.shape)

    # 向图片上写字
    img = Image.fromarray(cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB))
    img = add_txt(img, 18, 120, 67, '某某姓名')
    img = add_txt(img, 16, 120, 108, '男')
    img = add_txt(img, 16, 220, 108, '汉')
    img = add_txt(img, 16, 120, 144, '2020')
    img = add_txt(img, 16, 197, 144, '2')
    img = add_txt(img, 16, 237, 144, '26')
    img = add_txt(img, 16, 120, 185, '北京市哈哈哈哈')
    img = add_txt(img, 16, 120, 206, '某某　606')
    img = add_txt(img, 18, 197, 284, '987654321')

    cv2.imwrite('./make_card/word.jpg', np.array(img)[..., ::-1])


def make_white_mask():
    # 生成一个空白的模板mask
    ori_image = cv2.imread('./make_card/IDCard.png')
    ori_image = cv2.resize(ori_image, (0, 0), fx=0.4, fy=0.4)
    mask_image = np.ones_like(ori_image)
    mask_image *= 255
    print(mask_image.shape)
    cv2.imwrite('./make_card/mask.jpg', mask_image)

    # 往空白模板上写字(这里只能用PIL写，因为OpenCV写中文会乱码)
    img = Image.fromarray(cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))
    img = add_txt(img, 18, 90, 55, '某某姓名')
    img = add_txt(img, 16, 90, 87, '男')
    img = add_txt(img, 16, 190, 87, '汉')
    img = add_txt(img, 16, 90, 116, '2020')
    img = add_txt(img, 16, 160, 116, '2')
    img = add_txt(img, 16, 200, 116, '26')
    img = add_txt(img, 16, 90, 151, '北京市哈哈哈哈')
    img = add_txt(img, 16, 90, 172, '某某　606')
    img = add_txt(img, 18, 145, 234, '987654321')

    mask_image_txt = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite('./make_card/mask_image_txt.jpg', mask_image_txt)
    gray = cv2.cvtColor(mask_image_txt, cv2.COLOR_BGR2GRAY)
    # 高斯模糊，制造边缘模糊效果哦
    gray_Gaussianblur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 使用阈值对图片进行二值化
    th, res = cv2.threshold(gray_Gaussianblur, 200, 255, cv2.THRESH_BINARY)
    res_inv = cv2.bitwise_not(res)
    cv2.imwrite('./make_card/res_inv.jpg', res_inv)

    # 写字的模板保留文字部分
    img_bg = cv2.bitwise_and(mask_image_txt, mask_image_txt, mask=res_inv)
    cv2.imwrite('./make_card/img_bg.jpg', img_bg)
    # 原图保留除文字的其他部分
    img_fg = cv2.bitwise_and(ori_image, ori_image, mask=res)
    cv2.imwrite('./make_card/img_fg.jpg', img_fg)
    # 将两张图直接进行相加，即可
    final = cv2.add(img_bg, img_fg)
    cv2.imwrite('./make_card/final.jpg', final)


if __name__ == '__main__':
    make_fake_id_card()
    # make_white_mask()


