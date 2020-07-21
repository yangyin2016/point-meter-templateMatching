import cv2
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from math import cos, pi, sin
from time import time

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
method = cv2.TM_CCOEFF

center=[121 , 116]
a={20:( 52,189 ) ,30:( 34,168 ) ,40: ( 24,144 ),50:( 22,103 ) ,
   60:( 40,60 ) ,70:( 90,25 ) ,80:( 166,31 ) ,90:( 218,90 ) ,100:(193,186)}
count=0
result={}
for k ,v in a.items():
    r=math.acos((v[0]-center[0])/((v[0]-center[0])**2 + (v[1]-center[1])**2)**0.5)
    r=r*180/math.pi
    a[k]=r
    if count >= 4 and k != 100:
        r=360-r
        # print(k, r)
    result[k]=r
    count+=1
d=360-result[90]+result[100]
d1=360-result[90]
t=90+10*(d1/d)
result[t]=0
result_list=result.items()
lst=sorted(result_list,key=lambda x:x[1])
def get_next(c):
    l=len(lst)
    n=0
    for i in range(len(lst)):
        if lst[i][0]==c:
            n=i+1
            if n==l:
                n=0
            break
    return lst[n]

def get_rad_val(rad):
    old=None
    for k, v in lst:
        # print(k,v)
        if rad > v :
            old = k
    #print(old)
    r=result[old]
    d=rad-r
    nx=get_next(old)
    #print(10*abs(d/(nx[1] - r)))
    #print(nx)
    t=old+10*abs(d/(nx[1] - r))
    #print(t)
    return t

def normalized_picture(img):
    y, x = img.shape[:2]
    y_s = 422
    x_s = x * y_s / y
    x_x = int(x_s)
    crop_size = (x_x, y_s)
    nor = cv2.resize(img, crop_size, interpolation=cv2.INTER_LINEAR)
    #nor = cv2.resize(img, None, fx = 1, fy = 1, interpolation=cv2.INTER_LINEAR)
    y, x = nor.shape[:2]
    print("图片的长和宽为：", x, y)
    return nor

def get_match_rect(template,img,method):
    '''获取模板匹配的矩形的左上角和右下角的坐标'''
    w, h = template.shape[1],template.shape[0]
    res = cv2.matchTemplate(img, template, method)
    mn_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的方法，对结果的解释不同
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left,bottom_right

def get_center_point(top_left,bottom_right):
    '''传入左上角和右下角坐标，获取中心点'''
    c_x, c_y = ((np.array(top_left) + np.array(bottom_right)) / 2).astype(np.int)
    return c_x,c_y

def get_circle_field_color(img,center,r,thickness):
    '''获取中心圆形区域的色值集'''
    temp=img.copy().astype(np.int)
    cv2.circle(temp,center,r,-100,thickness=thickness)
    return img[temp == -100]

def v2_by_center_circle(img,colors):
    '''二值化通过中心圆的颜色集合'''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a = img[i, j]
            if a in colors:
                img[i, j] = 0
            else:
                img[i, j] = 255

def v2_by_k_means(img):
    '''使用k-means二值化'''
    original_img = np.array(img, dtype=np.float64)
    src = original_img.copy()
    delta_y = int(original_img.shape[0] * (0.4))
    delta_x = int(original_img.shape[1] * (0.4))
    original_img = original_img[delta_y:-delta_y, delta_x:-delta_x]
    h, w, d = src.shape
    #print(w, h, d)
    dts = min([w, h])
    #print(dts)
    r2 = (dts / 2) ** 2
    c_x, c_y = w / 2, h / 2
    a: np.ndarray = original_img[:, :, 0:3].astype(np.uint8)
    # 获取尺寸(宽度、长度、深度)
    height, width = original_img.shape[0], original_img.shape[1]
    depth = 3
    #print(depth)
    image_flattened = np.reshape(original_img, (width * height, depth))
    '''
    用K-Means算法在随机中选择1000个颜色样本中建立64个类。
    每个类都可能是压缩调色板中的一种颜色。
    '''
    image_array_sample = shuffle(image_flattened, random_state=0)
    estimator = KMeans(n_clusters=2, random_state=0)
    estimator.fit(image_array_sample)
    '''
    我们为原始图片的每个像素进行类的分配。
    '''
    src_shape = src.shape
    new_img_flattened = np.reshape(src, (src_shape[0] * src_shape[1], depth))
    cluster_assignments = estimator.predict(new_img_flattened)
    '''
    我们建立通过压缩调色板和类分配结果创建压缩后的图片
    '''
    compressed_palette = estimator.cluster_centers_
    #print(compressed_palette)
    a = np.apply_along_axis(func1d=lambda x: np.uint8(compressed_palette[x]), arr=cluster_assignments, axis=0)
    img = a.reshape(src_shape[0], src_shape[1], depth)
    #print(compressed_palette[0, 0])
    threshold = (compressed_palette[0, 0] + compressed_palette[1, 0]) / 2
    img[img[:, :, 0] > threshold] = 255
    img[img[:, :, 0] < threshold] = 0
    #cv2.imshow('sd0', img)
    for x in range(w):
        for y in range(h):
            distance = ((x - c_x) ** 2 + (y - c_y) ** 2)
            if distance > r2:
                pass
                img[y, x] = (255, 255, 255)
    cv2.imshow('kmeans', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img

def get_pointer_rad(img):
    '''获取角度'''
    shape = img.shape
    c_y, c_x, depth = int(shape[0] / 2), int(shape[1] / 2), shape[2]
    x1=c_x+c_x*0.8
    src = img.copy()
    freq_list = []
    for i in range(1800):
        x = (x1 - c_x) * cos(i * pi / 900) + c_x
        y = (x1 - c_x) * sin(i * pi / 900) + c_y
        temp = src.copy()
        cv2.line(temp, (c_x, c_y), (int(x), int(y)), (0, 0, 255), thickness=3)
        t1 = img.copy()
        t1[temp[:, :, 2] == 255] = 255
        c = img[temp[:, :, 2] == 255]
        points = c[c == 0]
        i = i / 5
        freq_list.append((len(points), i))
        #cv2.imshow('d0', temp)
        cv2.imshow('zhixian', t1)
        cv2.waitKey(1)
    print('重合数量和对应角度:',max(freq_list, key=lambda x: x[0]))
    #cv2.destroyAllWindows()
    return max(freq_list, key=lambda x: x[0])


if __name__ == '__main__':
    #for x in range(1,32):
        #获取测试图像

        img_s = cv2.imread('0.jpg')
        img=cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
        t1 = time()
        template = cv2.imread('template2.png')
        template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        #匹配并返回矩形坐标
        top_left,bottom_right=get_match_rect(template,img,method)
        c_x,c_y=get_center_point(top_left,bottom_right)
        #print(c_x,c_y)
        #绘制矩形
        cv2.rectangle(img_s, top_left, bottom_right, 255, 2)
        cv2.imshow('yuantu',cv2.resize(img_s,(int(img.shape[1]*0.5),int(img.shape[0]*0.5))))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        new = img_s[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        template = cv2.imread('template3.png')
        top_left, bottom_right = get_match_rect(template, new, method=method)
        new_ = new[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        # 二值化图像
        cv2.imshow('jietu',new_)
        nor = normalized_picture(new_)
        img=v2_by_k_means(nor)
        rad=get_pointer_rad(img)
        print('刻度 =', get_rad_val(rad[1]))

        cv2.waitKey(0)