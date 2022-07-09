import cv2
from cv2 import imshow
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

plt.style.use('dark_background')
img_ori = cv2.imread('1.jpg')
height, width, channel = img_ori.shape

##Maximize Contrast(Optional)
# structuringEleement = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringEleement)
# imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringEleement)
# imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
# gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

##그레이스케일 이미지로 변환
#hsv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HLS)
#gray = hsv[:,:,2]
gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

# plt.figure(figsize = (12,10))
# plt.imshow(img_ori, cmap='gray')

#blur and threshold
img_blurred = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
img_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)


#Find Contours
plate_cv, plate_cy=508.75, 300.5
width, height = 940, 626
plate_width_plate_height = 188.5

contoursData, hierarchy = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)
# 각각의 컨투의 갯수 출력 ---⑤
# print('도형의 갯수: (%d)'% (len(contoursData)))

img_contour = np.zeros((height, width, channel), dtype=np.uint8)
cv2.drawContours(img_thresh, contours=contoursData, contourIdx=-1, color=(255,255,255))
contours_dict=[]
for contour in contoursData:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(img_contour, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255), thickness =2)
    contours_dict.append({
        'contour':contour,
        'x':x,
        'y':y,
        'w':w,
        'h':h,
        'cx':x+(w/2),
        'cy':y+(h/2)
    })


MIN_AREA=80
MIN_WIDTH, MIN_HEIGHT=2,8
MIN_RATIO, MAX_RATIO=0.25, 1.0

posible_contours=[]
cnt=0
for d in contours_dict:
    area=d['w']*d['h']
    ratio=d['w']/d['h']
    
    if area>MIN_AREA and d['w']>MIN_WIDTH and d['h']>MIN_HEIGHT and MIN_RATIO<ratio<MAX_RATIO:
        d['idx']=cnt
        cnt+=1
        posible_contours.append(d)

img_PossibleContour = np.zeros((height, width, channel), dtype=np.uint8)
for d in posible_contours:
    cv2.rectangle(img_PossibleContour, pt1=(d['x'],d['y']), pt2=(d['x']+d['w'],d['y']+d['h']), color=(255,255,255), thickness =2)


MAX_DIAG_MULTIPLYER = 5
MAX_ANGLE_DIFF = 12.0
MAX_AREA_DIFF = 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3

def find_chars(contour_list):
    matched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue
            
            dx = abs(d1['cx'] - d2['cx']) #두 컨투어 중심좌표끼리의 가로길이
            dy = abs(d1['cy'] - d2['cy']) #두 컨투어 중심좌표끼리의 세로길이
            
            diagonal_length1 = np.sqrt(d1['w']**2 + d1['h']**2) #컨투어의 대각선 길이
            distance = np.linalg.norm(np.array([d1['cx'],d1['cy']])-np.array([d2['cx'],d2['cy']])) #두 컨투어의 대각선 중심점 사이의 길이
            if dx==0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy/dx))
            area_diff = abs(d1['w']*d1['h'] - d2['w']*d2['h']) / (d1['w']*d1['h'])
            width_diff = abs(d1['w'] - d1['h']) / d1['w']
            height_diff = abs(d1['w'] - d1['h']) / d1['h']
    
# cv2.imshow('CHAIN_APPROX_NONE', img_ori)
# cv2.imshow('CHAIN_APPROX_THRESH', img_thresh)
# cv2.imshow('CHAIN_APPROX_POSSIBLE', img_temp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plt.figure(figsize = (8,2))

plt.subplot(1,5,1)
plt.title('Original only')
plt.imshow(img_ori, cmap='gray')

plt.subplot(1,5,2)
plt.title('Blurred only')
plt.imshow(img_blurred, cmap='gray')

plt.subplot(1,5,3)
plt.title('Blur and Threshold')
plt.imshow(img_thresh, cmap='gray')

plt.subplot(1,5,4)
plt.title('Contour')
plt.imshow(img_contour, cmap='gray')

plt.subplot(1,5,5)
plt.title('Possible Contour')
plt.imshow(img_PossibleContour, cmap='gray')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()