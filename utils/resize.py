import os
import cv2
from tqdm import tqdm
import numpy as np

DATAPATH = "/media/TOSHIBA6T/ICC2022/dataset/valid/"
NEWPATH = "/media/TOSHIBA6T/ICC2022/dataset/valid_resized/"
NEW_HEIGHT = 224

if not os.path.exists(NEWPATH):
    os.makedirs(NEWPATH)
    
min_h = np.inf
min_w = np.inf
max_h = 0
max_w = 0

for f in tqdm(os.listdir(DATAPATH)):
    if(f.endswith(".jpg") or f.endswith('.png')):
        
        img = cv2.imread(os.path.join(DATAPATH, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        if h < min_h:
            min_h = h
        if w < min_w:
            min_w = w

        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w
        
        ratio = w / h
        new_w = int(np.ceil(NEW_HEIGHT * ratio))
        new_img = cv2.resize(img, (new_w, NEW_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(NEWPATH, f), new_img)

print(min_h, min_w)
print(max_h, max_w)