import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# read sample image
#enter your path
image_path = str(r"C:\Users\vangu\PycharmProjects\Acne-detection-wth-TensorFlow\src\Acne04-Detection-5\train\levle0_0_jpg.rf.2809c8a0d7e1779910e9975cd0f98b71.jpg")
print(f"image path : {image_path}")
sample_img = cv.imread(image_path, cv.IMREAD_COLOR)
# plt.imshow(cv.cvtColor(sample_img, cv.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()

anno_path = r"C:\Users\vangu\PycharmProjects\Acne-detection-wth-TensorFlow\src\Acne04-Detection-5\train\_annotations.csv"
anno = pd.read_csv(anno_path)
sample_img_name = "levle0_0_jpg.rf.2809c8a0d7e1779910e9975cd0f98b71.jpg"
boxes = anno[anno['filename'] == sample_img_name]#get bounding box coordinate
bounding_boxes = boxes[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()#plot bb coordinate to list
print(f"bounding box of {sample_img_name} : {bounding_boxes}")#show list of bb

print(f"\n{"="*100}\n")

img = cv.imread(image_path)  # โหลดเป็นภาพสี (BGR)
if img is None:
    raise FileNotFoundError(f"not found image at path: {image_path}")

# ตรวจสอบขนาดภาพ
h, w = img.shape[:2]
if h != 640 or w != 640:
    print(f" this image is  {w}x{h} , not 640x640")

# ----- วาดกรอบ -----
for box in bounding_boxes:
    x1, y1, x2, y2 = box
    cv.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# ----- แสดงผล -----
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # เปลี่ยนเป็น RGB สำหรับ matplotlib
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Bounding Boxes on 640x640 Image")
plt.show()