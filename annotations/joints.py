import json
import cv2
import numpy as np

def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

with open('train.json') as f:
	data = json.load(f)

for i in range(len(data)):
    if (data[i]['image'] == "000001163.jpg"):
        img_name = data[i]['image']
        kpt = np.asarray(data[i]['joints'])

img = cv2.imread("../mpi_subset50/" + img_name)

stride = 8

if img.shape[0] != 368 or img.shape[1] != 368:
    kpt[:,0] = kpt[:,0] * (368/img.shape[1])
    kpt[:,1] = kpt[:,1] * (368/img.shape[0])
    img = cv2.resize(img,(368,368))
height, width, _ = img.shape

heatmap = np.zeros((int(height/stride), int(width/stride), int(len(kpt)+1)), dtype=np.float32)
for i in range(len(kpt)):
    # resize from 368 to 46
    x = int(kpt[i][0]) * 1.0 / stride
    y = int(kpt[i][1]) * 1.0 / stride
    heat_map = guassian_kernel(size_h=int(height/stride),size_w=int(width/stride), center_x=x, center_y=y, sigma=3)

    heat_map[heat_map > 1e-15] = 1
    # heat_map[heat_map < 0.0099] = 0

    #normalize here instead?

    heatmap[:, :, i + 1] = heat_map

heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

heat = cv2.applyColorMap(np.uint8(255*heatmap[:,:,0]), cv2.COLORMAP_JET)
im_heat  = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
# img = cv2.line(img1,j1, j2,(255,0,0),5)
# img = cv2.resize(img, (368, 368))

cv2.imshow('image',im_heat)
cv2.waitKey(0)
cv2.destroyAllWindows()