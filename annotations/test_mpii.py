import json
import cv2
import numpy as np
import pickle
import os
from os import path
from google.cloud import storage

imagenamelist = []
imagelist = []
kptlist = []

with open('valid.json') as f:
    data = json.load(f)

storage_client = storage.Client("pose_estimation")
bucket = storage_client.get_bucket('pose_estimation_datasets')

print(len(data))

for i in range(len(data)):
    # if (data[i]['image'] == "000004812.jpg"):
    img_name = data[i]['image']
    
    blob = bucket.blob('MPII/images/' +  img_name)
    blob.content_type = 'image/jpeg'
    image = np.asarray(bytearray(blob.download_as_string()))
    img = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    
#     if  os.path.exists("pose_estimation_datasets/MPII/images/" + img_name):
    kpt = np.asarray(data[i]['joints'], dtype=np.int32)
#     img = cv2.imread("pose_estimation_datasets/MPII/images/" + img_name)
    imagenamelist.append(data[i]['image'])

    if img.shape[0] != 960 or img.shape[1] != 720:
        kpt[:,0] = kpt[:,0] * (960/img.shape[1])
        kpt[:,1] = kpt[:,1] * (720/img.shape[0])
        img = cv2.resize(img,(960,720))
        img = np.array(img)
#     height, width, _ = img.shape
    
    imagelist.append(img)
    kptlist.append(kpt)

pickle_out = pickle.dumps(imagenamelist)
nblob = bucket.blob('test_image_name')
nblob.upload_from_string(pickle_out)

pickle_out = pickle.dumps(imagelist)
nblob = bucket.blob('test_images')
nblob.upload_from_string(pickle_out)

pickle_out = pickle.dumps(kptlist)
nblob = bucket.blob('test_joint_labels')
nblob.upload_from_string(pickle_out)    


# filename = 'test_image_name'
# outfile = open(filename,'wb')
# pickle.dump(imagenamelist, outfile)
# outfile.close()

# filename = 'test_images'
# outfile = open(filename,'wb')
# pickle.dump(imagelist, outfile)
# outfile.close()

# filename = 'test_joint_labels'
# outfile = open(filename,'wb')
# pickle.dump(kptlist, outfile)
# outfile.close()



# filename = 'train_images'
# infile = open(filename,'rb')
# new_dict = pickle.load(infile)
# infile.close()

# filename = 'train_kpt'
# infile = open(filename,'rb')
# kpt_in = pickle.load(infile)
# infile.close()

# img1 = new_dict[0]
# j1 = tuple(kpt_in[0][8])
# j2 = tuple(kpt_in[0][9])

# print(j1)
# print(j2)

# print(img.shape) (1080, 1920, 3)

# img1 = cv2.line(img1,j1, j2,(255,0,0),5)
# cv2.imshow('image',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# imagename, img, joints