import json
import cv2
import numpy as np
import pickle
import os
from os import path
from google.cloud import storage
from tempfile import NamedTemporaryFile

imagenamelist = []
imagelist = []
kptlist = []

with open('train.json') as f:
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
#         img = cv2.resize(img,(960,720))
#         img = np.array(img)
#     height, width, _ = img.shape
        
#     with NamedTemporaryFile() as temp:
#         name = str(i) + ".jpg"
#         cv2.imwrite(name, img)
#         blob = bucket.blob('resized_MPII_images/' + img_name)
#         blob.upload_from_filename(name, content_type='image/jpeg')
    
#     imagelist.append(img)
    kptlist.append(kpt)
    
print("Exited Loop")
    
# pickle_out = pickle.dumps(imagenamelist)
# nblob = bucket.blob('train_image_name')
# nblob.upload_from_string(pickle_out)

# pickle_out = pickle.dumps(imagelist)
# nblob = bucket.blob('train_images')
# nblob.upload_from_string(pickle_out)

# pickle_out = pickle.dumps(kptlist)
# nblob = bucket.blob('train_joint_labels')
# nblob.upload_from_string(pickle_out)

# filename = 'train_images_1'
# outfile = open(filename,'wb')
# pickle.dump(imagelist, outfile)
# outfile.close()

filename = 'train_joint_labels_1'
outfile = open(filename,'wb')
pickle.dump(kptlist, outfile)
outfile.close()

filename = 'train_image_name_1'
outfile = open(filename,'wb')
pickle.dump(imagenamelist, outfile)
outfile.close()



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