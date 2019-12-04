
# coding: utf-8

# ###############
# Script Author = Richesh Chouksey
# ###############
# Created on: "4/12/2019 6:20 pm ist"

# #### Script details
# Script reads files from source directory having images and applies CLAHE transformation and saves images in destination folder path.
# To run script use command "python CLAHE_transformation_openCV.py -src '/home/ronitpc/myFiles/work/openSourceData/subset_all-images/' -dest '/home/ronitpc/myFiles/work/openSourceData/Clahe_implemented_Img/'
# "
# the -src is path for source folder and -dest is path for destination folder.

# In[53]:


import cv2
import numpy as np
from PIL import Image
import argparse
import glob
import os
import regex as re


# In[54]:


#imageName = 'im0011.ppm'
#srcFolder = '/home/ronitpc/myFiles/work/openSourceData/subset_all-images/'
#destFolder = '/home/ronitpc/myFiles/work/openSourceData/Clahe_implemented_Img/'


# In[55]:


##### function converts image from pil to cv2 format

def pil_to_cv2(pil_image):
    cv2_image = np.array(pil_image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image


# In[56]:


##### function converts image from cv2 format to pil format

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


# In[57]:


#### function enhances image luminosity using CLAHE transformation

def enhance(image_path, clip_limit=3):
    image = cv2.imread(image_path)
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert iamge from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return cv2_to_pil(final_image) 


# In[ ]:


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-src", "--source", required=True, help="Path to source image folder")
ap.add_argument("-dest", "--destination", required=True,
	help="Path to folder where modified images will be saved")

args = vars(ap.parse_args())

# paths
srcFolder = args['source']
destFolder = args['destination']

####### check if source folder exists
if os.path.isdir(srcFolder) and os.listdir(srcFolder):
    pass
else:
    print('invalid Source path given')
    exit()
### if destination folder not exists create
os.makedirs(destFolder, exist_ok=True)

####### read images from source folder, apply Clahe transformation and save images
for filename in os.listdir(srcFolder):
    sourceImage = os.path.join(srcFolder,filename)
    if sourceImage is not None:
        img = enhance(sourceImage)
        img = pil_to_cv2(img)
        
        ### save image in destination folder
        # if ppm file type then change name for changes saving purpose
        if '.ppm' in filename:
            file = re.sub('.ppm', '.jpg', filename)
        else:
            file = filename
            
        # save file
        destinationName = destFolder + file
        cv2.imwrite(destinationName, img)


# #cv2.imshow('image', img)
# #cv2.waitKey(0)
# 
# sourceImage = '/home/ronitpc/myFiles/work/openSourceData/subset_all-images/im0085.ppm'
# 
# if sourceImage is not None:
#     print("came here")
#     img = enhance(sourceImage)
#     img = pil_to_cv2(img)
# 
#     ### save image in destination folder
#     # if ppm file type then change name for changes saving purpose
#     filename = 'im0085.ppm'
#     if '.ppm' in filename:
#         file = re.sub('.ppm', '.jpg', filename)
#         print(file)
#     else:
#         file = filename
# 
#     # save file
#     destinationName = destFolder + file
#     cv2.imwrite(destinationName, img)
