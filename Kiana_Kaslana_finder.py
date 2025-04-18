"""import os 
from test_pre_train_model.ssd_anime_face_detect import ssd_anime_face_detect

def get_nb_face_found(directory):
    nohead=0
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        head = ssd_anime_face_detect(img_path,"test_pre_train_model\ssd_anime_face_detect.pth")
        if head == []:
            nohead +=1

print(get_nb_face_found("imgs_test\kiana"))"""

# import libraries
from yoloface import face_analysis
import numpy
import cv2
face=face_analysis()        #  Auto Download a large weight files from Google Drive.
                            #  only first time.
                            #  Automatically  create folder .yoloface on cwd.
# example 1
#%%time
img,box,conf=face.face_detection(image_path='imgs_test\kiana\kiana_2.jpg',model='tiny')
print(box)                  # box[i]=[x,y,w,h]
print(conf)                 #  value between(0 - 1)  or probability
face.show_output(img,box)