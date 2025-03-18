import os 
from test_pre_train_model.ssd_anime_face_detect import ssd_anime_face_detect

def get_nb_face_found(directory):
    nohead+=1
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        head = ssd_anime_face_detect(img_path,"test_pre_train_model\ssd_anime_face_detect.pth")
<<<<<<< HEAD
    if head == []:
        nohead +=1
        

=======
>>>>>>> 3f2e1f7aae439f08fdd52141ad1aa3096df88552

print(get_nb_face_found("imgs_test\kiana"))