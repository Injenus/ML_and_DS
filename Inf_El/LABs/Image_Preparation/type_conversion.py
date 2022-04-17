import os
from PIL import Image

source_dir_name_Ilya = "C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/raw_data_png/Ilya"
source_dir_name_Rob = "C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/raw_data_png/Rob"

out_dir_name_Ilya = 'C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/raw_data_temp/Ilya/'
out_dir_name_Rob = 'C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/raw_data_temp/Rob/'

folder = os.listdir(source_dir_name_Ilya)
for file_name in folder:
    if file_name.endswith(".png"):
        # print(file_name)
        im = Image.open(source_dir_name_Ilya + '/' + file_name)
        rgb_im = im.convert('RGB')
        rgb_im.save(out_dir_name_Ilya + 'k' + file_name[:-3] + 'jpg')

folder = os.listdir(source_dir_name_Rob)
for file_name in folder:
    if file_name.endswith(".png"):
        # print(file_name)
        im = Image.open(source_dir_name_Rob + '/' + file_name)
        rgb_im = im.convert('RGB')
        rgb_im.save(out_dir_name_Rob + 'k' + file_name[:-3] + 'jpg')
