import os
from PIL import Image

img_size = (250, 250)


def crop_max_square(pil_img):
    img_width, img_height = pil_img.img_size
    min_size = min(img_width, img_height)
    return pil_img.crop(((img_width - min_size) // 2,
                         (img_height - min_size) // 2,
                         (img_width + min_size) // 2,
                         (img_height + min_size) // 2))


dir_name_Ilya = "C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/RAW_DATA/Ilya_Muromets/"
dir_name_Rob = "C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/RAW_DATA/The_Nightingale_The_Robber/"

for file_name in os.listdir(dir_name_Ilya):
    im = Image.open(dir_name_Ilya + file_name)
    square_im = crop_max_square(im)
    square_im.save(
        'C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/PROCESSED_DATA/SQUARE/Ilya/' + file_name)
    scaled_im = square_im.resize(img_size)
    scaled_im.save(
        'C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/PROCESSED_DATA/SCALED/Ilya_Muromets/' + file_name)

for file_name in os.listdir(dir_name_Rob):
    im = Image.open(dir_name_Rob + file_name)
    square_im = crop_max_square(im)
    square_im.save(
        'C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/PROCESSED_DATA/SQUARE/Rob/' + file_name)
    scaled_im = square_im.resize(img_size)
    scaled_im.save(
        'C:/Users/injen/Desktop/PyProjects/ML&DS/Inf_El/LABs/Image_Preparation/PROCESSED_DATA/SCALED/The_Nightingale_The_Robber/' + file_name)
