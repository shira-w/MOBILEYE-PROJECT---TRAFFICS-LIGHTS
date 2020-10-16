from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import parser
import argparse
import pathlib
import numpy as np
import random


# AUXILIARY FUNCS   ============================================
def to_binary(data):
    for d in data:
        image_array = d["image"]
        image_array = image_array.astype('uint8')
        is_tfl = d["is_tfl"]
        with open('data.bin', 'ab') as file:
            np.save(file, image_array)
        with open('labels.bin', 'ab') as f2:
            f2.write(is_tfl)


def crop(path_image: str, x, y):
    left = y - 40
    top = x - 40
    right = y + 40
    bottom = x + 40
    im = plt.imread(path_image)
    cropped_image = im[top:bottom, left:right]
    if len(cropped_image) > 0:
        print(path_image)
    return cropped_image

# SECONDERY FUNCTIONS  ==================================================================

def find_tl(_image: np.ndarray, all_tl):
    rand = random.randint(0, len(all_tl))
    tl_x = all_tl[0][rand]
    tl_y = all_tl[1][rand]
    return tl_x, tl_y


def find_not_tf(_image: np.ndarray, all_tl):
    mask = np.in1d(np.arange(np.shape(_image)[1]), all_tl[1])
    no_tl = np.where(~mask)[0]
    rand = random.randint(0, len(no_tl) - 1)
    not_tl_x = random.randint(0, np.shape(_image)[0])
    not_tl_y = no_tl[rand]
    return not_tl_x, not_tl_y


def lable_data(path_image, grey_image: np.ndarray):
    all_tl = np.where(grey_image == 19)
    if len(all_tl[0]) == 0:
        return []
    tl_x, tl_y = find_tl(grey_image, all_tl)
    not_tl_x, not_tl_y = find_not_tf(grey_image, all_tl)
    tl = crop(path_image, tl_x, tl_y)
    ntl = crop(path_image, not_tl_x, not_tl_y)
    return [{"image": tl, "is_tfl": b"00000001"}, {"image": ntl, "is_tfl": b"00000000"}]

# ==============================================================================


def load_images():
    files = glob.glob(os.path.join("gtFine", "train", "*", "*labelids.png"))
    for image_path in files:
        image_array = np.array(Image.open(image_path), dtype='uint8')
        image_name = "\\".join(image_path.split("\\")[-2:])
        color_name = image_name.replace("gtFine_labelIds.png", "leftImg8bit.png")
        color_path = os.path.join("leftImg8bit_trainvaltest", "leftImg8bit", "train", color_name)
        dict_res = lable_data(color_path, image_array)
        to_binary(dict_res)


if __name__ == '__main__':
    load_images()
