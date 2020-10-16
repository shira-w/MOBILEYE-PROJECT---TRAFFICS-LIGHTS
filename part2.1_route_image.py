# AUXILIARY
# using as so to route images and make file of pictures for training bigger

import matplotlib.pyplot as plt
from load_data import crop


def center_crop(path_image):
    pic = plt.imread(path_image)
    plt.imshow(pic)
    plt.show()
    x = len(pic[0]) / 2
    y = len(pic) / 2
    print(x)
    print(y)
    crop(path_image, x, y)


def find_tfl_yourself(_image):
    plt.imshow(_image)
    plt.show()
    y = input("enter x  ")
    y = int(y)
    x = input("enter y  ")
    x = int(x)
    while True:
        is_tfl = input("is it tfl? enter y or n")
        if is_tfl == "y":
            is_tfl = b"00000001"
            break
        elif is_tfl == "n":
            is_tfl = b"00000000"
            break
    return x, y, is_tfl


def crop_yourself(path_image):
    _image = plt.imread(path_image)
    x, y, is_tfl = find_tfl_yourself(_image)
    crop(path_image, x, y)

