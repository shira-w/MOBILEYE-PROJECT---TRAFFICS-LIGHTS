# pkl
# frame 1
# frame 2

def build_pls(image_name, start_i, end_i):
    image_name_pls = image_name.replace("pkl", "pls")
    with open(image_name_pls, "w") as f:
        f.write(image_name+"\n")
        #f.write(str(start_i) + "\n")
        for i in range(start_i, end_i + 1):
            frame_name = image_name.replace(".pkl", "_0000" + str(i) + "_leftImg8bit.png")
            f.write(frame_name + "\n")


build_pls("dusseldorf_000049.pkl",24,29)