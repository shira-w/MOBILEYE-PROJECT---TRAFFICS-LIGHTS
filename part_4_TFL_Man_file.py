import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sbn
from skimage import color
import scipy.ndimage as ndimage
from scipy.ndimage.filters import maximum_filter
from skimage.feature import peak_local_max

from load_data import crop
from run_attention import find_tfl_lights
from run_attention import test_find_tfl_lights
from frame_data import FrameData
from SFM_standAlone import FrameContainer
import SFM


class TFL_Man:
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        self.pp = data['principle_point']
        self.focal = data['flx']
        self.pkl_path = pkl_path
        self.current_frame = FrameData()
        self.prev_frame = FrameData()
        self.model = load_model("model.h5")
        self.fig = None
        self.axs = None

    def visualize(self, red_x, red_y, green_x, green_y, image, index, label):
        self.axs[index].imshow(image)
        self.axs[index].set_ylabel(label)
        self.axs[index].plot(red_x, red_y, 'ro', color='r', markersize=4)
        self.axs[index].plot(green_x, green_y, 'ro', color='g', markersize=4)

    def part1_tfl_candidates(self, image):
        x_red, y_red, x_green, y_green = find_tfl_lights(image)
        assert len(x_red) == len(y_red)
        assert len(x_green) == len(y_green)
        self.visualize(x_red, y_red, x_green, y_green, image, 0, "traffic lights")
        red_list = np.column_stack((x_red, y_red))
        green_list = np.column_stack((x_green, y_green))
        candidates = np.vstack([red_list, green_list])
        auxiliary = ["red"] * len(red_list) + ["green"] * (len(green_list))
        return candidates, auxiliary

    def part_2_tfl_valid(self, image_path, candidates, auxiliary):
        image = np.array(Image.open(image_path))
        new_auxiliry = []
        sub_candidates = np.empty(shape=[0, 2])
        index = 0
        i = 0
        for candidate in candidates:
            cropped_image = crop(image_path, candidate[0], candidate[1])
            crop_shape = (81, 81)
            test_image = cropped_image.reshape([-1] + list(crop_shape) + [3])
            l_predictions = self.model.predict(test_image)
            predicted_label = np.argmax(l_predictions, axis=-1)
            if predicted_label[0] == 1:
                sub_candidates = np.vstack([sub_candidates, candidate])
                new_auxiliry.append(auxiliary[index])
            if auxiliary[index] == "green" and auxiliary[index - 1] == "red":
                i = index

            index += 1

        x_red = sub_candidates[:i, 0]
        y_red = sub_candidates[:i, 1]
        x_green = sub_candidates[i:, 0]
        y_green = sub_candidates[i:, 1]
        self.visualize(x_red, y_red, x_green, y_green, image, 1, "validation")
        return sub_candidates, new_auxiliry

    def part_3_visualize(self, prev_container, curr_container, focal, pp):
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, focal, pp)
        norm_rot_pts = SFM.rotate(norm_prev_pts, R)
        rot_pts = SFM.unnormalize(norm_rot_pts, focal, pp)
        foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, pp))
        prev_p = prev_container.traffic_light
        self.axs[2].imshow(curr_container.img)
        curr_p = curr_container.traffic_light
        self.axs[2].plot(curr_p[:, 0], curr_p[:, 1], 'b+')
        self.axs[2].plot(curr_p[:, 0], curr_p[:, 1], 'b+')
        for i in range(len(curr_p)):
            self.axs[2].plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
            if curr_container.valid[i]:
                self.axs[2].text(curr_p[i, 0], curr_p[i, 1],
                                 r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')

        self.axs[2].plot(foe[0], foe[1], 'r+')
        self.axs[2].plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
        plt.draw()
        plt.show()

    def part_3_find_distance(self):
        prev_container = FrameContainer(self.prev_frame.path)
        curr_container = FrameContainer(self.current_frame.path)
        prev_container.traffic_light = self.prev_frame.traffic_light
        curr_container.traffic_light = self.current_frame.traffic_light
        curr_container.EM = self.current_frame.Em
        curr_container = SFM.calc_TFL_dist(prev_container, curr_container, self.focal, self.pp)
        self.part_3_visualize(prev_container, curr_container, self.focal, self.pp)

    def run(self, image_path, index, Em):
        fig, axs = plt.subplots(3, 1)
        self.fig = fig
        self.axs = axs
        image = np.array(Image.open(image_path))
        candidates, auxiliary = self.part1_tfl_candidates(image)
        i = 0
        for ind, x in enumerate(auxiliary):
            if x == "green":
                i = ind
        # self.(x_red, y_red, x_green, y_green, image, 0, "traffic lights")
        sub_candidates, new_auxiliary = self.part_2_tfl_valid(image_path, candidates, auxiliary)
        assert len(candidates) >= len(sub_candidates)
        self.current_frame.change(index, image_path, Em, sub_candidates, new_auxiliary)
        if self.prev_frame.path == "":
            plt.draw()
            plt.show(block=True)
        else:
            self.part_3_find_distance()
        self.prev_frame.__copy__(self.current_frame)
