from TFL_Man_file import TFL_Man
import pickle
import numpy as np


class Controller:
    def __init__(self, playlist_file_name):
        with open(playlist_file_name, "r") as playlist_file:
            pkl_path = playlist_file.readline()[:-1]
        with open(pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        self.tfl_m = TFL_Man(pkl_path)
        self.playlist_file_name = playlist_file_name
        self.data = data

    def run(self):
        with open(self.playlist_file_name, "r") as playlist_file:
            next(playlist_file)
            for i, path in enumerate(playlist_file):
                if i==0:
                    start_i = int(path.split("_")[-2].replace("0", ""))
                EM = []
                if i > 0:
                    EM = np.eye(4)
                    EM = np.dot(self.data['egomotion_' + str(i + start_i - 1) + '-' + str(i + start_i)], EM)
                self.tfl_m.run(path[:-1], i + start_i, EM)


c = Controller("dusseldorf_000049.pls")
c.run()
