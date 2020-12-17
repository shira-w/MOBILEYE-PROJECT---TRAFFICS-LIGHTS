from PIL import Image
import numpy as np


class FrameData:
    def __init__(self):
        self.path = ""
        self.array = []
        self.Em = []
        self.traffic_light = []
        self.auxiliary = []
        self.id = 0

    def change(self, id, path, Em, traffic_light, auxiliary):
        self.id = id
        self.path = path
        self.array = np.asarray(Image.open(path))
        self.Em = Em
        self.traffic_light = traffic_light
        self.auxiliary = auxiliary

    def __copy__(self, other):
        self.path = other.path
        self.array = other.array
        self.Em = other.Em
        self.traffic_light = other.traffic_light
        self.auxiliary = other.auxiliary
        self.id=other.id
