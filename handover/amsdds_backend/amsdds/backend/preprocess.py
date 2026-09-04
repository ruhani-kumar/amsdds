import io
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def shades_of_gray(arr, power=6):
    a = arr.astype(np.float32)
    v = np.power(np.mean(np.power(a, power), (0, 1)), 1.0 / power)
    v = v / (np.sqrt((v ** 2).sum()) + 1e-8)
    return np.clip(a / (v * np.sqrt(3) + 1e-8), 0, 255).astype(np.uint8)

def load_image(src):
    if isinstance(src, Image.Image): return src.convert("RGB")
    if isinstance(src, (bytes, bytearray)): return Image.open(io.BytesIO(src)).convert("RGB")
    return Image.open(src).convert("RGB")

class Preprocessor:
    def __init__(self, cfg):
        p = cfg["preprocessing"]
        self.cc, self.power = bool(p["color_constancy"]), int(p["color_constancy_power"])
        l1, l2 = cfg["layer1"], cfg["layer2"]
        self.tf1 = T.Compose([T.Resize(l1["resize_short"]), T.CenterCrop(l1["img_size"]),
                              T.ToTensor(), T.Normalize(MEAN, STD)])
        self.tf2 = T.Compose([T.Resize(l2["resize_short"]), T.CenterCrop(l2["img_size"]),
                              T.ToTensor(), T.Normalize(MEAN, STD)])
    def normalise_colour(self, img):
        return Image.fromarray(shades_of_gray(np.array(img), self.power)) if self.cc else img
    def for_layer1(self, img): return self.tf1(img).unsqueeze(0)
    def for_layer2(self, img): return self.tf2(img).unsqueeze(0)
