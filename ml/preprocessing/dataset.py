from PIL import Image
from torch.utils.data import Dataset


class HAM(Dataset):
    def __init__(self, frame, tf):
        self.f = frame.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        r = self.f.iloc[i]
        return self.tf(Image.open(r.path).convert("RGB")), int(r.y)
