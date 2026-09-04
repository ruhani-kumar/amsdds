import torchvision.transforms as T

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Inference / evaluation. Deterministic.
eval_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

# Training. The photometric jitter is deliberate — it targets the lighting
# and image-quality failure modes named in the problem statement, not just
# geometric invariance.
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0), ratio=(0.85, 1.18)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(30)], p=0.5),
    T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.03),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.25),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
