import torch
import random
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 69
random.seed(SEED)
DATA_LIST = list(glob.glob('dataset/**/*.0.jpg', recursive=True))
random.shuffle(DATA_LIST)
TRAIN_LEN = int(len(DATA_LIST) * 0.8)
TRAIN_LIST = DATA_LIST[:TRAIN_LEN]
VALID_LIST = DATA_LIST[TRAIN_LEN:]
print(VALID_LIST)
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=1024, height=1024),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),
        # A.ColorJitter(p=0.2), ## Watch out this, Not good for cell I assume
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)