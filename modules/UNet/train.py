"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model, utils
from torch.utils.data import DataLoader, random_split

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 15
BATCH_SIZE = 16
VAL_PERCENT = 0.2
LEARNING_RATE = 0.001
# NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 4

# Setup directories
img_dir_train = "../../data/VOC_DATA_FINAL_SPLIT/image/train"
mask_dir_train = "../../data/VOC_DATA_FINAL_SPLIT/bcc/train"
true_mask_dir_train = "../../data/VOC_DATA_FINAL_SPLIT/target/train"

img_dir_val = "../../data/VOC_DATA_FINAL_SPLIT/image/val"
mask_dir_val = "../../data/VOC_DATA_FINAL_SPLIT/bcc/val"
true_mask_dir_val = "../../data/VOC_DATA_FINAL_SPLIT/target/val"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create simple transform
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create train/test dataloader and get class names as a list
dataset_train = data_setup.SegmentationDataset(image_dir=img_dir_train, mask_dir=mask_dir_train, true_mask_dir=true_mask_dir_train)
dataset_val = data_setup.SegmentationDataset(image_dir=img_dir_val, mask_dir=mask_dir_val, true_mask_dir=true_mask_dir_val)

# 3. Create data loaders
loader_args = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
train_loader = DataLoader(dataset_train, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader = DataLoader(dataset_val, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


# Create model
model = model.UNet(in_channels=11, out_channels=21).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    # loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device)

# Save the model with help from utils.py
utils.save_model(
    model=model,
    target_dir="models",
    model_name="unet_model2.pth")