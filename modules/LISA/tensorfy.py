import os
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Resize
from tqdm import tqdm

OUTPUT_DIRECTORY = "/path/to/output/directory/"
CHAIR_DIR = "/home/scur2194/CV2/DAVOS/data/LISA_outputs/chair/"
PERSON_DIR = "/home/scur2194/CV2/DAVOS/data/LISA_outputs/person/"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
resize_transform = Resize((256, 256))

# Iterate over all images in the directories
for image_name in tqdm(os.listdir(CHAIR_DIR)):
    if image_name.endswith("_masked_img_0.jpg"):
        image_id = image_name.split('_masked_img_0.jpg')[0]
        chair_path = os.path.join(CHAIR_DIR, f"{image_id}_masked_img_0.jpg")
        person_path = os.path.join(PERSON_DIR, f"{image_id}_masked_img_0.jpg")

        # Initialize background mask
        background = np.ones((256, 256), dtype=np.uint8)

        # Initialize chair and person masks
        chair_mask = np.zeros((256, 256), dtype=np.uint8)
        person_mask = np.zeros((256, 256), dtype=np.uint8)

        # Load and resize chair mask
        if os.path.exists(chair_path):
            chair_img = Image.open(chair_path).convert("L")
            chair_mask = np.array(resize_transform(chair_img), dtype=np.uint8)
        
        # Load and resize person mask
        if os.path.exists(person_path):
            person_img = Image.open(person_path).convert("L")
            person_mask = np.array(resize_transform(person_img), dtype=np.uint8)
        
        # Create combined mask and update background mask
        combined_mask = np.maximum(chair_mask, person_mask)
        background[combined_mask > 0] = 0

        # Convert masks to tensors
        chair_tensor = torch.from_numpy(chair_mask).float()
        person_tensor = torch.from_numpy(person_mask).float()
        background_tensor = torch.from_numpy(background).float()

        # Stack tensors to create the final combined tensor
        combined_tensor = torch.stack([background_tensor, chair_tensor, person_tensor], dim=0)

        # Save the tensor
        torch.save(combined_tensor, os.path.join(OUTPUT_DIRECTORY, f"{image_id}.pt"))
