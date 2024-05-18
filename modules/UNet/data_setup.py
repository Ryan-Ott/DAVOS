import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, true_mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.true_mask_dir = true_mask_dir
        # self.filenames = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.filenames = [f for f in os.listdir(mask_dir)]
        self.transform = transform
        
    def load_image(self, idx: int):
        filename_with_jpg = os.path.splitext(self.filenames[idx])[0] + '.jpg'
        image_path = os.path.join(self.image_dir, filename_with_jpg)
        mask_path = os.path.join(self.mask_dir, self.filenames[idx])
        true_mask_path = os.path.join(self.true_mask_dir, self.filenames[idx])
        return cv2.imread(image_path), torch.load(mask_path), torch.load(true_mask_path)
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # image_path = os.path.join(self.image_dir, self.filenames[idx])
        # mask_path = os.path.join(self.mask_dir, self.filenames[idx])

        # image = torch.load(image_path)  # Load the image tensor
        # mask = torch.load(mask_path)    # Load the mask tensor
        image, mask, true_mask = self.load_image(idx)
        image = torch.tensor(image, dtype=torch.float32)
        
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(256), mode='nearest')  # upsample bcc_mask from 32*32 to 256*256
        mask = torch.nn.functional.one_hot(mask.to(torch.int64), num_classes=8).squeeze().permute([2,0,1]).to(torch.float) # 8 classes (see Ozzy's paper)
        # mask = mask.permute(1, 2, 0)
        
        # true_mask = true_mask.squeeze(0).permute(1, 2, 0)
        true_mask = true_mask.squeeze(0)

        # if self.transform:
            # input_tensor = self.transform(input_tensor)
            # true_mask = self.transform(true_mask)
            # tr# Added our own dataset class
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.ToTensor()  
        ])
        
        image = image.permute(2, 0, 1)
        to_pil = transforms.ToPILImage()
        image = to_pil(image)
        
        # image = transform(image).permute(1, 2, 0)
        image = transform(image)
        input_tensor = torch.cat([image, mask], dim=0)  # Concatenating image and mask

        return input_tensor, true_mask