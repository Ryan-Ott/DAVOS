import os
import shutil
import random

def create_directories(base_path, folders):
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join(base_path, folder, 'test'), exist_ok=True)
        os.makedirs(os.path.join(base_path, folder, 'val'), exist_ok=True)

def move_files(file_list, base_path, new_base, set_type):
    with open(file_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            paths = line.strip().split(',')
            for path in paths:
                src = os.path.join(base_path, path)
                dest = os.path.join(new_base, os.path.dirname(path), set_type)
                shutil.move(src, dest)

def split_train_val(base_path, new_base, folders, val_ratio=0.2):
    for folder in folders:
        train_path = os.path.join(new_base, folder, 'train')
        val_path = os.path.join(new_base, folder, 'val')
        
        # Get all files in the train directory
        all_files = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
        
        # Shuffle and split files
        random.shuffle(all_files)
        val_size = int(len(all_files) * val_ratio)
        val_files = all_files[:val_size]

        # Move files to the val directory
        for file in val_files:
            src = os.path.join(train_path, file)
            dest = os.path.join(val_path, file)
            shutil.move(src, dest)

def main():
    base_path = 'VOC_DATA_FINAL'  # Adjust if your base path is different
    new_base = 'VOC_DATA_FINAL_SPLIT'
    train_list = 'VOC_DATA_FINAL/train_pairs_bcc.txt'
    test_list = 'VOC_DATA_FINAL/test_pairs_bcc.txt'
    folders = ['image', 'target', 'bcc']

    create_directories(new_base, folders)
    
    move_files(train_list, base_path, new_base, 'train')
    move_files(test_list, base_path, new_base, 'test')
    
    split_train_val(base_path, new_base, folders, val_ratio=0.2)

if __name__ == "__main__":
    main()
