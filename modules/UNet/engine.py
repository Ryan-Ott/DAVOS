from data_setup import SegmentationDataset
from model import UNet
# from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
from loss import dice_loss
import wandb

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    # loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device):

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        # loss = loss_fn(y_pred, y)
        loss = dice_loss(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()


    train_loss = train_loss / len(dataloader)

    return train_loss

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    # loss_fn: torch.nn.Module,
    device: torch.device):
    
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            pred = model(X)

            # 2. Calculate and accumulate loss
            # loss = loss_fn(pred, y)
            loss = dice_loss(pred, y)
            test_loss += loss.item()


    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    return test_loss

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    # loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device):
    
    # wandb.init(project="DAVOS", entity="DAVOS-CV", name='UNet-exp', settings=wandb.Settings(code_dir="."), config={
    #     "learning_rate": optimizer.param_groups[0]['lr'],
    #     "epochs": epochs,
    #     "batch_size": train_dataloader.batch_size
    # })

    
    # Create empty results dictionary
    results = {"train_loss": [],
        "test_loss": []
    }

        # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                        dataloader=train_dataloader,
                                        # loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
        test_loss = test_step(model=model,
        dataloader=test_dataloader,
        # loss_fn=loss_fn,
        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        
        # wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        
    # wandb.finish()

    # Return the filled results at the end of the epochs
    return results