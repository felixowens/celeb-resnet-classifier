import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from celeb_classifier.dataset import CelebDataModule
from celeb_classifier.model import CelebModel
from datetime import datetime
import os
import random
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    # Training loop
    learning_rate = 0.001
    batch_size = 64
    epochs = 10

    # Set seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cuda')
    print(f"Using device: {device}")

    # Initialize tensorboard
    log_dir = 'runs/experiment_' + f"lr{learning_rate}_bs{batch_size}_ep{epochs}_" + datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir)

    # Setup data
    data_module = CelebDataModule(batch_size=batch_size, val_batch_size=1000)
    train_loader, test_loader = data_module.get_dataloaders()

    # Initialize model, optimizer, and loss function
    model = CelebModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)  # Decay LR by a factor of 0.1 every 2 epochs
    criterion = nn.CrossEntropyLoss()


    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)

                if batch_idx == 0:
                    print(f"images shape: {images.shape}")
                    print(f"labels shape: {labels.shape}")
                    
                    # print number of images in batch
                    print(f"Number of images in batch: {len(images)}")

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()


                # Update tqdm progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'accuracy': 100. * correct / total,
                    'step': batch_idx + 1,
                    'lr': current_lr,
                })
                pbar.update(1)

                if batch_idx % 100 == 99:
                    writer.add_scalar('training loss',
                                      running_loss / 100,
                                      epoch * len(train_loader) + batch_idx)
                    writer.add_scalar('training accuracy',
                                      100. * correct / total,
                                      epoch * len(train_loader) + batch_idx)
                    running_loss = 0.0

            writer.add_scalar('learning rate', current_lr, epoch)

        scheduler.step()  # Update the learning rate

        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        writer.add_scalar('test accuracy', accuracy, epoch)
        print(f'Epoch {epoch+1}: Test Accuracy: {accuracy:.2f}%')

    writer.close()

    # Ensure the directory exists
    os.makedirs("./models", exist_ok=True)

    # Format the filename with the config parameters
    filename = f"./models/model_lr{learning_rate}_bs{batch_size}_ep{epochs}.pth"
    torch.save(model.state_dict(), filename)

if __name__ == "__main__":
    train()