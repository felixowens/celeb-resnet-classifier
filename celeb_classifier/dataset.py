import torch
from torchvision import datasets, transforms

class CelebDataModule:
    def __init__(self, batch_size=64, val_batch_size=1000):
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        
    def get_dataloaders(self):
        """Create training and test dataloaders."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.Celeb(root='./data', train=True, transform=transform, download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader