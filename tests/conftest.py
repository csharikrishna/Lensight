import torch
import torch.nn as nn
import torch.utils.data as data
import pytest


class TinyCNN(nn.Module):
    """A minimal CNN classifier — enough Conv2d/Linear structure to exercise
    every explainer and analyzer without needing any downloaded weights."""

    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class RandomImageDataset(data.Dataset):
    """Deterministic pseudo-random image/label pairs — a stand-in for a real dataset."""

    def __init__(self, n_samples=32, num_classes=4, image_size=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n_samples, 3, image_size, image_size, generator=g)
        self.labels = torch.randint(0, num_classes, (n_samples,), generator=g)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


@pytest.fixture
def num_classes():
    return 4


@pytest.fixture
def model(num_classes):
    torch.manual_seed(0)
    m = TinyCNN(num_classes=num_classes)
    m.eval()
    return m


@pytest.fixture
def single_image():
    torch.manual_seed(1)
    return torch.randn(3, 16, 16)


@pytest.fixture
def dataloader(num_classes):
    ds = RandomImageDataset(n_samples=40, num_classes=num_classes, image_size=16)
    return data.DataLoader(ds, batch_size=8, shuffle=False)
