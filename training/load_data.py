from pathlib import Path
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent / "data"

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)