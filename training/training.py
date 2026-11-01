import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from load_data import train_data
from pathlib import Path

#get data
batch_size = 1

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

#network structure
layer1 = nn.Linear(784, 128)
layer2 = nn.Linear(128, 64)
layer3 = nn.Linear(64, 10)

#define loss function
loss_fn = nn.CrossEntropyLoss()

#optimiser
optimiser = torch.optim.Adam(
    list(layer1.parameters()) +
    list(layer2.parameters()) +
    list(layer3.parameters()),
    lr=0.001
)

#training
for batch, (images, labels) in enumerate(train_loader):
    X = images.view(-1, 784)

    #forward propagation
    Z1 = layer1(X)
    A1 = torch.relu(Z1)

    Z2 = layer2(A1)
    A2 = torch.relu(Z2)

    Y = layer3(A2)

    #loss
    loss = loss_fn(Y, labels)
    
    #backward propagation
    optimiser.zero_grad()
    loss.backward()

    optimiser.step()

    #print progress
    if (batch + 1) % 100 == 0:
        trained = min((batch + 1) * batch_size, len(train_data))

        print(f"Trained: {trained}/{len(train_data)}, Loss: {loss.item():.6f}")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

torch.save({ 
    "layer1": layer1.state_dict(),
    "layer2": layer2.state_dict(),
    "layer3": layer3.state_dict()
}, WEIGHTS_DIR / "mnist_mlp.pth")