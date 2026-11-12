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

#one training step
def train_step(images, labels):
    """One optimiser step, exactly as the loop below always did it.

    The forward pass, the loss, the backward pass and the optimiser update are
    unchanged; the intermediate tensors are returned so the visualiser can show
    the values of this step (they are read before the weights were updated).
    """
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

    return {
        "X": X,
        "A1": A1,
        "A2": A2,
        "Y": Y,
        "loss": loss,
        "images": images,
        "labels": labels
    }

#training
def train():
    for batch, (images, labels) in enumerate(train_loader):
        result = train_step(images, labels)
        loss = result["loss"]

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


# Only start training when this file is run directly. The recorder imports this
# module to read the network above (layer1..layer3); importing it must never
# start a training run.
if __name__ == "__main__":
    train()
