from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from syne_tune import Reporter

DATA_PATH = "/home/tzhao/Workspace/hyperparameter-tuning-example/data"
NUM_EPOCHS = 4
DEVICE = "cpu"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_syne(batch_size, lr, momentum, verbose=True, download=True):
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Load nn
    net = Net().to(DEVICE)

    # Load loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    #################################
    ##### SYNE: define reporter #####
    #################################
    
    reporter = Reporter()
    
    #################################
    #################################
    #################################

    
    # Train
    for epoch in range(NUM_EPOCHS):
        
        # Train single epoch
        for data in trainloader:
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Test
        test_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = net(inputs)

                # Get test loss
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Get test acc
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total

        if verbose:
            print(f"Epoch {epoch}: Test loss: {test_loss:.2f}")
            print(f"Epoch {epoch}: Test acc: {test_acc:.2f}")
            
        #################################
        ##### SYNE: report test acc #####
        #################################

        reporter(epoch=epoch + 1, test_inacc=1 - test_acc)

        #################################
        #################################
        #################################

    if verbose:
        print("Training completed.")
        print(f"Final test loss: {test_loss:.2f}")
        print(f"Final test acc: {test_acc:.2f}")
    
    return test_acc

def main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    args, _ = parser.parse_known_args()
    
    train_syne(batch_size=args.batch_size, lr=args.lr, momentum=args.momentum, verbose=False, download=False)

if __name__ == "__main__":
    main()
