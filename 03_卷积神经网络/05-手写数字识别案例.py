import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary


def create_dataset():
    train_dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())
    valid_dataset = MNIST(root="./data", train=False, download=False, transform=ToTensor())

    return train_dataset, valid_dataset

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(in_features=250, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=70)

        self.out = nn.Linear(in_features=70, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = torch.relu(x)

        x = self.linear2(x)
        x = torch.relu(x)

        x = self.out(x)
        return x


def train(train_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    model = ImageModel().to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    epoches = 10
    for epoch in range(epoches):
        start = time.time()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model.train()
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(y)
        print(f"epoch: {epoch+1}, loss: {total_loss:.2f}, time: {time.time()-start:.2f}")
    torch.save(obj=model.state_dict(), f="./model/MNIST.pth")

def test(valid_dataset):
    dataloader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False)
    model = ImageModel()
    model.load_state_dict(torch.load("./model/MNIST.pth"))
    total_correct = 0
    total_samples = 0
    for x, y in dataloader:
        model.eval()
        output = model(x)
        y_pred = torch.argmax(output, dim=1)
        print(f"y_pred: {y_pred}")
        total_correct += (y_pred==y).sum()
        total_samples += len(y)
    print(f"ACC: {total_correct/total_samples:.2f}")

def test2(valid_dataset):
    model = ImageModel()
    model.load_state_dict(torch.load("./model/MNIST.pth", map_location="cuda"))
    # print(valid_dataset.data.shape)
    # print(valid_dataset.targets.shape)
    while True:
        i = int(input("请输入要预测的: "))
        x, y = valid_dataset[i]
        plt.figure(figsize=(2, 2))
        plt.imshow(x.permute(dims=(2, 1, 0)))
        plt.show()
        x = x.unsqueeze(dim=0)
        output = model(x)
        y_pred = torch.argmax(output)
        print(f"预测: {y_pred}, 真实: {y}")


if __name__ == "__main__":
    train_dataset, valid_dataset = create_dataset()
    # train(train_dataset=train_dataset)
    # test(valid_dataset=valid_dataset)
    test2(valid_dataset)
  