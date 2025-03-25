import torch
import torch.nn as nn  # 代表 Neural Network（神经网络） 模块


def dm01():
    x = torch.ones(size=(2, 5))
    y = torch.zeros(size=(2, 3))
    print(x)
    print(y)

    w = torch.randn(size=(5, 3), requires_grad=True)
    b = torch.randn(size=(3,), requires_grad=True)
    print(w)
    print(b)

    y_pred = torch.matmul(x, w) + b
    print(y_pred)

    # 创建MSE对象
    criterion = nn.MSELoss()
    loss = criterion(y_pred, y)

    loss.sum().backward()
    print(w.grad)
    print(b.grad)


if __name__ == "__main__":
    dm01()