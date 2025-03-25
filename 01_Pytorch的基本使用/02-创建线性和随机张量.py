import torch


torch.random.manual_seed(22)  # 设置随机数种子
torch.random.initial_seed()  # 设置随机数种子

# 线性张量
def dem01():
    t1 = torch.arange(1, 10, 2)
    print(t1)
    print(t1.dtype)

    t2 = torch.linspace(1, 10, 7)
    print(t2)
    print(t2.dtype)

    t3 = torch.logspace(2, 5, 10, base=4)
    print(t3)
    print(t3.dtype)

# 随机张量
def dem02():
    t1 = torch.rand(3, 4)
    print(t1)
    print(t1.dtype)

    t2 = torch.randn(3, 4)
    print(t2)
    print(t2.dtype)

    t3 = torch.randint(1, 10, size=(3, 4))
    print(t3)
    print(type(t3))







if __name__ == "__main__":
    dem01()
    # dem02()