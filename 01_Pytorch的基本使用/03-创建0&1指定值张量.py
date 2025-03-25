import torch


def demo1():
    t1 = torch.ones(3, 4)
    print(t1)
    print(t1.dtype)
    print('\n')

    t2 = torch.ones_like(t1)
    print(t2)
    print(t2.dtype)
    print('\n')

    t3 = torch.zeros(3, 4)
    print(t3)
    print(t3.dtype)
    print('\n')

    t4 = torch.zeros_like(t1)
    print(t4)
    print(t4.dtype)
    print('\n')

    t5 = torch.full(size=(2, 3), fill_value=10)
    print(t5)
    print(t5.dtype)
    print('\n')

    t6 = torch.full_like(t5, fill_value=23)
    print(t6)
    print(t6.dtype)



if __name__ == "__main__":
    demo1()