import torch
import numpy as np

def dm01():
    list1 = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    int1 = 10
    list2 = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    t1 = torch.tensor(data=list1)
    print(f"t1的值: {t1}")
    print(f"t1的类型: {type(t1)}")
    print(f"t1的数据类型: {t1.dtype}")
    t2 = torch.tensor(data=int1)
    print(f"t2的值: {t2}")
    print(f"t2的数据类型: {t2.dtype}")
    print(f"t2的类型: {type(t2)}")
    t3 = torch.tensor(data=list2)
    print(f"t3的值: {t3}")
    print(f"t3的数据类型: {t3.dtype}")
    print(f"t3的类型: {type(t3)}")

def dm02():
    t1 = torch.Tensor([
        [1.1, 2.2, 3.3],
        [3.2, 3.5, 3.4]
    ])
    print(f"t1的值: {t1}")
    print(f"t1的类型: {type(t1)}")
    print(f"t1的数据类型: {t1.dtype}")

def dm03():
    t1 = torch.IntTensor([
        [1.1, 2.3, 4.7],
        [4, 5, 6]
    ])
    print(f"t1的值: {t1}")
    print(f"t1的类型: {type(t1)}")
    print(f"t1的数据类型: {t1.dtype}")
    t2 = torch.FloatTensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print(f"t2的值: {t2}")
    print(f"t2的类型: {type(t2)}")
    print(f"t2的数据类型: {t2.dtype}")




if __name__ == "__main__":
    # dm01()
    # dm02()
    dm03()