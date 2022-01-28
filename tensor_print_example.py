from numpy import NaN
import torch


def main():
    # 0D
    print(torch.tensor(3))
    print(torch.tensor(3.14))
    print(torch.tensor(3, device="cuda:0"))
    print(torch.tensor(float('nan')))
    print(torch.tensor(float('inf')))
    print(torch.tensor(float('-inf')))

    # 1D
    print(torch.tensor([1, 2, 3]))
    print(torch.tensor([True, False]))

    # 2D
    print(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    print(torch.tensor([[1.2, 2., 3], [4, 5, 6]]))

    # 3D
    print(torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))


if __name__ == "__main__":
    main()
