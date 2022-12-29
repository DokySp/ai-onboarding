import torch.nn as nn

class MyLeNet5_1(nn.Module):
    def __init__(self):
        # self -> 생성된 인스턴스를 가리킴!!

        # 부모 생성자 콜
        super(MyLeNet5_1, self).__init__()

        # input: 24 / 1 channel

        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # Stride 1로하면 ex> 4x4 -> 3x3 이 된다!
        # https://www.researchgate.net/figure/Featuring-steps-to-max-pooling-Here-we-use-kernel-size-of-2-2-and-stride-of-1-ie_fig5_343356912

        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_3 = nn.Conv2d(16, 120, kernel_size=5)
        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(120, 84)
        self.fc_2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv_1(x)  # channel I:1 / O:6
        x = self.relu(x)  # activation func.
        x = self.maxpool_1(x)  # size 24 -> 12

        x = self.conv_2(x)  # channel I:6 / O:16
        x = self.relu(x)  # activation func.
        # x = self.maxpool_2(x)  # size 12 -> 6
        x = self.maxpool_1(x)  # size 24 -> 12
        # TODO: 굳이 맥스풀을 1,2로 나눈 이유가 있나..?

        x = self.conv_3(x)  # channel I:16 / O:120

        x = x.view(-1, 120)  # flatten
        x = self.fc_1(x)  # FC
        x = self.relu(x)  # activation func
        res = self.fc_2(x)  # FC

        return res