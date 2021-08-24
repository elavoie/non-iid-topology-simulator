import torch
import torch.nn as nn
import torch.nn.functional as F


class GN_LeNet(nn.Module):
    """
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791
    Layer parameters taken from: https://github.com/kevinhsieh/non_iid_dml/blob/master/apps/caffe/examples/cifar10/1parts/gnlenet_train_val.prototxt.template
    (with Group Normalisation).
    Results for previous model described in http://proceedings.mlr.press/v119/hsieh20a.html
    """

    def __init__(self, input_channel=3, output=10, args=None):
        super(GN_LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.GroupNorm(2, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(2, 32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(2, 64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        if args == None or args.dataset == 'cifar10':
            self.classifier = nn.Sequential(
                nn.Linear(576, output),
            )
        elif args.dataset == 'mnist':
            self.classifier = nn.Sequential(
                nn.Linear(256, output),
            )
        else:
            raise Exception("Invalid dataset: {}".format(args.dataset))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

