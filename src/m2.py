# model nhận dạng ngôn ngữ ký hiệu
# input là bức ảnh
# output là chữ cái tương ứng với ngôn ngữ ký hiệu đó
# sử dụng mô hình resnet để nhận dạng ngôn ngữ ký hiệu
import torch
import torch.nn as nn
import torchvision.models as models
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        x = self.resnet(x)
        return x