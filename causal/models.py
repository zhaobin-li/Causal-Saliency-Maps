"""get models"""

from torch import nn
from torchvision.models import resnet18


def get_cnn_with_softmax():
    model = resnet18(pretrained=True)
    # model = resnet50(pretrained=True)
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.eval()
    return model
