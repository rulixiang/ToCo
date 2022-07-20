import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 3 x 3 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 1 x 1 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, dilation=dilation, bias=False)

class LargeFOV(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=5):
        super(LargeFOV, self).__init__()
        self.embed_dim = 512
        self.dilation = dilation
        self.conv6 = conv3x3(in_planes=in_planes, out_planes=self.embed_dim, padding=self.dilation, dilation=self.dilation)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = conv3x3(in_planes=self.embed_dim, out_planes=self.embed_dim, padding=self.dilation, dilation=self.dilation)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = conv1x1(in_planes=self.embed_dim, out_planes=out_planes, padding=0)

    def _init_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        return None

    def forward(self, x):
        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        out = self.conv8(x)

        return out

class ASPP(nn.Module):
    def __init__(self, in_planes, out_planes, atrous_rates=[6, 12, 18, 24]):
        super(ASPP, self).__init__()
        for i, rate in enumerate(atrous_rates):
            self.add_module("c%d"%(i), nn.Conv2d(in_planes, out_planes, 3, 1, padding=rate, dilation=rate, bias=True))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
        return None
    def forward(self, x):
        return sum([stage(x) for stage in self.children()])