import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)
        self.dropout = nn.Dropout(p=0.5)
        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll


class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_angles = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_angles(x)
        return x


class AlexNet(nn.Module):
    # AlexNet laid out as a Hopenet - classify Euler angles in bins and
    # regress the expected value.
    def __init__(self, num_bins):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc_yaw = nn.Linear(4096, num_bins)
        self.fc_pitch = nn.Linear(4096, num_bins)
        self.fc_roll = nn.Linear(4096, num_bins)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        return yaw, pitch, roll


class Hopenet_VAE(Hopenet):

    def __init__(self, block, layers, num_bins, alpha, beta):
        super(Hopenet_VAE, self).__init__(block, layers, num_bins)
        super(Hopenet_VAE, self)

        self.yaw_mean = nn.Linear(512 * block.expansion, 512)
        self.yaw_logvar = nn.Linear(512 * block.expansion, 512)
        self.roll_mean = nn.Linear(512 * block.expansion, 512)
        self.roll_logvar = nn.Linear(512 * block.expansion, 512)
        self.pitch_mean = nn.Linear(512 * block.expansion, 512)
        self.pitch_logvar = nn.Linear(512 * block.expansion, 512)

        self.dropout(p=0.5)

        self.fc_yaw = nn.Linear(512, num_bins)
        self.fc_pitch = nn.Linear(512, num_bins)
        self.fc_roll = nn.Linear(512, num_bins)

        self.CEL = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

        self.alpha = torch.tensor(alpha, dtype=torch.float)
        self.beta = torch.tensor(beta, dtype=torch.float)

        idx_tensor = [idx for idx in xrange(66)]
        self.idx_tensor = torch.tensor(idx_tensor)

        self.register_buffer('idx_tensor', self.idx_tensor)
        self.register_buffer('alpha', self.alpha)
        self.register_buffer('beta', self.beta)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        pre_yaw_ = self.yaw_mean(x)
        logvar_yaw = self.yaw_logvar(x)

        pre_roll_ = self.roll_mean(x)
        logvar_roll = self.roll_logvar(x)

        pre_pitch_ = self.pitch_mean(x)
        logvar_pitch = self.pitch_logvar(x)

        pre_yaw = self.reparameterize(pre_yaw_, logvar_yaw)
        pre_pitch = self.reparameterize(pre_pitch_, logvar_pitch)
        pre_roll = self.reparameterize(pre_roll_, logvar_roll)

        pre_yaw = self.fc_yaw(pre_yaw)
        pre_pitch = self.fc_pitch(pre_pitch)
        pre_roll = self.fc_pitch(pre_roll)

        return (pre_yaw, pre_yaw_, logvar_yaw), (pre_pitch, pre_pitch_, logvar_pitch), (pre_roll, pre_roll_, logvar_roll)

    def reparameterize(self, mean, logvar):

        if self.training:
            std = torch.exp(logvar*0.5)
            norm = torch.randn_like(std, requires_grad=False)
            return mean + std*norm
        else:
            return mean

    def loss_function(self, mean, logvar, pre, label, label_cont):
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar))
        pre = self.softmax(pre)
        CEL = self.CEL(pre, label)
        pre = self.softmax(pre)
        pre = torch.sum(pre * self.idx_tensor, 1) * 3 - 99
        MSE = self.MSE(pre, label_cont)

        return KLD + self.alpha*CEL + self.beta*MSE


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        def conv_sd(in_channels, out_channels, stride, kernel_size):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def conv_dw(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                          stride=stride, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channel=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            
            conv_sd(3, 32, 2, 3),
            conv_dw(32, 32, 1),
            conv_sd(32, 64, 1, 1),
            conv_dw(64, 64, 2),

            conv_sd(64, 128, 1, 1),
            conv_dw(128, 128, 1),
            conv_sd(128, 128, 1, 1),
            conv_dw(128, 128, 2),

            conv_sd(64, 128, 1, 1),
            conv_dw(128, 128, 1),
            conv_sd(128, 128, 1, 1),
            conv_dw(128, 128, 2),

            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """ https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py """
    def __init__(self, n_class=66, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        input_channel = int(32)
        last_channel = int(1280)

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2_angle_header(MobileNetV2):

    def __init__(self, n_class=66, input_size=224, width_mult=1.):
        super(MobileNetV2_angle_header, self).__init__(n_class=n_class, input_size=input_size, width_mult=width_mult)
        super(MobileNetV2_angle_header, self)
        self._initialize_weights()
        self.dropout = nn.Dropout(p=0.5)
        self.fc_yaw = nn.Linear(self.last_channel, n_class)
        self.fc_pitch = nn.Linear(self.last_channel, n_class)
        self.fc_roll = nn.Linear(self.last_channel, n_class)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)

        x = self.dropout(x)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll
