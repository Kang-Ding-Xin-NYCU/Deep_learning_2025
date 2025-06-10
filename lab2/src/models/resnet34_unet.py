import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input  = self.input_layer(3, 64)
        self.layer1 = self.make_layer(64, 64, 3, stride=1)
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)
        self.layer5 = self.bottleneck(512, 256, 2, stride=1)

    def input_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def bottleneck(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input(x)
        e1 = self.layer1(x)  # 64x64x64
        e2 = self.layer2(e1) # 128x32x32
        e3 = self.layer3(e2) # 256x16x16
        e4 = self.layer4(e3) # 512x8x8
        e5 = self.layer5(e4) # 256x8x8

        return e1, e2, e3, e4, e5

class ResNet34UNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.encoder = ResNet34Encoder()
        
        self.decoder4 = self.conv_block(512 + 256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 128, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 64, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 32, 64)

        self.upconv0 = self.final_block(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def final_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)

        d4 = self.decoder4(torch.cat([e4, e5], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))
        d0 = self.upconv0(d1)

        return torch.sigmoid(self.final(d0))
