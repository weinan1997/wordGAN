import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.maxLength = args['max_length']
        self.channel_num = args['channel_num']
        
        self.main = nn.Sequential(
            nn.Conv2d(1, self.channel_num, (self.maxLength//4+1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(self.channel_num),
            nn.Conv2d(self.channel_num, self.channel_num*2, (self.maxLength//4+1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(self.channel_num*2),
            nn.Conv2d(self.channel_num*2, self.channel_num*4, (self.maxLength//4+1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(self.channel_num*4),
            nn.Conv2d(self.channel_num*4, 1, (self.maxLength//4, 300), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
            return self.main(input)
