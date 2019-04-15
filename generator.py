import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.vocabSize = args['vocab_size']
        self.nz = args['nz']
        self.maxLength = args['max_length']

        self.main = nn.Sequential(
            nn.ConvTranspose1d(self.nz, self.vocabSize//2, self.maxLength//2, 1, 0, bias=False),
            nn.BatchNorm1d(self.vocabSize//2),
            nn.LeakyReLU(True),
            #nn.ConvTranspose1d(self.vocabSize*8, self.vocabSize*4, self.maxLength//4+1, 1, 0, bias=False),
            #nn.BatchNorm1d(self.vocabSize*4),
            #nn.LeakyReLU(True),
            #nn.ConvTranspose1d(self.vocabSize*4, self.vocabSize*2, self.maxLength//4+1, 1, 0, bias=False),
            #nn.BatchNorm1d(self.vocabSize*2),
            #nn.LeakyReLU(True),
            nn.ConvTranspose1d(self.vocabSize//2, self.vocabSize, self.maxLength//2+1, 1, 0, bias=False),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)
