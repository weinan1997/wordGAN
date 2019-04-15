import torch
import torch.nn as nn
import torch.utils.data
import preprocess
import numpy as np
import argparse
import generator
import discriminator
import train
import os

def readData(maxLength):
    lang = preprocess.Lang()
    lines = lang.readFile("data/image_coco.txt")
    indexed_texts = np.array(lang.tokenize(lines, maxLength))
    W = lang.loadWord2Vec("data/GoogleNews-vectors-negative300.bin.gz")
    return indexed_texts, W


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_length', default=40, type=int, help="Max length for data preprocess")
    parser.add_argument('-b', '--batch_size', default=128, type=int, help="Batch Size")
    parser.add_argument('-e', '--epoch_num', default=50, type=int, help="Epoch number")
    parser.add_argument('-l', '--learning_rate', default=0.0002, type=float, help="Learning rate for optimizer")
    parser.add_argument('--beta1', default=0.5, type=float, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument('-z', '--hidden_vec', default=100, type=int, help="Hidden vector length")
    parser.add_argument('-c', '--channel_num', default=100, type=int, help="Channel number for discriminator")

    options = parser.parse_args()
    args = {
        'max_length': options.max_length,
        'batch_size': options.batch_size,
        'epoch_num':  options.epoch_num,
        'learning_rate': options.learning_rate,
        'beta1': options.beta1,
        'hidden_vec': options.hidden_vec,
        'channel_num': options.channel_num
    }
    return args

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

  
def main():
    args = parseArgs()
    if not os.path.exists("data/processed.data"):
        train_text, W = readData(args['max_length'])
        torch.save([train_text, W], "data/processed.data")
    else:
        train_text, W = torch.load("data/processed.data")      
    train_text = torch.tensor(train_text)
    W = torch.tensor(W)
    target = torch.full((train_text.shape[0],1,1,1), 1)
    data = torch.utils.data.TensorDataset(train_text, target)
    dataloader = torch.utils.data.DataLoader(data, batch_size=args['batch_size'], shuffle=True)

    Gargs = {
        'vocab_size': W.shape[0],
        'nz': args['hidden_vec'],
        'max_length': args['max_length']
    }
    netG = generator.Generator(Gargs)
    Dargs = {
        'max_length': args['max_length'],
        'channel_num': args['channel_num']
    }
    netD = discriminator.Discriminator(Dargs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netD.to(device)
    netG.to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    Targs = {
        'learning_rate': args['learning_rate'],
        'beta1': args['beta1'],
        'epoch_num': args['epoch_num'],
        'device': device,
        'word2vec': W,
        'batch_size': args['batch_size'],
        'nz': args['hidden_vec']
    }

    G_losses, D_losses = train.train(netD, netG, dataloader, Targs)
    train.plotLoss(G_losses, D_losses)

if __name__ == "__main__":
    main()
