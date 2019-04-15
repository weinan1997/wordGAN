import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train(netD, netG, dataloader, args):
    lr = args['learning_rate']
    beta1 = args['beta1']
    epoch_num = args['epoch_num']
    device = args['device']
    word2vec = args['word2vec'].to(device)
    nz = args['nz']
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    G_losses = []
    D_losses = []

    print('Start training...')
    for epoch in range(epoch_num):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_data = data[0].to(device)
            label = data[1].to(device)
            embedded_data = F.embedding(real_data, word2vec)
            embedded_data.unsqueeze_(1)
            output = netD(embedded_data)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()
            noise = torch.randn(real_data.shape[0], nz, 1, device=device)
            fake = netG(noise)
            fake = fake.permute(0, 2, 1)
            fake = torch.matmul(fake, word2vec)
            fake.unsqueeze_(1)
            label.fill_(0)
            output = netD(fake.detach())
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optD.step()

            netG.zero_grad()
            label.fill_(1)
            output = netD(fake)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, epoch_num, i, len(dataloader),
                        lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

    return G_losses, D_losses

def plotLoss(G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.save('loss.png')
