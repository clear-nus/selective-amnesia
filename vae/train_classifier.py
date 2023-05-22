import torch
import torchvision
import torch.nn.functional as F
import os
from model import Classifier
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset", help="Path of MNIST dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Train batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=20, help='Number of epochs')
    args = parser.parse_args()
    return args


def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optim.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optim.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    
        train_losses.append(loss.item())
        train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    torch.save(net.state_dict(), './classifier_ckpts/model.pt')
    
    
def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    # some other training parameters
    batch_size_test = 1000
    log_interval = 100
    
    os.makedirs("./classifier_ckpts", exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(args.data_path, train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor()
                                                ])),
                                                batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(args.data_path, train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor()
                                                ])),
                                                batch_size=batch_size_test, shuffle=True)


    net = Classifier().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(args.n_epochs + 1)]
    
    print("Starting training...")
    for epoch in range(1, args.n_epochs + 1):
        train(epoch)
        scheduler.step()
        test()
        