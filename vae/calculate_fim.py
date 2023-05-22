# prerequisites
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import pickle
import tqdm
import argparse
import logging
import os

from model import OneHotCVAE, loss_function


def parse_args_and_ckpt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--ckpt_folder", type=str, required=True, help="Path to folder of original VAE"
    )
    
    parser.add_argument(
        "--n_fim_samples", type=int, default=50000, help="Number of samples to calculate FIM with. Only applicable for true FIM."
    )
    
    args = parser.parse_args()
    ckpt = torch.load(os.path.join(args.ckpt_folder, "ckpts/ckpt.pt"), map_location=device)
    
    return args, ckpt


def save_fim():
    fisher_dict = {}
    params_mle_dict = {}
    
    for name, param in vae.named_parameters():
        params_mle_dict[name] = param.data.clone()
        fisher_dict[name] = param.data.clone().zero_()
    
    for _ in tqdm.tqdm(range(args.n_fim_samples)):
        
        with torch.no_grad():
            z = torch.randn(1, config.z_dim).to(device)
            c = torch.randint(0,10, (1,)).to(device)
            c = F.one_hot(c, 10)
            vae.eval()
            sample = vae.decoder(z, c)
        
        vae.train()
        vae.zero_grad()
        recon_batch, mu, log_var = vae(sample, c)
        loss = loss_function(recon_batch, sample, mu, log_var)
        loss.backward()

        for name, param in vae.named_parameters():
            if torch.isnan(param.grad.data).any():
                print("NAN detected")
            fisher_dict[name] += ((param.grad.data) ** 2) / args.n_fim_samples
        
    with open(os.path.join(config.exp_root_dir, 'fisher_dict.pkl'), 'wb') as f:
        pickle.dump(fisher_dict, f)
    # with open(os.path.join(config.exp_root_dir, 'params_mle_dict.pkl'), 'wb') as f:
    #     pickle.dump(params_mle_dict, f)


def one_class_mnist_dataset(class_label):

    train_dataset = datasets.MNIST(
        './dataset',
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    
    test_dataset = datasets.MNIST(
        './dataset',
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    def find_indices(lst, condition):
        return [i for i, elem in enumerate(lst) if elem == condition]

    train_idx = find_indices(train_dataset.targets, class_label)
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True, drop_last=False)
    
    test_idx = find_indices(test_dataset.targets, class_label)
    test_subset = torch.utils.data.Subset(test_dataset, test_idx)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=True, drop_last=False)
    
    return train_loader, test_loader

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args, ckpt = parse_args_and_ckpt()
    config = ckpt['config']

    # build model
    vae = OneHotCVAE(x_dim=config.x_dim, h_dim1= config.h_dim1, h_dim2=config.h_dim2, z_dim=config.z_dim)
    vae = vae.to(device)

    vae.load_state_dict(ckpt['model'])
    vae.train()
    save_fim()