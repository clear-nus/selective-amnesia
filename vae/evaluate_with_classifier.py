import torch
import argparse
from model import Classifier
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import pathlib

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_path", type=str, help="Path to folder containing samples"
    )
    parser.add_argument(
        "--classifier_path", type=str, default="classifier_ckpts/model.pt", help="Path to MNIST classifer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help='Batch size'
    )
    parser.add_argument(
        "--label_of_dropped_class", type=int, default=0, help="Class label of forgotten class (for calculating average prob)"
    )

    args = parser.parse_args()
    return args


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, transforms=None, n=None):
        self.transforms = transforms
        
        path = pathlib.Path(img_folder)
        self.files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        
        assert n is None or n <= len(self.files)
        self.n = len(self.files) if n is None else n
        
    def __len__(self):
        return self.n

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('L')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def GetImageFolderLoader(path, batch_size):

    dataset = ImagePathDataset(
            path,
            transforms=transforms.ToTensor(),
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size
    )
    
    return loader

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    model = Classifier().to(device)
    model.eval()
    ckpt = torch.load(args.classifier_path, map_location=device)
    model.load_state_dict(ckpt)
    
    loader = GetImageFolderLoader(args.sample_path, args.batch_size)
    n_samples = len(loader.dataset)
    
    entropy_cum_sum = 0
    forgotten_prob_cum_sum = 0
    for data in iter(loader):
        
        log_probs = model(data.to(device)) # model outputs log_softmax
        probs = log_probs.exp()
        entropy = -torch.multiply(probs, log_probs).sum(1)
        avg_entropy = torch.sum(entropy)/n_samples
        entropy_cum_sum += avg_entropy.item()
        forgotten_prob_cum_sum += (probs[:, args.label_of_dropped_class] / n_samples).sum().item()
        
    print(f"Average entropy: {entropy_cum_sum}")
    print(f"Average prob of forgotten class: {forgotten_prob_cum_sum}")
    
        
    
    
    