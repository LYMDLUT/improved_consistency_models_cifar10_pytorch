import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import torchvision.datasets as datasets
from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(normalize=True).to('cuda')

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
train_dataset = datasets.CIFAR10(root='../data', train=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

save_dir = "clean-ict"
for i, batch in enumerate(train_dataloader):
    fid.update(batch[0].to('cuda'), real=True)
    print(i)

for i in range(1559):
    img = torch.load(save_dir + f"/img_batch_{i}.pt", map_location='cuda')
    fid.update(img, real=False)
    print(i)

print(f"FID: {float(fid.compute())}")
