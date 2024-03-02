import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from torchvision.utils import save_image

from models.ncsnpp import NCSNpp
from configs.default_cifar10_configs import get_config
config = get_config()


clf_model_path = './origin.t7'
purification_ratio = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model_ema = model

    def forward(self, *args, **kwargs):
        return self.model_ema(*args, **kwargs)
    
dae_model = NCSNpp(config)
#dae_model = ModelWrapper(dae_model)
dae_model.load_state_dict(torch.load("/home/liuyiming/improved_consistency_models_cifar10_pytorch/ict_ema_2048e.pth", map_location='cpu'),strict=False)

# dae_model.load_state_dict(torch.load(dae_model_path, map_location=device))
dae_model.to(device)
dae_model.eval()

# 加载分类器模型
clf_model = torch.load(clf_model_path, map_location='cpu')['net'].to(device)
clf_model.eval()

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载测试数据集
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

# 准备分类器模型用于分类
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform_cifar10 = transforms.Compose(
[transforms.Normalize(mean, std)]
)

time_min: float = 0.002
total_correct = 0
total_samples = 0

# 净化和分类
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 添加高斯噪声
        noise = torch.randn_like(images) 
        noisy_images = images + noise * purification_ratio
        
        # 净化图像
        outputs = dae_model(noisy_images/((0.5**2+purification_ratio**2)**0.5), 0.25 * torch.log(torch.tensor(purification_ratio)).repeat(images.shape[0]).to(images.device))
        skip_coef = 0.5**2 / ((purification_ratio - time_min) ** 2 + 0.5**2)
        out_coef = 0.5 * (purification_ratio-time_min) / torch.sqrt(torch.tensor(purification_ratio**2 + 0.5**2))

        sample = (noisy_images * skip_coef + outputs * out_coef).clamp(-1.0, 1.0)
        clf_output = clf_model(transform_cifar10(sample/2+0.5))

        # 反归一化图像
        # noisy_images = (noisy_images.cpu().numpy() * 0.5) + 0.5
        # images = (images.cpu().numpy() * 0.5) + 0.5
        # purified_images = (purified_images.cpu().numpy() * 0.5) + 0.5

        # 保存原始图像、添加噪声后的图像和净化后的图像
        # save_image(torch.from_numpy(images.squeeze()), '/data2/final_code_v2/lkz_dae/original.png')
        # save_image(torch.from_numpy(noisy_images.squeeze()), '/data2/final_code_v2/lkz_dae/noisy.png')
        # save_image(torch.from_numpy(purified_images.squeeze()), '/data2/final_code_v2/lkz_dae/denoised.png')

        
        _, predictions = clf_output.max(1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        print(f"Accuracy: {total_correct/total_samples} - {total_correct} / {total_samples}")
