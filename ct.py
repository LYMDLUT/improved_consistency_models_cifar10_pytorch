import torch
import sys
from torch import nn
from typing import Tuple, Union
from accelerate import Accelerator
from copy import deepcopy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.ncsnpp import NCSNpp
from configs.default_cifar10_configs import get_config
from consistency_models import ConsistencySamplingAndEditing, ConsistencyTraining, ema_decay_rate_schedule
from utils import update_ema_model_

class PerceptualLoss(nn.Module):
    def __init__(
        self,
        *,
        net_type: Union[str, Tuple[str, ...]] = "vgg",
        l1_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        available_net_types = ("vgg", "alex", "squeeze")

        def _append_net_type(net_type: str):
            if net_type in available_net_types:
                self.lpips_losses.append(
                    LearnedPerceptualImagePatchSimilarity(net_type)
                )
            else:
                raise TypeError(
                    f"'net_type' should be on of {available_net_types}, got {net_type}"
                )

        self.lpips_losses = nn.ModuleList()

        if isinstance(net_type, str):
            _append_net_type(net_type)

        elif isinstance(net_type, Sequence):
            for _net_type in sorted(net_type):
                _append_net_type(_net_type)

        self.lpips_losses.requires_grad_(False)

        self.l1_weight = l1_weight

    def forward(self, input, target):
        upscaled_input = F.interpolate(input, (224, 224), mode="bilinear")
        upscaled_target = F.interpolate(target, (224, 224), mode="bilinear")

        lpips_loss = sum(
            _lpips_loss(upscaled_input, upscaled_target)
            for _lpips_loss in self.lpips_losses
        )

        return lpips_loss + self.l1_weight * F.l1_loss(input, target)


if __name__ == "__main__":
    # create model
    config = get_config()
    cm_model = NCSNpp(config)
    #cm_model.load_state_dict(torch.load('/home/liuyiming/final_code_v2/convert_ckpt/ct-lpips/checkpoint_74.pth', map_location='cpu'))
    cm_model_ema = deepcopy(cm_model)
    ema_student_model = deepcopy(cm_model)
    cm_model.train()
    for param in cm_model_ema.parameters():
        param.requires_grad = False
    cm_model_ema.eval()
    for param in ema_student_model.parameters():
        param.requires_grad = False
    ema_student_model.eval()
    
    batch_size = 20
    num_epochs = 4096
    # load dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

    # 定义数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # create optimizer
    optimizer = torch.optim.Adam(cm_model.parameters(), lr=1e-4, betas=(0.9, 0.995)) # setup your optimizer
    scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-5,
            total_iters=100,
        )
    
    # Initialize the training module using
    consistency_training = ConsistencyTraining(
        sigma_min = 0.002, # minimum std of noise
        sigma_max = 80.0, # maximum std of noise
        rho = 7.0, # karras-schedule hyper-parameter
        sigma_data = 0.5, # std of the data
        initial_timesteps = 2, # number of discrete timesteps during training start
        final_timesteps = 150, # number of discrete timesteps during training end
    )
    
    consistency_sampling_and_editing = ConsistencySamplingAndEditing(
    sigma_min = 0.002, # minimum std of noise
    sigma_data = 0.5, # std of the data
    )
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
    accelerator = Accelerator()
    cm_model, cm_model_ema, ema_student_model, optimizer, test_loader, train_loader, scheduler, lpips, consistency_training = accelerator.prepare(cm_model, cm_model_ema, ema_student_model, optimizer, test_loader, train_loader, scheduler, lpips, consistency_training)
    
    current_training_step = 0
    total_steps = len(train_loader)
    total_training_steps = num_epochs * len(train_loader)
    epoch_temp = 0
    for epoch in tqdm(range(num_epochs)):
        for i, (images, lablel) in enumerate(train_loader):
            # Zero out Grads
            optimizer.zero_grad()
            # Forward Pass
            output = consistency_training(
                cm_model,
                cm_model_ema,
                images,
                current_training_step,
                total_training_steps,
            )

            # Loss Computation
            lpips_loss = lpips(
            output.predicted.clamp(-1.0, 1.0), output.target.clamp(-1.0, 1.0))
            loss = lpips_loss
            
            # Backward Pass & Weights Update
            # Backward Pass & Weights Update
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            current_training_step = current_training_step + 1
            # EMA Update
            ema_decay_rate = ema_decay_rate_schedule(
                output.num_timesteps,
                initial_ema_decay_rate=0.95,
                initial_timesteps=consistency_training.initial_timesteps,
            )
            update_ema_model_(cm_model_ema, cm_model, ema_decay_rate)
            
            # Update EMA student model
            update_ema_model_(
                ema_student_model,
                cm_model,
                0.99993,
            )
            
            if accelerator.process_index == 0 and epoch_temp != epoch:
                epoch_temp = epoch
                print('训练比例',current_training_step/total_training_steps*100,'%')
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')
                unet_ema = accelerator.unwrap_model(cm_model_ema)
                torch.save(unet_ema.state_dict(), f'ct_ema_4096e.pth')
                if epoch % 1000 == 0:
                    torch.save(unet_ema.state_dict(), f'ct_ema_4096e_{epoch}e.pth')
                unet = accelerator.unwrap_model(cm_model)
                torch.save(unet.state_dict(), f'ct_4096e.pth')
                if epoch % 1000 == 0:
                    torch.save(unet.state_dict(), f'ct_4096e_{epoch}e.pth')
                unet_ema.eval()
                with torch.no_grad():
                    samples = consistency_sampling_and_editing(
                        unet_ema, # student model or any trained model
                        torch.randn((8, 3, 32, 32),device=accelerator.device), # used to infer the shapes
                        sigmas=[80.0], # sampling starts at the maximum std (T)
                        clip_denoised=True, # whether to clamp values to [-1, 1] range
                        verbose=True,
                    )
                from torchvision.utils import save_image
                save_image((samples/2+0.5).cpu().detach(), f'ct_images_{epoch}.png')
