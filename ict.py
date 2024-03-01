import torch
import sys
from accelerate import Accelerator
from copy import deepcopy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from models.ncsnpp import NCSNpp
from configs.default_cifar10_configs import get_config
from consistency_models import ConsistencySamplingAndEditing, ImprovedConsistencyTraining, pseudo_huber_loss
from utils import update_ema_model_
from torchmetrics.image.fid import FrechetInceptionDistance

if __name__ == "__main__":
    # create model
    config = get_config()
    cm_model = NCSNpp(config)
    #cm_model.load_state_dict(torch.load('./checkpoint_74.pth', map_location='cpu'))
    cm_model_ema = deepcopy(cm_model)
    cm_model.train()
    for param in cm_model_ema.parameters():
        param.requires_grad = False
    cm_model_ema.eval()
    
    accelerator = Accelerator()
    batch_size = 128
    num_epochs = 4096
    # load dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    fid_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
    # 定义数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    fid_loader = torch.utils.data.DataLoader(dataset=fid_dataset, batch_size=batch_size, shuffle=False)
    
    # create optimizer
    optimizer = torch.optim.RAdam(cm_model.parameters(), lr=1e-4, betas=(0.9, 0.999)) # setup your optimizer
    scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-5,
            total_iters=1000,
        )
    
    # Initialize the training module using
    improved_consistency_training = ImprovedConsistencyTraining(
        sigma_min = 0.002, # minimum std of noise
        sigma_max = 80.0, # maximum std of noise
        rho = 7.0, # karras-schedule hyper-parameter
        sigma_data = 0.5, # std of the data
        initial_timesteps = 10, # number of discrete timesteps during training start
        final_timesteps = 1280, # number of discrete timesteps during training end
        lognormal_mean = -1.1, # mean of the lognormal timestep distribution
        lognormal_std = 2.0, # std of the lognormal timestep distribution
    )
    
    consistency_sampling_and_editing = ConsistencySamplingAndEditing(
    sigma_min = 0.002, # minimum std of noise
    sigma_data = 0.5, # std of the data
    )
    fid = FrechetInceptionDistance(reset_real_features=False, normalize=True).to(accelerator.device)
    for i, batch in enumerate(fid_loader):
        fid.update(batch[0].to(accelerator.device), real=True)
    torch.cuda.empty_cache() 
    cm_model, cm_model_ema, optimizer, train_loader, pseudo_huber_loss, scheduler, improved_consistency_training, fid = accelerator.prepare(cm_model, cm_model_ema, optimizer, train_loader, pseudo_huber_loss, scheduler, improved_consistency_training, fid)

    current_training_step = 0
    total_steps = len(train_loader)
    total_training_steps = num_epochs * len(train_loader)
    
    for epoch in range(num_epochs):
        for i, (images, lablel) in enumerate(train_loader):
            # Zero out Grads
            optimizer.zero_grad()
            # Forward Pass
            
            output = improved_consistency_training(
                cm_model,
                images,
                current_training_step,
                total_training_steps,
            )

            # Loss Computation
            loss = (pseudo_huber_loss(output.predicted, output.target) * output.loss_weights).mean()

            # Backward Pass & Weights Update
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            update_ema_model_(cm_model_ema, cm_model, 0.9999)
            torch.distributed.barrier()
            current_training_step = current_training_step + 1
            if accelerator.process_index == 0 and i % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')
                
        if accelerator.process_index == 0:
            print('训练比例',current_training_step/total_training_steps*100,'%') 
            unet_ema = accelerator.unwrap_model(cm_model_ema)
            torch.save(unet_ema.state_dict(), f'ict_ema_4096e.pth')
            if epoch % 50 == 0 and epoch != 0:
                torch.save(unet_ema.state_dict(), f'ict_ema_4096e_{epoch}e.pth')
            unet = accelerator.unwrap_model(cm_model)
            torch.save(unet.state_dict(), f'ict_4096e.pth')
            if epoch % 50 == 0 and epoch != 0:
                torch.save(unet.state_dict(), f'ict_4096e_{epoch}e.pth')
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
            save_image((samples/2+0.5).cpu().detach(), f'ict_images_{epoch}.png')
        
        if epoch % 10 == 0:
            for i in range(int(50000 / batch_size / accelerator.num_processes)):
                with torch.no_grad():
                    samples = consistency_sampling_and_editing(
                        cm_model_ema, # student model or any trained model
                        torch.randn((batch_size, 3, 32, 32), device=accelerator.device), # used to infer the shapes
                        sigmas=[80.0], # sampling starts at the maximum std (T)
                        clip_denoised=True, # whether to clamp values to [-1, 1] range
                        verbose=True)
                image = (samples / 2 + 0.5).clamp(0, 1)  
                fid.update(accelerator.gather(image), real=False)
            fid_result = float(fid.compute())
            if accelerator.is_main_process:
                print(f"FID: {fid_result}")
            fid.reset()
            torch.cuda.empty_cache()
