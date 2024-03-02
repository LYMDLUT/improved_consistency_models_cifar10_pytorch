import os
import torch
from accelerate import Accelerator
from tqdm import tqdm
import sys
sys.path.append('/home/liuyiming/improved_consistency_models_cifar10_pytorch')
from consistency_models import ConsistencySamplingAndEditing

save_dir = 'clean-ict'
os.makedirs(save_dir,exist_ok=True)

accelerator = Accelerator()
rank = accelerator.device.index
ranktotal = accelerator.num_processes


totalimg = 50000
bs = 32

consistency_sampling_and_editing = ConsistencySamplingAndEditing(
sigma_min = 0.002, # minimum std of noise
sigma_data = 0.5, # std of the data
)

from models.ncsnpp import NCSNpp
from configs.default_cifar10_configs import get_config
config = get_config()
cm_model = NCSNpp(config)
cm_model.load_state_dict(torch.load("/home/liuyiming/improved_consistency_models_cifar10_pytorch/ict_ema_4096e_21132.pth", map_location='cpu'))
cm_model = accelerator.prepare(cm_model)

for i in tqdm(range(int(totalimg / bs / ranktotal))):
    with torch.no_grad():
        samples = consistency_sampling_and_editing(
            cm_model, # student model or any trained model
            torch.randn((bs, 3, 32, 32),device=accelerator.device), # used to infer the shapes
            sigmas=[80.0], # sampling starts at the maximum std (T)
            clip_denoised=True, # whether to clamp values to [-1, 1] range
            verbose=True,
            )

    image = (samples / 2 + 0.5).clamp(0, 1)
    torch.save(image, save_dir + f'/img_batch_{i * ranktotal + rank}.pt')
