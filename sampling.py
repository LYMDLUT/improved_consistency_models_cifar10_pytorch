import torch
import sys
sys.path.append('/home/liuyiming/final_code_v2/lkz_dae/cm')
from consistency_models import ConsistencySamplingAndEditing
sys.path.append('/home/liuyiming/final_code_v2/lkz_dae')
from configs.default_cifar10_configs import get_config
from models.ncsnpp import NCSNpp



consistency_sampling_and_editing = ConsistencySamplingAndEditing(
    sigma_min = 0.002, # minimum std of noise
    sigma_data = 0.5, # std of the data
)

config = get_config()
cm_model = NCSNpp(config)
cm_model.load_state_dict(torch.load('/home/liuyiming/final_code_v2/lkz_dae/cm/ict_ema_10e_bs128.pth', map_location='cpu'))
cm_model.eval()

with torch.no_grad():
    samples = consistency_sampling_and_editing(
        cm_model, # student model or any trained model
        torch.randn((10, 3, 32, 32)), # used to infer the shapes
        sigmas=[80.0], # sampling starts at the maximum std (T)
        clip_denoised=True, # whether to clamp values to [-1, 1] range
        verbose=True,
    )
from torchvision.utils import save_image
save_image((samples/2+0.5).cpu().detach(), 'test.png')