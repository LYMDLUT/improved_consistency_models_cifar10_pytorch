import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "CIFAR10"
    data.image_size = 32
    data.random_flip = True
    data.uniform_dequantization = False
    data.num_channels = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.02
    model.sigma_max = 100
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.t_min = 0.002
    model.t_max = 80.0
    model.embedding_type = "fourier"
    model.double_heads = False

    model.name = "ncsnpp"
    model.ema_rate = 0.9999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "none"
    model.progressive_input = "residual"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.conv_size = 3
    model.rho = 7.0
    model.data_std = 0.5
    model.num_scales = 18
    model.dropout = 0.0
    return config
