from experiment import *
from choices import *


def stegan_config(debug: bool = False, scale_up_gpus: int = 1):
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.debug = debug

    conf.model_type = ModelType.stegan
    conf.stegan_type = SteganType.semantics
    conf.encoder_pretrain = 'checkpoints/ffhq128_autoenc_130M/last.ckpt'

    conf.num_workers = 16 if not debug else 0
    conf.batch_size = 48 if not debug else 1
    conf.img_size = 64

    conf.name = ''
    conf.model_name = ModelName.steganography
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhqlmdb128_steg'
    conf.diffusion_type = 'beatgans'
    conf.scale_up_gpus(scale_up_gpus)

    conf.fp16 = True
    conf.lr = 1e-4

    conf.eval_every_samples = 10_000_000
    conf.sample_every_samples = 20_000
    conf.sample_size = 32

    conf.T_eval = 80
    conf.T = 80

    conf.net_attn = (16,)
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_beatgans_resnet_three_cond = True
    conf.net_enc_pool = 'adaptivenonzero'

    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_out = 1024 if conf.stegan_type == SteganType.images else 512
    conf.enc_in_channels = 6 if conf.stegan_type == SteganType.images else 3
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.make_model_conf()
    return conf
