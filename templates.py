from experiment import *



def stegan_config():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()

    conf.num_workers = 32
    conf.batch_size = 12

    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhqlmdb128'
    conf.diffusion_type = 'beatgans'
    conf.scale_up_gpus(4)

    conf.fp16 = True
    conf.lr = 1e-4

    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.sample_size = 32

    conf.T_eval = 20
    conf.T = 80

    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_enc_pool = 'adaptivenonzero'

    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.make_model_conf()
    return conf


