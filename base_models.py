from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from model.unet_autoenc import BeatGANsAutoencConfig, BeatGANsEncoderConfig
from renderer import render_condition
from utils import BaseReturn, show_tensor_image
from config_base import BaseConfig
from choices import SteganType


@dataclass
class SteganConfig(BaseConfig):
    encoder_pretrain: str = None
    stegan_type: SteganType = SteganType.images
    image_size: int = 128
    in_channels: int = 3
    enc_in_channels: int = 6
    dec_in_channels: int = 3
    enc_cond_vec_size: int = 1024
    dec_cond_vec_size: int = 512
    net_enc_pool: str = 'adaptivenonzero'
    # base channels, will be multiplied
    model_channels: int = 64
    # output of the unet
    # suggest: 3
    # you only need 6 if you also model the variance of the noise prediction (usually we use an analytical variance hence 3)
    out_channels: int = 3
    # how many repeating resblocks per resolution
    # the decoding side would have "one more" resblock
    # default: 2
    num_res_blocks: int = 2
    # you can also set the number of resblocks specifically for the input blocks
    # default: None = above
    num_input_res_blocks: int = None
    # number of time embed channels and style channels
    embed_channels: int = 512
    # at what resolutions you want to do self-attention of the feature maps
    # attentions generally improve performance
    # default: [16]
    # beatgans: [32, 16, 8]
    attention_resolutions: Tuple[int] = (16,)
    # number of time embed channels
    time_embed_channels: int = None
    # dropout applies to the resblocks (on feature maps)
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    input_channel_mult: Tuple[int] = None
    conv_resample: bool = True
    # always 2 = 2d conv
    dims: int = 2
    # don't use this, legacy from BeatGANs
    num_classes: int = None
    use_checkpoint: bool = False
    # number of attention heads
    num_heads: int = 1
    # or specify the number of channels per attention head
    num_head_channels: int = -1
    # what's this?
    num_heads_upsample: int = -1
    # use resblock for upscale/downscale blocks (expensive)
    # default: True (BeatGANs)
    resblock_updown: bool = True
    # never tried
    use_new_attention_order: bool = False
    resnet_two_cond: bool = False
    resnet_three_cond: bool = False
    resnet_cond_channels: int = None
    # init the decoding conv layers with zero weights, this speeds up training
    # default: True (BeattGANs)
    resnet_use_zero_module: bool = True
    # gradient checkpoint the attention operation
    attn_checkpoint: bool = False

    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: Tuple[int] = None
    net_enc_grad_checkpoint: bool = False
    net_enc_attn_resolutions: Tuple[int] = None
    net_enc_out: int = 512

    def add_base(self):
        super().inherit(self)

    def make_model(self):
        self.add_base()
        return BaseModel(self)


class BaseModel(nn.Module):
    def __init__(self, conf, stop_pretrain_loading=False):
        super().__init__()
        self.conf = conf
        self.stegan_type = self.conf.stegan_type

        if self.stegan_type == SteganType.images:
            self._setup_images_stegan_model(self.conf)
        elif self.stegan_type == SteganType.semantics:
            self._setup_semantics_stegan_model(self.conf)

    def _setup_images_stegan_model(self, conf):
        self.encoder = self._setup_encoder_by_conf(conf)
        self.decoder = self._setup_decoder_by_conf(conf)

    def _setup_semantics_stegan_model(self, conf):
        self.encoder = self._setup_encoder_by_conf(conf)
        self.decoder = self._setup_decoder_by_conf(conf)

        #self.decoder = self._setup_semantic_decoder_by_conf(conf)
        if conf.encoder_pretrain:
            print(f'loading pretrain ... {conf.encoder_pretrain}')
            state = torch.load(conf.encoder_pretrain, map_location='cpu')
            new_state_dict = {}
            for k, v in state['state_dict'].items():
                if k == 'x_T':
                    pass
                else:
                    new_state_dict[k.replace('model.', '')] = v
            print('step:', state['global_step'])
            self.encoder.load_state_dict(new_state_dict, strict=False)
            for n, p in self.encoder.named_parameters():
                if n.startswith('encoder'):
                    p.requires_grad = False
            print('Freezed the Encoder, encoder')

            new_state_dict = {}
            for k, v in state['state_dict'].items():
                if k == 'x_T':
                    pass
                else:
                    if not k.startswith('encoder'):
                        new_state_dict[k.replace('model.', '')] = v
            print('step:', state['global_step'])
            self.decoder.load_state_dict(new_state_dict, strict=False)


    def _setup_encoder_by_conf(self, conf):
        model = BeatGANsAutoencConfig(
            attention_resolutions=conf.attention_resolutions,
            channel_mult=conf.channel_mult,
            conv_resample=True,
            dims=2,
            dropout=conf.dropout,
            embed_channels=conf.embed_channels,
            enc_out_channels=conf.net_enc_out,
            enc_pool=conf.net_enc_pool,
            enc_num_res_block=conf.net_enc_num_res_blocks,
            enc_channel_mult=conf.net_enc_channel_mult,
            enc_grad_checkpoint=conf.net_enc_grad_checkpoint,
            enc_attn_resolutions=conf.net_enc_attn_resolutions,
            enc_in_channels=conf.enc_in_channels,
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            num_classes=None,
            num_head_channels=-1,
            num_heads_upsample=-1,
            num_heads=conf.num_heads,
            num_res_blocks=conf.num_res_blocks,
            num_input_res_blocks=conf.num_input_res_blocks,
            out_channels=conf.out_channels,
            resblock_updown=conf.resblock_updown,
            use_checkpoint=conf.use_checkpoint,
            use_new_attention_order=False,
            resnet_two_cond=conf.resnet_two_cond,
            resnet_three_cond=conf.resnet_three_cond,
            resnet_use_zero_module=conf.resnet_use_zero_module,
            latent_net_conf=None,
            resnet_cond_channels=conf.enc_cond_vec_size,
        ).make_model()
        return model

    def _setup_decoder_by_conf(self, conf):
        model = BeatGANsAutoencConfig(
            attention_resolutions=conf.attention_resolutions,
            channel_mult=conf.channel_mult,
            conv_resample=True,
            dims=2,
            dropout=conf.dropout,
            embed_channels=conf.embed_channels,
            enc_out_channels=conf.net_enc_out,
            enc_pool=conf.net_enc_pool,
            enc_num_res_block=conf.net_enc_num_res_blocks,
            enc_channel_mult=conf.net_enc_channel_mult,
            enc_grad_checkpoint=conf.net_enc_grad_checkpoint,
            enc_attn_resolutions=conf.net_enc_attn_resolutions,
            image_size=conf.image_size,
            in_channels=conf.dec_in_channels,
            model_channels=conf.model_channels,
            num_classes=None,
            num_head_channels=-1,
            num_heads_upsample=-1,
            num_heads=conf.num_heads,
            num_res_blocks=conf.num_res_blocks,
            num_input_res_blocks=conf.num_input_res_blocks,
            out_channels=conf.out_channels,
            resblock_updown=conf.resblock_updown,
            use_checkpoint=conf.use_checkpoint,
            use_new_attention_order=False,
            resnet_two_cond=conf.resnet_two_cond,
            resnet_three_cond=False,
            resnet_use_zero_module=conf.resnet_use_zero_module,
            latent_net_conf=None,
            resnet_cond_channels=conf.dec_cond_vec_size,
        ).make_model()
        return model

    def _setup_semantic_decoder_by_conf(self, conf):
        model = BeatGANsEncoderConfig(image_size=conf.image_size, in_channels=conf.dec_in_channels,
                                      model_channels=conf.model_channels, out_hid_channels=conf.net_enc_out,
                                      out_channels=conf.net_enc_out, num_res_blocks=conf.num_res_blocks,
                                      attention_resolutions=(
                                                  conf.net_enc_attn_resolutions or conf.attention_resolutions),
                                      dropout=conf.dropout,
                                      channel_mult=conf.net_enc_channel_mult or conf.channel_mult,
                                      use_time_condition=False, conv_resample=conf.conv_resample, dims=conf.dims,
                                      use_checkpoint=conf.use_checkpoint or conf.net_enc_grad_checkpoint,
                                      num_heads=conf.num_heads, num_head_channels=conf.num_head_channels,
                                      resblock_updown=conf.resblock_updown,
                                      use_new_attention_order=conf.use_new_attention_order,
                                      pool=conf.net_enc_pool).make_model()
        return model

    def forward(self, cover, hide, c_noise, sampler=None):
        assert (sampler), 'Need to define a sampelr'
        cover = cover.unsqueeze(0) if len(cover.shape) < 4 else cover
        hide = hide.unsqueeze(0) if len(hide.shape) < 4 else hide
        c_noise = c_noise.unsqueeze(0) if len(c_noise.shape) < 4 else c_noise

        with torch.no_grad():
            # Use concat to combine the inputs, and detach because we don't want to train the encoders
            if self.stegan_type == SteganType.images:
                x_concat_sem = torch.concat((cover, hide), dim=1)
                encode_cond = self.encoder.encode(x_concat_sem)['cond']
                encoded = render_condition(self.encoder, c_noise, sampler=sampler, cond=encode_cond, h_cond=None)
                decode_cond = self.decoder.encode(encoded)['cond']
                decoded = render_condition(self.decoder, c_noise, sampler=sampler, cond=decode_cond, h_cond=None)
            elif self.stegan_type == SteganType.semantics:
                encode_cond = self.encoder.encode(cover)['cond'].detach()
                h_cond = self.encoder.encode(hide)['cond'].detach()
                encoded = render_condition(self.encoder, c_noise, sampler=sampler, cond=encode_cond, h_cond=h_cond)
                decode_cond = self.decoder.encode(encoded)['cond']
                decoded = render_condition(self.decoder, c_noise, sampler=sampler, cond=decode_cond, h_cond=None)
            else:
                raise Exception('Not implemented')




        return BaseReturn(encoded=encoded, decoded=decoded)
