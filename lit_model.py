import copy
import json
import re

import numpy as np
import pytorch_lightning as pl
import torch.nn
from torch.optim.optimizer import Optimizer
from torch.autograd import grad
from torch.cuda import amp
from torchvision.utils import make_grid, save_image

from lmdb_writer import *
from config import TrainConfig, TrainMode
from dist_utils import get_world_size
from choices import OptimizerType, SteganType
from utils import WarmupLR, show_tensor_image, is_time
from metrics import evaluate_fid, evaluate_lpips


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf
        self.stegan_type = self.conf.stegan_type
        self.model = conf.make_model_conf().make_model()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        # DiffAE Leftovers which might be possible to delete
        self.latent_sampler = None
        self.eval_latent_sampler = None
        self.conds_mean = None
        self.conds_std = None

        self.mse_loss = torch.nn.MSELoss()
        if conf.pretrain is not None:
            print(f'loading pretrain ... {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print('step:', state['global_step'])
            self.load_state_dict(state['state_dict'], strict=False)

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def encode(self, x):
        assert self.conf.model_type.has_autoenc()
        cond = self.ema_model.encoder.forward(x)
        return cond

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_model, x, model_kwargs={'cond': cond})
        return out['sample']

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            model = self.model
            gen = self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.conf.make_dataset(split='train')
        print('train data:', len(self.train_data))
        self.val_data = self.conf.make_dataset(split='val')
        print('val data:', len(self.val_data))

    def _train_dataloader(self, drop_last=True):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        dataloader = conf.make_loader(self.train_data, shuffle=True, drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        """
        """
        print('on train dataloader start ...')
        return self._train_dataloader()

    def _val_dataloader(self, drop_last=True):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        dataloader = conf.make_loader(self.val_data, shuffle=False, drop_last=drop_last)
        return dataloader

    def val_dataloader(self):
        """
        """
        print('on bsl dataloader start ...')
        return self._val_dataloader()

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop?
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast(False):
            cover, hide, noise = batch['cover'], batch['hide'], batch['noise']
            batch_size = cover.shape[0]
            device = cover.device

            if self.conf.sample_on_train_start:
                self.log_sample(cover=cover, hide=hide, noise=noise, mode='log_on_start')
                self.conf.sample_on_train_start = False

            if self.stegan_type == SteganType.images:
                semantic = False
                x_start_concat = torch.concat((cover, hide), dim=1).detach()
                cond = None
                h_cond = None

            elif self.stegan_type == SteganType.semantics:
                semantic = True
                x_start_concat = torch.randn_like(cover).detach()  # doesn't matter because its ignored if cond are not None
                cond = self.model.encoder.encode(cover)['cond'].detach()
                h_cond = self.model.encoder.encode(hide)['cond'].detach()

            # with numpy seed we have the problem that the sample t's are related!
            t, weight = self.T_sampler.sample(batch_size, device)
            encoder_losses = self.sampler.training_losses(model=self.model.encoder, x_start=x_start_concat, t=t,
                                                          noise=noise,
                                                          model_kwargs={'cover': cover, 'cond': cond, 'h_cond': h_cond})

            encoded_pred_xstart = encoder_losses['pred_xstart']
            t, weight = self.T_sampler.sample(batch_size, device)
            decoder_losses = self.sampler.training_losses(model=self.model.decoder, x_start=encoded_pred_xstart, t=t,
                                                          noise=noise, model_kwargs={'hide': hide})

            loss = self.conf.enc_loss_scale * encoder_losses['loss'].mean() + decoder_losses['loss'].mean()
            if semantic:
                hidden_vector_loss = self.mse_loss(decoder_losses['cond'], h_cond).mean()
                loss += hidden_vector_loss

            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae']:
                if key in encoder_losses and key in decoder_losses:
                    encoder_losses[key] = self.all_gather(encoder_losses[key]).mean()
                    decoder_losses[key] = self.all_gather(decoder_losses[key]).mean()
                    if semantic:
                        hidden_vector_loss = self.all_gather(hidden_vector_loss).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('Total Loss', loss, self.num_samples)
                self.logger.experiment.add_scalar('Encoder Loss', encoder_losses['loss'], self.num_samples)
                self.logger.experiment.add_scalar('Decoder Loss', decoder_losses['loss'], self.num_samples)
                if semantic:
                    self.logger.experiment.add_scalar('Vector MSE Loss', hidden_vector_loss, self.num_samples)
            self.log('train_loss', loss, on_epoch=True)

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # logging
            cover, hide, noise = batch['cover'], batch['hide'], batch['noise']
            self.log_sample(cover=cover, hide=hide, noise=noise, mode='train')
            # TODO: Finish implementing this!
            # self.evaluate_scores()

    def validation_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """

        # logging
        cover, hide, noise = batch['cover'], batch['hide'], batch['noise']
        self.log_sample(cover=cover, hide=hide, noise=noise, mode='eval')
        # TODO: Finish implementing this!
        return

    def on_before_optimizer_step(self, optimizer: Optimizer,
                                 optimizer_idx: int) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [p for group in optimizer.param_groups for p in group['params']]
            # print('before:', grads_norm(iter_opt_params(optimizer)))
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.conf.grad_clip)
            # print('after:', grads_norm(iter_opt_params(optimizer)))

    def log_sample(self, cover, hide, noise, mode='train'):
        """
        put images to the tensorboard
        """
        is_time_to_sample = self.conf.sample_every_samples > 0 and is_time(self.num_samples,
                                                                           self.conf.sample_every_samples,
                                                                           self.conf.batch_size_effective)
        cover = cover[:8]
        hide = hide[:8]
        noise = noise[:8]
        def do(model, postfix, save_real=False):
            model.eval()
            with torch.no_grad():
                l_encoded, l_decoded = [], []
                for c, h, n in zip(cover, hide, noise):
                    with amp.autocast(self.conf.fp16):
                        gen = self.model(cover=c, hide=h, c_noise=n, sampler=self.eval_sampler)
                    l_encoded.append(gen.encoded)
                    l_decoded.append(gen.decoded)


                encoded_imgs = self.all_gather(torch.cat(l_encoded))
                decoded_imgs = self.all_gather(torch.cat(l_decoded))

                if encoded_imgs.dim() == 5:
                    # (n, c, h, w)
                    encoded_imgs = encoded_imgs.flatten(0, 1)

                if decoded_imgs.dim() == 5:
                    # (n, c, h, w)
                    decoded_imgs = decoded_imgs.flatten(0, 1)

                if save_real:
                    # save the original images to the tensorboard
                    real_cover = self.all_gather(cover)
                    real_hide = self.all_gather(hide)

                    if real_cover.dim() == 5:
                        real_cover = real_cover.flatten(0, 1)
                    if real_hide.dim() == 5:
                        real_hide = real_hide.flatten(0, 1)

                    if self.global_rank == 0:
                        encoded_grid = (make_grid(encoded_imgs) + 1) / 2
                        decoded_grid = (make_grid(decoded_imgs) + 1) / 2
                        grid_real_cover = (make_grid(real_cover) + 1) / 2
                        grid_real_hide = (make_grid(real_hide) + 1) / 2

                        self.logger.experiment.add_image(f'sample{postfix}/1. cover', grid_real_cover,
                                                         self.num_samples)
                        self.logger.experiment.add_image(f'sample{postfix}/2. encoded/', encoded_grid, self.num_samples)
                        self.logger.experiment.add_image(f'sample{postfix}/3. hide', grid_real_hide, self.num_samples)
                        self.logger.experiment.add_image(f'sample{postfix}/4. decoded/', decoded_grid, self.num_samples)

                        sample_dir_c = os.path.join(self.conf.logdir, f'sample{postfix}', 'cover')
                        sample_dir_h = os.path.join(self.conf.logdir, f'sample{postfix}', 'hide')
                        sample_dir_e = os.path.join(self.conf.logdir, f'sample{postfix}', 'encoded')
                        sample_dir_d = os.path.join(self.conf.logdir, f'sample{postfix}', 'decoded')

                        if not os.path.exists(sample_dir_c):
                            os.makedirs(sample_dir_c)
                        if not os.path.exists(sample_dir_h):
                            os.makedirs(sample_dir_h)
                        if not os.path.exists(sample_dir_e):
                            os.makedirs(sample_dir_e)
                        if not os.path.exists(sample_dir_d):
                            os.makedirs(sample_dir_d)

                        path_c = os.path.join(sample_dir_c, f'{self.num_samples:010}.png')
                        path_h = os.path.join(sample_dir_h, f'{self.num_samples:010}.png')
                        path_e = os.path.join(sample_dir_e, f'{self.num_samples:010}.png')
                        path_d = os.path.join(sample_dir_d, f'{self.num_samples:010}.png')
                        save_image(grid_real_cover, path_c)
                        save_image(grid_real_hide, path_h)
                        save_image(encoded_grid, path_e)
                        save_image(decoded_grid, path_d)

            model.train()

        if mode == 'eval':
            do(self.model, '_eval', save_real=True)
        elif mode == 'train' and is_time_to_sample:
            do(self.model, '_train', save_real=True)
        elif mode == 'log_on_start':
            do(self.model, '_log_on_star', save_real=True)

    def evaluate_scores(self):
        """
        evaluate FID and other scores during training (put to the tensorboard)
        For, FID. It is a fast version with 5k images (gold standard is 50k).
        Don't use its results in the paper!
        """

        def fid(model, postfix):
            score = evaluate_fid(self.eval_sampler, model, self.conf, device=self.device, train_data=self.train_data,
                                 val_data=self.val_data)
            if self.global_rank == 0:
                self.logger.experiment.add_scalar(f'FID{postfix}', score, self.num_samples)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, 'eval.txt'), 'a') as f:
                    metrics = {f'FID{postfix}': score, 'num_samples': self.num_samples}
                    f.write(json.dumps(metrics) + "\n")

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc() and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(self.eval_sampler, model, self.conf, device=self.device, val_data=self.val_data,
                                       latent_sampler=self.eval_latent_sampler)

                if self.global_rank == 0:
                    for key, val in score.items():
                        self.logger.experiment.add_scalar(f'{key}{postfix}', val, self.num_samples)

        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(self.num_samples,
                                                                                 self.conf.eval_every_samples,
                                                                                 self.conf.batch_size_effective):
            print(f'eval fid @ {self.num_samples}')
            lpips(self.model, '')
            fid(self.model, '')

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out['optimizer'] = optim
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=WarmupLR(self.conf.warmup))
            out['lr_scheduler'] = {'scheduler': sched, 'interval': 'step'}
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        for the "eval" mode.
        We first select what to do according to the "conf.eval_programs".
        test_step will only run for "one iteration" (it's a hack!).

        We just want the multi-gpu support.
        """
        # make sure you seed each worker differently!
        self.setup()

        # it will run only one step!
        print('global step:', self.global_step)
        """
        "infer" = predict the latent variables using the encoder on the whole dataset
        """
        if 'infer' in self.conf.eval_programs:
            if 'infer' in self.conf.eval_programs:
                print('infer ...')
                conds = self.infer_whole_dataset().float()
                # NOTE: always use this path for the latent.pkl files
                save_path = f'checkpoints/{self.conf.name}/latent.pkl'
            else:
                raise NotImplementedError()

            if self.global_rank == 0:
                conds_mean = conds.mean(dim=0)
                conds_std = conds.std(dim=0)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(
                    {
                        'conds': conds,
                        'conds_mean': conds_mean,
                        'conds_std': conds_std,
                    }, save_path)
        """
        "infer+render" = predict the latent variables using the encoder on the whole dataset
        THIS ALSO GENERATE CORRESPONDING IMAGES
        """
        # infer + reconstruction quality of the input
        for each in self.conf.eval_programs:
            if each.startswith('infer+render'):
                m = re.match(r'infer\+render([0-9]+)', each)
                if m is not None:
                    T = int(m[1])
                    self.setup()
                    print(f'infer + reconstruction T{T} ...')
                    conds = self.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=
                        f'latent_infer_render{T}/{self.conf.name}.lmdb',
                    )
                    save_path = f'latent_infer_render{T}/{self.conf.name}.pkl'
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    torch.save(
                        {
                            'conds': conds,
                            'conds_mean': conds_mean,
                            'conds_std': conds_std,
                        }, save_path)

        # evals those "fidXX"
        """
        "fid<T>" = unconditional generation (conf.train_mode = diffusion).
            Note:   Diff. autoenc will still receive real images in this mode.
        "fid<T>,<T_latent>" = unconditional generation for latent models (conf.train_mode = latent_diffusion).
            Note:   Diff. autoenc will still NOT receive real images in this made.
                    but you need to make sure that the train_mode is latent_diffusion.
        """
        for each in self.conf.eval_programs:
            if each.startswith('fid'):
                m = re.match(r'fid\(([0-9]+),([0-9]+)\)', each)
                clip_latent_noise = False
                if m is not None:
                    # eval(T1,T2)
                    T = int(m[1])
                    T_latent = int(m[2])
                    print(f'evaluating FID T = {T}... latent T = {T_latent}')
                else:
                    m = re.match(r'fidclip\(([0-9]+),([0-9]+)\)', each)
                    if m is not None:
                        # fidclip(T1,T2)
                        T = int(m[1])
                        T_latent = int(m[2])
                        clip_latent_noise = True
                        print(
                            f'evaluating FID (clip latent noise) T = {T}... latent T = {T_latent}'
                        )
                    else:
                        # evalT
                        _, T = each.split('fid')
                        T = int(T)
                        T_latent = None
                        print(f'evaluating FID T = {T}...')

                self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                if T_latent is not None:
                    latent_sampler = self.conf._make_latent_diffusion_conf(
                        T=T_latent).make_sampler()
                else:
                    latent_sampler = None

                conf = self.conf.clone()
                conf.eval_num_images = 50_000
                score = evaluate_fid(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    latent_sampler=latent_sampler,
                    conds_mean=self.conds_mean,
                    conds_std=self.conds_std,
                    remove_cache=False,
                    clip_latent_noise=clip_latent_noise,
                )
                if T_latent is None:
                    self.log(f'fid_ema_T{T}', score)
                else:
                    name = 'fid'
                    if clip_latent_noise:
                        name += '_clip'
                    name += f'_ema_T{T}_Tlatent{T_latent}'
                    self.log(name, score)
        """
        "recon<T>" = reconstruction & autoencoding (without noise inversion)
        """
        for each in self.conf.eval_programs:
            if each.startswith('recon'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('recon')
                T = int(T)
                print(f'evaluating reconstruction T = {T}...')

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=None)
                for k, v in score.items():
                    self.log(f'{k}_ema_T{T}', v)
        """
        "inv<T>" = reconstruction with noise inversion
        """
        for each in self.conf.eval_programs:
            if each.startswith('inv'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('inv')
                T = int(T)
                print(
                    f'evaluating reconstruction with noise inversion T = {T}...'
                )

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=None,
                                       use_inverted_noise=True)
                for k, v in score.items():
                    self.log(f'{k}_inv_ema_T{T}', v)
