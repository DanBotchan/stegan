import os
import shutil

import torch
import torchvision
from pytorch_fid import fid_score
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm, trange

from renderer import *
from config import *
from diffusion import Sampler
from dist_utils import *
import lpips
from ssim import ssim


def make_subset_loader(conf: TrainConfig, dataset: Dataset, batch_size: int, shuffle: bool, parallel: bool,
                       drop_last=True, debug=False):
    dataset = SubsetDataset(dataset, size=conf.eval_num_images)
    if parallel and distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      # with sampler, use the sample instead of this option
                      shuffle=False if sampler else shuffle,
                      num_workers=conf.num_workers, pin_memory=True,
                      drop_last=drop_last, multiprocessing_context=get_context('fork') if not debug else None)


def evaluate_lpips(sampler: Sampler, model, conf: TrainConfig, device, val_data: Dataset,
                   latent_sampler: Sampler = None, use_inverted_noise: bool = False):
    """
    compare the generated images from autoencoder on validation dataset

    Args:
        use_inversed_noise: the noise is also inverted from DDIM
    """
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    val_loader = make_subset_loader(conf, dataset=val_data, batch_size=conf.batch_size_eval, shuffle=False,
                                    parallel=True)

    model.eval()
    with torch.no_grad():
        scores = {'lpips': [], 'mse': [], 'ssim': [], 'psnr': []}
        for batch in tqdm(val_loader, desc='lpips'):
            imgs = batch['img'].to(device)

            if use_inverted_noise:
                # inverse the noise
                # with condition from the encoder
                model_kwargs = {}
                if conf.model_type.has_autoenc():
                    with torch.no_grad():
                        model_kwargs = model.encode(imgs)
                x_T = sampler.ddim_reverse_sample_loop(
                    model=model,
                    x=imgs,
                    clip_denoised=True,
                    model_kwargs=model_kwargs)
                x_T = x_T['sample']
            else:
                x_T = torch.randn((len(imgs), 3, conf.img_size, conf.img_size),
                                  device=device)

            if conf.model_type == ModelType.ddpm:
                # the case where you want to calculate the inversion capability of the DDIM model
                assert use_inverted_noise
                pred_imgs = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                )
            else:
                pred_imgs = render_condition(conf=conf,
                                             model=model,
                                             x_T=x_T,
                                             x_start=imgs,
                                             cond=None,
                                             sampler=sampler)
            # # returns {'cond', 'cond2'}
            # conds = model.encode(imgs)
            # pred_imgs = sampler.sample(model=model,
            #                            noise=x_T,
            #                            model_kwargs=conds)

            # (n, 1, 1, 1) => (n, )
            scores['lpips'].append(lpips_fn.forward(imgs, pred_imgs).view(-1))

            # need to normalize into [0, 1]
            norm_imgs = (imgs + 1) / 2
            norm_pred_imgs = (pred_imgs + 1) / 2
            # (n, )
            scores['ssim'].append(
                ssim(norm_imgs, norm_pred_imgs, size_average=False))
            # (n, )
            scores['mse'].append(
                (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3]))
            # (n, )
            scores['psnr'].append(psnr(norm_imgs, norm_pred_imgs))
        # (N, )
        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float()
    model.train()

    barrier()

    # support multi-gpu
    outs = {
        key: [
            torch.zeros(len(scores[key]), device=device)
            for i in range(get_world_size())
        ]
        for key in scores.keys()
    }
    for key in scores.keys():
        all_gather(outs[key], scores[key])

    # final scores
    for key in scores.keys():
        scores[key] = torch.cat(outs[key]).mean().item()

    # {'lpips', 'mse', 'ssim'}
    return scores


def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.
    # (n,)
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


def evaluate_fid(sampler: Sampler, model, conf: TrainConfig, device, train_data: Dataset, val_data: Dataset,
                 remove_cache: bool = True):
    assert conf.fid_cache is not None
    if get_rank() == 0:
        # no parallel
        # validation data for a comparing FID
        val_loader = make_subset_loader(conf, dataset=val_data, batch_size=conf.batch_size_eval, shuffle=False,
                                        parallel=False, debug=conf.debug)

        # put the val images to a directory
        cache_dir = f'{conf.fid_cache}_{conf.eval_num_images}'
        if (os.path.exists(cache_dir) and len(os.listdir(cache_dir)) < conf.eval_num_images):
            shutil.rmtree(cache_dir)

        if not os.path.exists(cache_dir):
            # write files to the cache
            # the images are normalized, hence need to denormalize first
            loader_to_path(val_loader, cache_dir, denormalize=True)

        # create the generate dir
        if os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)
        os.makedirs(conf.generate_dir)

    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size_eval, rank, world_size)

    def filename(idx):
        return world_size * idx + rank

    model.eval()
    with torch.no_grad():
        # evaulate autoencoder (given the images)
        # to make the FID fair, autoencoder must not see the validation dataset
        # also shuffle to make it closer to unconditional generation
        train_loader = make_subset_loader(conf, dataset=train_data, batch_size=batch_size, shuffle=True, parallel=True,
                                          debug=conf.debug)

        i = 0
        for batch in tqdm(train_loader, desc='generating images'):
            cover, hide, noise = batch['cover'], batch['hide'], batch['noise']
            x_start_concat = torch.concat((cover, hide), dim=1).detach()
            batch_encoded_images = render_condition(model=model.encoder, x_T=noise, x_start=x_start_concat,
                                                    sampler=sampler).cpu()
            batch_decoded_images = render_condition(model=model.decoder, x_T=noise, x_start=batch_encoded_images,
                                                    sampler=sampler).cpu()

            # denormalize the images
            batch_encoded_images = (batch_encoded_images + 1) / 2
            batch_decoded_images = (batch_decoded_images + 1) / 2
            # keep the generated images
            for j in range(len(batch_encoded_images)):
                img_name = filename(i + j)
                torchvision.utils.save_image(batch_encoded_images[j],
                                             os.path.join(conf.generate_dir, f'encoded_{img_name}.png'))
                torchvision.utils.save_image(batch_decoded_images[j],
                                             os.path.join(conf.generate_dir, f'decoded_{img_name}.png'))
            i += len(batch_encoded_images)
    model.train()

    barrier()

    if get_rank() == 0:
        fid = fid_score.calculate_fid_given_paths([cache_dir, conf.generate_dir], batch_size, device=device, dims=2048)
        # remove the cache
        if remove_cache and os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)

    barrier()
    if get_rank() == 0:
        # need to float it! unless the broadcasted value is wrong
        fid = torch.tensor(float(fid), device=device)
        broadcast(fid, 0)
    else:
        fid = torch.tensor(0., device=device)
        broadcast(fid, 0)
    fid = fid.item()
    print(f'fid ({get_rank()}):', fid)
    return fid


def loader_to_path(loader: DataLoader, path: str, denormalize: bool):
    # not process safe!
    if not os.path.exists(path):
        os.makedirs(path)
    # write the loader to files
    i = 0
    for batch in tqdm(loader, desc='copy images'):
        cover, hide = batch['cover'], batch['hide']
        if denormalize:
            cover = (cover + 1) / 2
            hide = (hide + 1) / 2
        for j in range(len(cover)):
            torchvision.utils.save_image(cover[j], os.path.join(path, f'cover_{i + j}.png'))
            torchvision.utils.save_image(hide[j], os.path.join(path, f'hide_{i + j}.png'))
        i += len(hide)
