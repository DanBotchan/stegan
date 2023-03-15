import os
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *

from lit_model import LitModel
from config import TrainConfig


def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
    print('conf:', conf.name)
    # assert not (conf.fp16 and conf.grad_clip > 0
    #             ), 'pytorch lightning has bug with amp + gradient clipping'
    model = LitModel(conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)

    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}', save_last=True, save_top_k=1,
                                 every_n_train_steps=conf.save_every_samples // conf.batch_size_effective)
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    print('ckpt path:', checkpoint_path)

    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print('resume!')
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir, name=None, version='')

    # from pytorch_lightning.
    strategy = None
    if len(gpus) == 1 and nodes == 1:
        accelerator = None
    elif len(gpus) > 1:
        accelerator = 'cuda'
        strategy = 'ddp_find_unused_parameters_false'
    else:
        accelerator = 'cpu'

    trainer = pl.Trainer(max_steps=(conf.total_samples // conf.batch_size_effective) * 4, resume_from_checkpoint=resume,
                         gpus=gpus, num_nodes=nodes, accelerator=accelerator, precision=16 if conf.fp16 else 32,
                         callbacks=[checkpoint, LearningRateMonitor()],
                         # clip in the model instead
                         # gradient_clip_val=conf.grad_clip,
                         replace_sampler_ddp=True, num_sanity_val_steps=0,
                         logger=tb_logger, accumulate_grad_batches=conf.accum_batches, strategy=strategy,
                         )

    if mode == 'train':
        trainer.fit(model)

    # elif mode == 'eval':
    #     # load the latest checkpoint
    #     # perform lpips
    #     # dummy loader to allow calling "test_step"
    #     dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
    #                        batch_size=conf.batch_size)
    #     eval_path = conf.eval_path or checkpoint_path
    #     # conf.eval_num_images = 50
    #     print('loading from:', eval_path)
    #     state = torch.load(eval_path, map_location='cpu')
    #     print('step:', state['global_step'])
    #     model.load_state_dict(state['state_dict'])
    #     # trainer.fit(model)
    #     out = trainer.test(model, dataloaders=dummy)
    #     # first (and only) loader
    #     out = out[0]
    #     print(out)
    #
    #     if get_rank() == 0:
    #         # save to tensorboard
    #         for k, v in out.items():
    #             tb_logger.experiment.add_scalar(
    #                 k, v, state['global_step'] * conf.batch_size_effective)
    #
    #         # # save to file
    #         # # make it a dict of list
    #         # for k, v in out.items():
    #         #     out[k] = [v]
    #         tgt = f'evals/{conf.name}.txt'
    #         dirname = os.path.dirname(tgt)
    #         if not os.path.exists(dirname):
    #             os.makedirs(dirname)
    #         with open(tgt, 'a') as f:
    #             f.write(json.dumps(out) + "\n")
    #         # pd.DataFrame(out).to_csv(tgt)

    else:
        raise NotImplementedError()
