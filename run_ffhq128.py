from templates import stegan_config
from experiment import train
from choices import SteganType

if __name__ == '__main__':

    debug = False
    if debug:
        gpus = []
    else:
        gpus = [0, 1]
    scale_up_gpus = len(gpus) if len(gpus) > 0 else 1

    conf = stegan_config(debug=debug, scale_up_gpus=scale_up_gpus)
    conf.name = 'semantics_12_04_23'
    conf.enc_loss_scale = 0.7
    conf.sample_on_train_start = False
    conf.stegan_type = SteganType.deter_decode
    train(conf, gpus=gpus)

    #
    # # infer the latents for training the latent DPM
    # # NOTE: not gpu heavy, but more gpus can be of use!
    # gpus = [0, 1, 2, 3]
    # conf.eval_programs = ['infer']
    # train(conf, gpus=gpus, mode='eval')

