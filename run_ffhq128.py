from templates import stegan_config
from experiment import train

if __name__ == '__main__':

    debug = True
    if debug:
        gpus = []
    else:
        gpus = [0, 1, 2, 3]
    scale_up_gpus = len(gpus) if len(gpus) > 0 else 1

    conf = stegan_config(debug=debug, scale_up_gpus=scale_up_gpus)
    conf.name = 'experiment_date'

    train(conf, gpus=gpus)

    #
    # # infer the latents for training the latent DPM
    # # NOTE: not gpu heavy, but more gpus can be of use!
    # gpus = [0, 1, 2, 3]
    # conf.eval_programs = ['infer']
    # train(conf, gpus=gpus, mode='eval')

