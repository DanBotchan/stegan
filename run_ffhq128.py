from templates import stegan_config
from experiment import train

if __name__ == '__main__':
    # train the autoenc moodel
    # this requires V100s.
    gpus = [0, 1, 2, 3]
    conf = stegan_config()
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

