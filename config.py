from types import SimpleNamespace

def get_config():
    config = SimpleNamespace()

    # 数据路径
    config.train_data = 'data/train'

    # 超参数
    config.batch_size = 1
    config.lr = 2e-5
    config.epochs = 30
    config.T = 10  # diffusion steps
    config.latent_dim = 128

    return config
