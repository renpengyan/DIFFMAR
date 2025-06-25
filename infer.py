import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from datasets.ct_dataset import CTMetalDataset
from torch.utils.data import DataLoader

from models.unet import GeneratorUNet
from models.embedding import TimeEmbedding
from models.diffusion import DiffusionSampler
from config import get_config


@torch.no_grad()
def inference():
    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试集（这里只需 xT）
    test_set = CTMetalDataset(cfg.train_data)  # 简化处理用 train_data，正式可设 val/test
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # 加载模型
    generator = GeneratorUNet().to(device)
    time_embed = TimeEmbedding(cfg.latent_dim).to(device)
    sampler = DiffusionSampler(generator, TLA=None, T=cfg.T)  # 推理不使用TLA

    generator.load_state_dict(torch.load('checkpoints/generator_epoch300.pt', map_location=device))
    generator.eval()

    os.makedirs('outputs', exist_ok=True)

    for idx, batch in enumerate(tqdm(loader)):
        xT = batch['xT'].to(device)

        restored = sampler.sample(xT, time_embed)
        save_image(restored, f'outputs/ct_{idx:04d}.png')


if __name__ == '__main__':
    inference()
