import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from datasets.ct_dataset import CTMetalDataset
from models.unet import GeneratorUNet
from models.sie import SIEEncoder
from models.tla import TLAModule
from models.embedding import TimeEmbedding
from models.diffusion import DiffusionSampler

from utils.losses import masked_l1_loss
from config import get_config

def train():
    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集
    train_set = CTMetalDataset(cfg.train_data)
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)

    # 模型
    generator = GeneratorUNet().to(device)
    sie = SIEEncoder().to(device)
    tla = TLAModule(latent_dim=cfg.latent_dim).to(device)
    time_embed = TimeEmbedding(cfg.latent_dim).to(device)
    sampler = DiffusionSampler(generator, TLA=tla, T=cfg.T)

    # 优化器
    optimizer = Adam(list(generator.parameters()) +
                     list(sie.parameters()) +
                     list(tla.parameters()) +
                     list(time_embed.parameters()), lr=cfg.lr)

    for epoch in range(cfg.epochs):
       pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            x0 = batch['x0'].to(device)
            xT = batch['xT'].to(device)
            xLI = batch['xLI'].to(device)
            mask = batch['mask'].to(device)

            B = x0.shape[0]
            t = torch.randint(1, cfg.T, (B,), device=device)

            # Step 1: Stage I 预测
            _, x0_hat = sampler.forward_train(x0, xT, t[0].item(), time_embed)

            # Step 2: Stage II + TLA
            alpha_schedule = sampler.alpha_schedule.to(device)
            alpha_t = alpha_schedule[t[0]]
            xT_est = (x0_hat - alpha_t * x0_hat) / (1 - alpha_t + 1e-6)
            alpha_t_minus_1 = alpha_schedule[t[0] - 1]
            x_t_minus_1 = (1 - alpha_t_minus_1) * x0_hat + alpha_t_minus_1 * xT_est

            Lt_minus_1 = time_embed(t - 1)
            Lt_adjusted = tla(Lt_minus_1, x_t_minus_1, x0_hat, xT)

            x0_refined = generator(x_t_minus_1, Lt_adjusted, sie(xLI)[0])

            # 损失计算
            loss1 = masked_l1_loss(x0_hat, x0, mask)
            loss2 = masked_l1_loss(x0_refined, x0, mask)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        # 保存模型
        torch.save(generator.state_dict(), f'checkpoints/generator_epoch{epoch+1}.pt')


if __name__ == '__main__':
    train()
