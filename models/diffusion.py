import torch
import torch.nn as nn

def linear_degradation(x0, xT, alpha_t):
    """
    执行退化操作: xt = (1 - alpha_t) * x0 + alpha_t * xT
    """
    return (1 - alpha_t) * x0 + alpha_t * xT


def compute_alpha_schedule(T):
    """
    返回 alpha_t 的线性调度（递增）
    """
    return torch.linspace(0, 1, steps=T + 1)  # T+1个数，从t=0到T


class DiffusionSampler(nn.Module):
    def __init__(self, generator, TLA=None, T=10):
        super().__init__()
        self.generator = generator  # R_theta
        self.TLA = TLA
        self.T = T
        self.alpha_schedule = compute_alpha_schedule(T)

    def restore_step(self, x_t, x_T, t, time_embedding_fn):
        """
        单步恢复: 预测 x0_hat, 计算 xt-1
        """
        B = x_t.size(0)
        t_tensor = torch.full((B,), t, dtype=torch.long, device=x_t.device)

        Lt = time_embedding_fn(t_tensor)
        x0_hat = self.generator(x_t, Lt)

        alpha_t = self.alpha_schedule[t].to(x_t.device)
        xT_approx = (x_t - alpha_t * x0_hat) / (1 - alpha_t + 1e-6)

        # 再次退化
        alpha_t_prev = self.alpha_schedule[t - 1].to(x_t.device)
        x_t_minus_1 = (1 - alpha_t_prev) * x0_hat + alpha_t_prev * xT_approx

        return x_t_minus_1, x0_hat

    def sample(self, x_T, time_embedding_fn):
        """
        完整采样流程: 从 x_T 迭代至 x0_hat
        """
        x_t = x_T
        for t in reversed(range(1, self.T + 1)):
            x_t, _ = self.restore_step(x_t, x_T, t, time_embedding_fn)
        return x_t

    def forward_train(self, x0, xT, t, time_embedding_fn):
        """
        用于训练阶段：输入真实 x0 和 xT，在指定时间步t生成 xt 和 x0_hat
        """
        alpha_t = self.alpha_schedule[t].to(x0.device)
        x_t = linear_degradation(x0, xT, alpha_t)
        t_tensor = torch.full((x0.size(0),), t, dtype=torch.long, device=x0.device)
        Lt = time_embedding_fn(t_tensor)
        x0_hat = self.generator(x_t, Lt)
        return x_t, x0_hat
