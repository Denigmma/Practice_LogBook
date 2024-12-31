import torch


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def q_sample(x, t, scheduler, noise=None):
    """
    Процесс зашумления: x_t = alpha_t * x0 + sigma_t * epsilon
    где epsilon ~ N(0, I)

    params:
        x: torch.Tensor[B, C, H, W] - чистые данные x0
        t: torch.Tensor[B] - номера текущих шагов
        scheduler: DDIMScheduler - расписание диффузионного процесса
        noise: torch.Tensor[B, C, H, W] - шум epsilon (опционально)

    return:
        x_t: torch.Tensor[B, C, H, W] - зашумленные данные на шаге t
    """
    # Извлекаем накопленные alphas_cumprod и вычисляем alpha и sigma
    alphas_cumprod = scheduler.alphas_cumprod.to(x.device)
    alphas = torch.sqrt(alphas_cumprod)  # α_t = sqrt(ᾱ_t)
    sigmas = torch.sqrt(1.0 - alphas_cumprod)  # σ_t = sqrt(1 - ᾱ_t)

    # Извлекаем α_t и σ_t для текущих шагов t
    sigmas_t = extract_into_tensor(sigmas, t, x.shape)
    alphas_t = extract_into_tensor(alphas, t, x.shape)

    # Если шум не задан, генерируем его
    if noise is None:
        noise = torch.randn_like(x)

    # Вычисляем x_t по формуле
    x_t = alphas_t * x + sigmas_t * noise
    return x_t
