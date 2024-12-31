import torch


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_xs_from_xt_naive(
        x_0, x_t, t, s,  # Не все эти аргументы могут быть вам нужны
        scheduler,
        noise=None,
        **kwargs
):
    """
    Получение точки x_s в CT режиме, т.е., аналитически.

    Params:
        x_0: torch.Tensor[B, C, H, W] - чистые данные x0
        x_t: torch.Tensor[B, C, H, W] - зашумленные данные на шаге t
        t: torch.Tensor[B] - номера текущих шагов
        s: torch.Tensor[B] - номера следующих шагов
        scheduler: DDIMScheduler - расписание диффузионного процесса
        noise: torch.Tensor[B, C, H, W] - шум ε (опционально)

    Returns:
        x_s: torch.Tensor[B, C, H, W] - данные на шаге s
    """
    # Извлекаем alphas_cumprod и вычисляем alpha и sigma
    alphas_cumprod = scheduler.alphas_cumprod.to(x_t.device)
    alphas = torch.sqrt(alphas_cumprod)  # α_t = sqrt(ᾱ_t)
    sigmas = torch.sqrt(1.0 - alphas_cumprod)  # σ_t = sqrt(1 - ᾱ_t)

    # Извлекаем α_s, σ_s и α_t, σ_t для соответствующих шагов
    sigmas_s = extract_into_tensor(sigmas, s, x_t.shape)
    alphas_s = extract_into_tensor(alphas, s, x_t.shape)

    sigmas_t = extract_into_tensor(sigmas, t, x_t.shape)
    alphas_t = extract_into_tensor(alphas, t, x_t.shape)

    # Устанавливаем граничные условия
    alphas_s[s == 0] = 1.0
    sigmas_s[s == 0] = 0.0

    alphas_t[t == 0] = 1.0
    sigmas_t[t == 0] = 0.0

    # Вычисляем ε_theta
    epsilon_theta = (x_t - alphas_t * x_0) / sigmas_t

    # Вычисляем x_s по формуле DDIM
    x_s = alphas_s * x_0 + sigmas_s * epsilon_theta

    return x_s
