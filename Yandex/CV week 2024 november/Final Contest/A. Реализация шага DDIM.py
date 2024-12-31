import torch

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def ddim_solver_step(model_output, x_t, t, s, scheduler):
    """
    Реализация шага DDIM солвера для VP процесса зашумления и eps-prediction модели.

    Params:
        model_output: torch.Tensor[B, 4, 64, 64] - предсказание модели - шум ε
        x_t: torch.Tensor[B, 4, 64, 64] - сэмплы на шаге t
        t: torch.Tensor[B] - номер текущего шага
        s: torch.Tensor[B] - номер следующего шага
        scheduler: DDIMScheduler - расписание диффузионного процесса, чтобы получить alpha и sigma
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(x_t.device)
    alphas = torch.sqrt(alphas_cumprod)  # α_t = sqrt(ᾱ_t)
    sigmas = torch.sqrt(1.0 - alphas_cumprod)  # σ_t = sqrt(1 - ᾱ_t)

    sigmas_s = extract_into_tensor(sigmas, s, x_t.shape)
    alphas_s = extract_into_tensor(alphas, s, x_t.shape)

    sigmas_t = extract_into_tensor(sigmas, t, x_t.shape)
    alphas_t = extract_into_tensor(alphas, t, x_t.shape)

    alphas_s[s == 0] = 1.0
    sigmas_s[s == 0] = 0.0

    alphas_t[t == 0] = 1.0
    sigmas_t[t == 0] = 0.0

    x_0 = (x_t - sigmas_t * model_output) / alphas_t

    x_s = alphas_s * x_0 + sigmas_s * model_output

    return x_s
