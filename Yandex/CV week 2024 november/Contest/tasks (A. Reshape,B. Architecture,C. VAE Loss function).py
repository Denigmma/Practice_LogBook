# some prelimenaries
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import numpy as np
import matplotlib.pylab as plt

torch.set_num_threads(16)
torch.manual_seed(0)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print('Using torch version {}'.format(torch.__version__))
print('Using {} device'.format(device))


def transpose_from_scratch(lst):
    """
    Меняет местами две оси (размерности) вложенного списка и возвращает результат в виде torch.Tensor.

    Parameters:
    lst: Входной многомерный вложенный список или torch.Tensor
    Returns:
    torch.Tensor: Новый тензор с переставленными осями
    """
    if isinstance(lst, torch.Tensor):
        lst = lst.tolist()
    transposed = [[lst[j][i] for j in range(len(lst))] for i in range(len(lst[0]))]
    return torch.tensor(transposed)

# Преобразование данных и загрузчики
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразуем в тензор
    lambda x: transpose_from_scratch(x[0])  # Применяем функцию
])

train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


d, nh, D = 32, 200, 28 * 28

# Архитектура VAE
enc = nn.Sequential(
    nn.Linear(D, nh),
    nn.ReLU(),
    nn.Linear(nh, nh),
    nn.ReLU(),
    nn.Linear(nh, 2 * d)  # Два параметра: среднее и логарифм дисперсии
)

dec = nn.Sequential(
    nn.Linear(d, nh),
    nn.ReLU(),
    nn.Linear(nh, nh),
    nn.ReLU(),
    nn.Linear(nh, D)  # Выход размерности D
)

from torch.distributions import Normal, Bernoulli, Independent, constraints
Bernoulli.support = constraints.interval(0, 1)


def loss_vae(x, encoder, decoder):
    """
    returns
    1. the avergave value of negative ELBO across the minibatch x
    2. and the output of the decoder
    """
    batch_size = x.size(0)
    # Этап энкодера: получение параметров распределения q(z|x)
    encoder_output = encoder(x)
    # Определение распределения q(z|x), где nu(x) и sigma(x) - это выходы энкодера
    qz_x = Independent(Normal(loc=encoder_output[:, :d],
                              scale=torch.exp(encoder_output[:, d:])),
                       reinterpreted_batch_ndims=1)
    # Сэмплирование из q(z|x) с использованием rsample для репараметризации
    z = qz_x.rsample()
    # Этап декодера: получение логитов p(x|z)
    decoder_output = decoder(z)
    # Определение распределения p(x|z) с использованием логитов
    px_z = Independent(Bernoulli(logits=decoder_output),
                       reinterpreted_batch_ndims=1)
    # Определение распределения p(z) для скрытых переменных
    pz = Independent(Normal(loc=torch.zeros(batch_size, d, device=device),
                            scale=torch.ones(batch_size, d, device=device)),
                     reinterpreted_batch_ndims=1)
    # Потери: вычисление отрицательной нижней границы ELBO
    loss = -(px_z.log_prob(x) + pz.log_prob(z) - qz_x.log_prob(z)).mean()
    return loss, decoder_output


# Пример данных: Перемещаем данные на устройство
x_sample = test_loader.dataset.data[0].float().unsqueeze(0).to(device) / 255.0
x_sample = x_sample.view(-1, D).to(device)  # D должен быть размером 28*28 для изображений 28x28

# Отправляем модели на нужное устройство
enc = enc.to(device)
dec = dec.to(device)

# Вычисление потерь без градиентов
with torch.no_grad():
    loss_value, _ = loss_vae(x_sample, enc, dec)

print(f"Loss: {loss_value.item()}")


from itertools import chain

def train_model(loss, model, batch_size=100, num_epochs=3, learning_rate=1e-3):
    gd = torch.optim.Adam(
        chain(*[x.parameters() for x in model
                if (isinstance(x, nn.Module) or isinstance(x, nn.Parameter))]),
        lr=learning_rate)
    train_losses = []
    test_results = []
    for _ in range(num_epochs):
        for i, (batch, _) in enumerate(train_loader):
            total = len(train_loader)
            gd.zero_grad()
            batch = batch.view(-1, D).to(device)
            loss_value, _ = loss(batch, *model)
            loss_value.backward()
            train_losses.append(loss_value.item())
            if (i + 1) % 10 == 0:
                print('\rTrain loss:', train_losses[-1],
                      'Batch', i + 1, 'of', total, ' ' * 10, end='', flush=True)
            gd.step()
        test_loss = 0.
        for i, (batch, _) in enumerate(test_loader):
            batch = batch.view(-1, D).to(device)
            batch_loss, _ = loss(batch, *model)
            test_loss += (batch_loss - test_loss) / (i + 1)
        print('\nTest loss after an epoch: {}'.format(test_loss))


train_model(loss_vae, model=[enc, dec], num_epochs=1)


###plot

def sample_vae(dec, n_samples=50):
    with torch.no_grad():
        samples = torch.sigmoid(dec(torch.randn(n_samples, d).to(device)))
        samples = samples.view(n_samples, 28, 28).cpu().numpy()
    return samples

def plot_samples(samples, h=5, w=10):
    fig, axes = plt.subplots(nrows=h,
                             ncols=w,
                             figsize=(int(1.4 * w), int(1.4 * h)),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i], cmap='gray')
plot_samples(sample_vae(dec=dec))
def plot_reconstructions(loss, model):
    with torch.no_grad():
        batch = (test_loader.dataset.data[:25].float() / 255.)
        batch = batch.view(-1, D).to(device)
        _, rec = loss(batch, *model)
        rec = torch.sigmoid(rec)
        rec = rec.view(-1, 28, 28).cpu().numpy()
        batch = batch.view(-1, 28, 28).cpu().numpy()

        fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(14, 7),
                                 subplot_kw={'xticks': [], 'yticks': []})
        for i in range(25):
            if i % 5 == 0:
                axes[i % 5, 2 * (i // 5)].set_title("Orig")
                axes[i % 5, 2 * (i // 5) + 1].set_title("Recon")
            axes[i % 5, 2 * (i // 5)].imshow(batch[i], cmap='gray')
            axes[i % 5, 2 * (i // 5) + 1].imshow(rec[i], cmap='gray')
plot_reconstructions(loss_vae, [enc, dec])