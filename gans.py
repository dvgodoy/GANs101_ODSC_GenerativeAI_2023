import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from torchvision.transforms import ToTensor
from PIL import Image
import random

def set_seed(self, seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        if len(list(layer.children())):
            reset_parameters(layer)

def latent_sample(batch_size, z_size, mode_z='uniform'):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size)*2-1 # make it zero centered
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size)
    return input_z

def show(tensor, ax=None, **kwargs):
    img = np.rollaxis(tensor.detach().cpu().numpy(), 0, 3)
    local_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1} if img.shape[-1] == 1 else {}
    kwargs = {**local_kwargs, **kwargs}
    if ax is None:
        plt.imshow(img.squeeze(), **kwargs)
    else:
        ax.imshow(img.squeeze(), **kwargs)

def set_fontsize(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

def figure1(dataset):
    real = dataset.tensors[0][:10].numpy()
    real = np.rollaxis(real, 1, 4)

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    axs = axs.flatten()
    for i in range(10):
        axs[i].set_title(f'Image #{i}')
        axs[i].imshow(real[i].squeeze(), cmap='gray', vmin=-1, vmax=1)
    fig.tight_layout()
    return fig

def comparison(generator, z_size, loader, device):
    generated = generator(latent_sample(100, z_size, 'normal').to(device)).detach().cpu().numpy()
    if generated.min() < 0: # generator is using tanh
        generated = (generated+1)/2
    generated = np.rollaxis(generated, 1, 4)
    img_gen = np.concatenate([np.concatenate(r, axis=1) for r in np.split(generated, 10, 0)], axis=0)

    n_batches = np.ceil(100/loader.batch_size).astype(int)
    real = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        real.append(x.numpy())
    in_channels = x.size(1)
    real = np.concatenate(real)[:100]
    real = np.rollaxis(real, 1, 4)
    img_real = np.concatenate([np.concatenate(r, axis=1) for r in np.split(real, 10, 0)], axis=0)
    if img_real.min() < 0:
        img_real = (img_real+1)/2

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(img_gen.squeeze(), cmap='gray' if in_channels == 1 else 'Blues')
    axs[0].set_title('Generated')
    axs[0].axis('off')
    axs[1].imshow(img_real.squeeze(), cmap='gray' if in_channels == 1 else 'Blues')
    axs[1].set_title('Real')
    axs[1].axis('off')
    return fig

def preview(loader):
    x, y = next(iter(loader))
    real = x[:10].numpy()
    real = np.rollaxis(real, 1, 4)
    if real.min() < 0:
        real = (real+1)/2

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    axs = axs.flatten()
    for i in range(10):
        axs[i].set_title(f'Image #{i}')
        axs[i].imshow(real[i].squeeze(), vmin=0, vmax=1)
    fig.tight_layout()
    return fig

def plot_losses(g, d, d_real, d_fake, d_penalty=None):
    fig = plt.figure(figsize=(16, 6))

    ax = fig.add_subplot(1, 2, 1)
    plt.plot(g, label='Generator loss')
    plt.plot(d, label='Discriminator loss', color='k')
    plt.legend(fontsize=8)
    ax.set_title('GAN')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Discriminator')
    plt.plot(d_fake, label='Discriminator loss (Fake)', color='r')
    plt.plot(d_real, label='Discriminator loss (Real)', color='g')
    if d_penalty is not None:
        plt.plot(d_penalty, label='Gradient Penalty', color='k')
    plt.legend(fontsize=8)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    return fig

def plot_distrib_real_vs_fake(yhat_real, yhat_fake, is_critic=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(yhat_real.squeeze().tolist(), bins=np.linspace(0, 1, 11) if not is_critic else 11, label='Real', alpha=0.6, color='g')
    ax.hist(yhat_fake.squeeze().tolist(), bins=np.linspace(0, 1, 11) if not is_critic else 11, label='Fake', alpha=0.6, color='r')
    ax.legend()
    if is_critic:
        ax.set_xlabel('Score')
        ax.set_title('Critic')
    else:
        ax.set_xlabel('Probability')
        ax.set_title('Discriminator')
    return fig

def make_f(slope, intercept):
    def f(x):
        return slope * x + intercept
    return f

def lipschitz(x, f, x0, title, k=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        fig = ax.get_figure()

    fa = f
    ax.plot(x + x0, fa(x + x0), c='k')
    ax.plot([x0, x0], [min(fa(x[0]+x0), k*x[0]+fa(x0)), fa(x0)], linestyle='--', c='r', alpha=0.7)
    ax.plot([x[0]+x0, x0], [fa(x0), fa(x0)], linestyle='--', c='r', alpha=0.7)
    ax.scatter([x0], [fa(x0)], c='k')
    ax.plot(x + x0, k*x + fa(x0), c='gray', linestyle='--', linewidth=2)
    ax.plot(x + x0, -k*x + fa(x0), c='gray', linestyle='--', linewidth=2)
    ax.fill_between(x + x0, k*x + fa(x0), -k*x + fa(x0), alpha=0.3)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title(title)
    fig.tight_layout()
    return fig

## compare to gradient clipping using norm
# https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
def add_arrow(line, position=None, direction='right', size=15, color=None, lw=2, alpha=1.0, text=None, text_offset=(0 , 0)):
    """
    add an arrow to a line.
    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle='--' if alpha < 1 else '-', alpha=alpha),
        size=size,
    )
    if text is not None:
        line.axes.annotate(text, color=color,
            xytext=(xdata[end_ind] + text_offset[0], ydata[end_ind] + text_offset[1]),
            xy=(xdata[end_ind], ydata[end_ind]),
            size=size,
        )

def make_line(ax, point, color='k', label=None):
    point = np.vstack([[0., 0.], np.array(point.squeeze().tolist())])
    line = ax.plot(*point.T, lw=1, color=color, label=label)[0]
    return line

def grad_norms(ks, colors=None, result=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.get_figure()
    
    line_k = []
    norm_k = []
    cos_k = []
    if colors is None:
        colors = ['gray'] * len(ks)
    for i, k in enumerate(ks):
        line_k.append(make_line(ax, k, color=colors[i], label='Data Points'))
        norm_k.append(np.linalg.norm(k))
        cos_k.append(np.dot(np.array([1, 0]), k)/(norm_k[-1]))

    for i in range(len(ks)):
        add_arrow(line_k[i], lw=2, color=colors[i], text='', size=12, text_offset=(-.33, .1))
        angle = np.sign(ks[i][1])*np.arccos(np.dot(np.array([1, 0]), ks[i])/(np.linalg.norm(ks[i])))/np.pi*180
        square = plt.Rectangle(ks[i]/max(norm_k[i], 1.01), abs(1-norm_k[i]/1.01), abs(1-norm_k[i]/1.01), angle=angle, fill=False, lw=2, color='r', label='Squared Diff')
        ax.add_artist(square)
    circle1 = plt.Circle((0, 0), 1., color='k', fill=False, lw=2)
    ax.add_artist(circle1)

    ax.set_ylim([-1.02, 1.02])
    ax.set_xlim([-1.02, 1.02])

    ax.set_xticks([-2.0, 0, 2.0])
    ax.set_xticklabels([-2.0, 0, 2.0], fontsize=12)
    ax.set_yticks([-2.0, 0, 2.0])
    ax.set_yticklabels([-2.0, 0, 2.0], fontsize=12)
    ax.set_xlabel(r'$grad_0$', fontsize=14)
    ax.set_ylabel(r'$grad_1$', fontsize=14)
    ax.set_title('Gradient Norms')
    fig.tight_layout()
    plt.legend(handles=[square, line_k[0]])
    return fig

def plot_interpolated(z_size, gan, loader, n_samples=5, seed=13):
    set_seed(seed)
    device = next(gan.parameters()).device.type
    fake_samples = gan.generator(latent_sample(n_samples, z_size, 'uniform').to(device)).detach().cpu()
    real_samples = next(iter(loader))[0][:n_samples]
    alpha = torch.rand(n_samples, 1, 1, 1, requires_grad=True)
    
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    prob_interpolated = gan.discriminator(interpolated.to(device))
    
    fig, axs = plt.subplots(3, n_samples, figsize=(n_samples*3, 10))
    rows = [real_samples, interpolated.detach(), fake_samples]
    for row in range(3):
        for col in range(n_samples):
            if col == 0:
                axs[row, col].set_ylabel(['Real', 'Interpolated', 'Fake'][row], fontsize=15)
            if row == 0:
                axs[row, col].set_title(f'Alpha: {alpha[col].squeeze().detach().numpy():.2f}')
            elif row == 1:
                axs[row, col].set_title(f'Critic Score: {prob_interpolated[col].squeeze().detach().cpu().numpy():.2f}')
            img = np.rollaxis(rows[row][col].numpy(), 0, 3)
            axs[row, col].imshow(img.squeeze(), cmap='gray', vmin=-1, vmax=1)
    fig.tight_layout()
    return fig

def pixel_gradients(gradients):
    n = gradients.size(0)
    grad_norm = gradients.view(5, -1).norm(2, dim=1)
    
    fig, axs = plt.subplots(1, n, figsize=(n*3, 3.3))
    axs = np.atleast_2d(axs)
    rows = [gradients.detach()]
    for row in range(1):
        for col in range(n):
            if col == 0:
                axs[row, col].set_ylabel(['Gradients'][row])
            img = np.rollaxis(rows[row][col].numpy(), 0, 3)
            axs[row, col].set_title(f'Gradient Norm: {grad_norm[col].detach().cpu().numpy():.2f}')
            im = axs[row, col].imshow(img.squeeze(), cmap='gray', vmin=gradients.min(), vmax=gradients.max())
            fig.colorbar(im, ax=axs[row, col])
    fig.tight_layout()
    return fig

def plot_evolution(epoch_samples, selected_epochs):
    has_tanh = np.array(epoch_samples).min() < 0
    fig = plt.figure(figsize=(10, 2*len(selected_epochs)))
    for i, epoch in enumerate(selected_epochs):
        for j in range(5):
            ax = fig.add_subplot(len(selected_epochs), 5, i*5+j+1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f'Epoch {epoch}', fontsize=16)

            image = epoch_samples[epoch-1][j]
            if has_tanh:
                image = (image+1)/2
            if image.shape[0] == 1:
                ax.imshow(image.squeeze(), cmap='gray')
            else:
                ax.imshow(np.rollaxis(image, 0, 3))
    return fig
