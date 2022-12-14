import numpy as np
import torch


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_dataLoader: torch.utils.data.Dataset,
                 validation_dataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 T: int = 100,
                 schedule: str = "linear",
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):

        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_dataLoader = training_dataLoader
        self.validation_dataLoader = validation_dataLoader
        self.lr_scheduler = lr_scheduler
        self.T = T
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

        assert schedule in ["linear"]
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps=T)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_dataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_dataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_dataLoader), 'Training', total=len(self.training_dataLoader),
                          leave=True, position=0)

        for i, (x, y) in batch_iter:
            x_0 = x.to(self.device)  # send to device (GPU or CPU)
            t = torch.randint(0, self.T, (self.training_dataLoader.batch_size,)).long().to(self.device)
            x_noisy, noise = forward_diffusion_sample(x_0, t, self.sqrt_alphas_cumprod,
                                                      self.sqrt_one_minus_alphas_cumprod, t.device)
            self.optimizer.zero_grad()  # zerograd the parameters
            noise_pred = self.model(x_noisy, t)  # one forward pass
            loss = self.criterion(noise, noise_pred)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_dataLoader), 'Validation', total=len(self.validation_dataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            x_0 = x.to(self.device)  # send to device (GPU or CPU)
            t = torch.randint(0, self.T, (self.validation_dataLoader.batch_size,)).long().to(self.device)
            x_noisy, noise = forward_diffusion_sample(x_0, t, self.sqrt_alphas_cumprod,
                                                      self.sqrt_one_minus_alphas_cumprod, t.device)

            with torch.no_grad():
                noise_pred = self.model(x_noisy, t)  # one forward pass
                loss = self.criterion(noise, noise_pred)  # calculate loss
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()


def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    index = t.cpu()
    out = vals.gather(-1, index)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)
