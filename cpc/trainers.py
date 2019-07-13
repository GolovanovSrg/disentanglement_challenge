import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import CPCLoss


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, value):
        self._sum += value
        self._count += 1

    def __call__(self):
        if self._count:
            return self._sum / self._count
        return 0


class Trainer:
    def __init__(self, encoder, autoregressor, predictor, optimizer_params={}, device=None, n_jobs=0):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        lr = optimizer_params.get('lr', 1e-3)
        weight_decay = optimizer_params.get('weight_decay', 0)
        amsgrad = optimizer_params.get('amsgrad', False)

        self.encoder = encoder.to(device)
        self.autoregressor = autoregressor.to(device)
        self.predictor = predictor.to(device)
        self.criterion = CPCLoss().to(device)

        param_optimizer = list(self.encoder.named_parameters()) + list(self.autoregressor.named_parameters()) + list(self.predictor.named_parameters())
        no_decay = ['bias']
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        self.optimizer = Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)

        self.last_epoch = 0
        self.device = device
        self.n_jobs = n_jobs

    def _train_epoch(self, train_dataloader):
        tqdm_train_dataloader = tqdm(train_dataloader, desc=f'Train, epoch #{self.last_epoch}')
        self.encoder.train()
        self.autoregressor.train()
        self.predictor.train()

        loss = AvgMeter()
        for images in tqdm_train_dataloader:
            images = images.to(self.device)
            embeddings = self.encoder(images)
            contexts = self.autoregressor(embeddings)
            predictions = self.predictor(contexts)

            batch_loss = 0
            for n, pred_item in enumerate(predictions):
                batch_loss += self.criterion(pred_item[:, :, :-n, :], embeddings[:, :, n:, :])

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            loss.update(batch_loss.item())
            tqdm_train_dataloader.set_postfix({'loss': loss()})

    def _test_epoch(self, test_dataloader):
        with torch.no_grad():
            tqdm_test_dataloader = tqdm(test_dataloader, desc=f'Test, epoch #{self.last_epoch}')
            self.encoder.eval()
            self.autoregressor.eval()
            self.predictor.eval()

            loss = AvgMeter()
            for images in tqdm_test_dataloader:
                images = images.to(self.device)
                embeddings = self.encoder(images)
                contexts = self.autoregressor(embeddings)
                predictions = self.predictor(contexts)

                batch_loss = 0
                for n, pred_item in enumerate(predictions):
                    batch_loss += self.criterion(pred_item[:, :, :-n, :], embeddings[:, :, n:, :])

                loss.update(batch_loss.item())
                tqdm_test_dataloader.set_postfix({'loss': loss()})

            return loss()

    def _save_checkpoint(self, checkpoint_dir, checkpoint_name):
        if checkpoint_dir is None:
            return

        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        checkpoint = {'encoder': self.encoder.state_dict(),
                      'autoregressor': self.autoregressor.state_dict(),
                      'predictor': self.predictor.state_dict()}
        torch.save(checkpoint, checkpoint_path)

    def train(self, train_data, n_epochs, batch_size, test_data=None, checkpoint_dir=None, save_last=False, save_best=False):
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=self.n_jobs)

        if test_data is not None:
            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=self.n_jobs)

        best_loss = float("inf")
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            self._train_epoch(train_dataloader)

            if save_last:
                self._save_checkpoint(checkpoint_dir, "last_checkpoint.pt")

            if test_data is not None:
                torch.cuda.empty_cache()
                loss = self._test_epoch(test_dataloader)

                if save_best:
                    if loss < best_loss:
                        best_loss = loss
                        self._save_checkpoint(checkpoint_dir, "best_checkpoint.pt")

            self.last_epoch += 1
