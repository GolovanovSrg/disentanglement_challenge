import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .optim import AdamW
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
    def __init__(self, encoder, autoregressor, predictor, optimizer_params={}, devices=[0], n_jobs=0):
        assert torch.cuda.is_available()

        lr = optimizer_params.get('lr', 1e-3)
        weight_decay = optimizer_params.get('weight_decay', 0)
        warmap = optimizer_params.get('warmap', 100)
        amsgrad = optimizer_params.get('amsgrad', False)

        self.device = torch.device('cuda:' + str(devices[0]))
        self.encoder = nn.DataParallel(encoder, device_ids=devices).to(self.device)
        self.autoregressor = nn.DataParallel(autoregressor, device_ids=devices).to(self.device)
        self.predictor = nn.DataParallel(predictor, device_ids=devices).to(self.device)
        self.criterion = CPCLoss().to(self.device)

        param_optimizer = list(self.encoder.named_parameters()) + list(self.autoregressor.named_parameters()) + list(self.predictor.named_parameters())
        no_decay = ['bias']
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)

        def scheduler_func(iteration):
            if iteration <= warmap:
                return iteration / warmap
            return 1

        self.scheduler = LambdaLR(self.optimizer, scheduler_func)

        self.last_epoch = 0
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
            embeddings = embeddings.view(*embeddings.shape[:2], -1)
            for n, pred_item in enumerate(predictions, 1):
                pred_item = pred_item.view(*pred_item.shape[:2], -1)
                batch_loss += self.criterion(pred_item[:, :, :-n], embeddings[:, :, n:])

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.scheduler.step()
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
                embeddings = embeddings.view(*embeddings.shape[:2], -1)
                for n, pred_item in enumerate(predictions, 1):
                    pred_item = pred_item.view(*pred_item.shape[:2], -1)
                    batch_loss += self.criterion(pred_item[:, :, :-n], embeddings[:, :, n:])

                loss.update(batch_loss.item())
                tqdm_test_dataloader.set_postfix({'loss': loss()})

            return loss()

    def _save_checkpoint(self, checkpoint_path):  
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = {'encoder': self.encoder.module.state_dict(),
                      'autoregressor': self.autoregressor.module.state_dict(),
                      'predictor': self.predictor.module.state_dict()}
        torch.save(checkpoint, checkpoint_path)

    def train(self, train_data, n_epochs, batch_size, test_data=None, checkpoint_dir=None,
              last_checkpoint_path=None, best_checkpoint_path=None):
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=self.n_jobs)

        if test_data is not None:
            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=self.n_jobs)

        best_loss = float("inf")
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            self._train_epoch(train_dataloader)

            if last_checkpoint_path is not None:
                self._save_checkpoint(last_checkpoint_path)

            if test_data is not None:
                torch.cuda.empty_cache()
                loss = self._test_epoch(test_dataloader)

                if best_checkpoint_path is not None:
                    if loss < best_loss:
                        best_loss = loss
                        self._save_checkpoint(best_checkpoint_path)

            self.last_epoch += 1
