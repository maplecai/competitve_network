import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer():
    """
    Trainer class
    """
    def __init__(self, config, model, criterion, optimizer, device, data_loader, max_epochs, log_step,
                 lr_scheduler=None):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.data_loader = data_loader
        self.max_epochs = max_epochs
        self.log_step = log_step
        self.lr_scheduler = lr_scheduler

        # counters
        self.iter = 0
        self.epoch = 0

        self.losses = []


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.losses = []
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        log = self.losses
        return log


    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.max_epochs):
            result = self._train_epoch(epoch)
            if epoch % self.log_step == 0:
                print('Train Epoch: {} Loss: {:.6f}'.format(epoch, np.mean(result)))

