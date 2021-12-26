import torch
from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, lr_scheduler,
                 config, trainloader, validloader=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, lr_scheduler,
                 config, trainloader, validloader, len_epoch)

    def _train_step(self, batch):
        """
        Training logic for a step

        :param batch: batch of current step
        :return: 
            loss: torch Variable with map for backwarding
            mets: metrics computed between output and target, dict
        """
        utts, data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        mets = {met.__name__ : met(output, target) for met in self.metric_ftns}
        return loss, mets


    def _valid_step(self, batch):
        """
        Valid logic for a step

        :param batch: batch of current step
        :return:
            loss: torch Variable without map
            mets: metrics computed between output and target, dict
        """
        utts, data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        with torch.no_grad():
            output = self.model(data)
        loss = self.criterion(output, target)
        mets = {met.__name__ : met(output, target) for met in self.metric_ftns}
        return loss, mets

