import torch
import time
from numpy import inf
from tqdm import tqdm
from abc import abstractmethod
from logger import TensorboardWriter
from utils import inf_loop, MetricTracker
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, lr_scheduler, 
                 config, trainloader, validloader=None, len_epoch=None):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.trainloader = trainloader
        self.validloader = validloader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.trainloader)
        else:
            # iteration-based training
            self.trainloader = inf_loop(trainloader)
            self.len_epoch = len_epoch

        # setup GPU device if available, move model into configured device
        n_gpu_use = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        self.model = model.to(self.device)
        self.model = torch.nn.DataParallel(model)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.log_step = cfg_trainer['log_step']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)


    @abstractmethod
    def _train_step(self, batch):
        """
        Training logic for a step

        :param batch: batch of current step
        :return: 
            loss: torch Variable with map for backwarding
            mets: metrics computed between output and target, dict
        """
        raise NotImplementedError 


    @abstractmethod
    def _valid_step(self, batch):
        """
        Valid logic for a step

        :param batch: batch of current step
        :return:
            loss: torch Variable without map
            mets: metrics computed between output and target, dict
        """
        raise NotImplementedError


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        tic = time.time()
        datatime = batchtime = 0
        batchbar = tqdm(self.trainloader)
        batch_idx = 0
        for batch in batchbar:
            datatime += time.time() - tic
            # -------------------------------------------------------------------------
            loss, mets = self._train_step(batch)
            # -------------------------------------------------------------------------
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batchtime += time.time() - tic
            tic = time.time()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for key, val in mets.items():
                self.train_metrics.update(key, val)
            batchbar.set_postfix({'train_'+k : v for k, v in self.train_metrics.result().items()})


            if batch_idx == self.len_epoch:
                break
            batch_idx += 1

        log = self.train_metrics.result()
        log = {'train_'+k : v for k, v in log.items()}

        if self.validloader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{'valid_'+k : v for k, v in val_log.items()})
        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        for batch_idx, batch in enumerate(self.validloader):
            # -------------------------------------------------------------------------
            loss, mets = self._valid_step(batch)
            # -------------------------------------------------------------------------
            self.writer.set_step((epoch - 1) * len(self.validloader) + batch_idx, 'valid')
            self.valid_metrics.update('loss', loss.item())
            for key, val in mets.items():
                self.valid_metrics.update(key, val)

        return self.valid_metrics.result()


    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            lr = self.optimizer.param_groups[0]['lr']
            log = {'epoch': epoch, 'lr': lr}
            log.update(result)

            # print logged informations to the screen
#             for key, value in log.items():
#                 self.logger.info('    {:20s}: {}'.format(str(key), value))
            toprint = ""
            for t in log.items():
                if t[0] == 'lr':
                    toprint = toprint + t[0] + ": " + str(t[1]) + "   "
                    continue
                if type(t[1]) == int:
                    toprint = toprint + '%s: %d      '%(t[0], t[1])
                else:
                    toprint = toprint + '%s: %.4f   '%(t[0], t[1])
            self.logger.info(toprint)
#             print(toprint)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(log[self.mnt_metric])
                else:
                    self.lr_scheduler.step()

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

            # add histogram of model parameters to the tensorboard
            self.writer.set_step(epoch)
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')


    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(), 
            'monitor_best': self.mnt_best
        }
        filename = str(self.checkpoint_dir / 'chkpt_last.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'chkpt_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: chkpt_best.pth ...")


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        try:
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.module.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.mnt_best = checkpoint['monitor_best']
        except KeyError:
            self.model.module.load_state_dict(checkpoint)

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
