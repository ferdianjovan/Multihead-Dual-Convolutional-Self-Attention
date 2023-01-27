import torch
import numpy as np
from torch import optim
from collections import defaultdict


class Lookahead(optim.Optimizer):
    '''
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    '''
    def __init__(self, optimizer, alpha: float=0.5, k: int=6, pullback_momentum: str="none"):
        '''
        optimizer:inner optimizer
        k: number of lookahead steps
        alpha: linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum: change to inner optimizer momentum on interpolation update
        '''
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'k':self.k,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """
        Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure=None):
        """
        Performs a single Lookahead optimization step.
        closure: A closure that reevaluates the model and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(1.0 - self.alpha, param_state['cached_params'])  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(
                            1.0 - self.alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss
    
    
class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    Original code comes from https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(
        self, patience: int=7, verbose: bool=False, delta: float=0, path: str='checkpoint.pt', 
        minimum_epoch: int=10, trace_func=print
    ):
        """
        patience: How long to wait after last time validation accuracy improved.
        verbose: If True, prints a message for each validation accuracy improvement. 
        delta: Minimum change in the monitored quantity to qualify as an improvement.
        path: Path for the checkpoint to be saved to.
        trace_func (function): trace print function.            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = np.Inf
        self.delta = delta
        self.path = path
        self.epoch_counter = 0
        self.minimum_epoch = minimum_epoch
        self.trace_func = trace_func
        
    def __call__(self, accuracy, model: torch.nn.Module):
        
        self.epoch_counter += 1
        if self.epoch_counter >= self.minimum_epoch:
            if self.best_score is None:
                self.best_score = accuracy
                self.save_checkpoint(accuracy, model)
            elif accuracy < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = accuracy
                self.save_checkpoint(accuracy, model)
                self.counter = 0
            
    def save_checkpoint(self, val_acc, model: torch.nn.Module):
        '''Saves model when accuracy decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_acc_min:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_min = val_acc