from copy import deepcopy


class LRSchedular(object):
    # TODO: add checkpoint for model and optimizer
    def __init__(self, optimizer, decay=0.3, model=None,
                 threshold=10, checkpoint=None, init=50, restart=None):
        self.global_counter = 0
        self.epoch_counter = 0
        self.threshold = threshold
        self.best = None
        self.optim = optimizer
        self.decay = decay
        self.current_lr = None
        # model checkpoint
        self.init = init
        self.model = model
        self.model_state = None
        self.optim_state = None
        self.checkpoint = checkpoint
        self.best_epoch = None
        self.reload = False
        self.restart = restart

    def step(self, val):
        self.epoch_counter += 1
        self.global_counter += 1
        self.reload = False
        # self.current_lr = self.get_lr()
        self.current_lr = max(self.get_lr(), 1.e-7)

        print '-----> history best: {0}, past epoch {1}, lr {2}'.format(self.best,
                                                                               self.epoch_counter,
                                                                               self.current_lr)
        if self.improve(val):
            self.epoch_counter = 0
            self.save_model()
        else:
            if self.epoch_counter >= self.threshold:
                self.load_model()
                self.decay_lr()
                self.epoch_counter = 0
            else:
                pass

    def decay_lr(self):
        if self.restart is None:
            change_LR(self.optim, self.decay, info=True, lr=self.current_lr)
        else:
            lr = self.current_lr
            self.optim = self.restart(self.model, lr * self.decay)

    def get_lr(self):
        lr = self.optim.param_groups[0]['lr']
        return lr

    def improve(self, val):
        # monitor accuracy
        if self.best is None:
            self.best = val
            return True
        elif self.best > val:
            print '-------------> replace best result {0} with current result {1}'.format(self.best, val)
            self.best = val
            return True
        else:
            return False

    def save_model(self):
        if self.checkpoint and self.global_counter >= self.init:
            self.model_state = deepcopy(self.model.state_dict())
            self.optim_state = deepcopy(self.optim.state_dict())
            self.best_epoch = self.global_counter
            print '-------------> [save model]: save best model at {0}'.format(self.best_epoch)
        else:
            pass

    def load_model(self):
        if self.checkpoint and self.model_state and self.optim_state:
            self.model.load_state_dict(self.model_state)
            self.optim.load_state_dict(self.optim_state)
            print '-------------> [load model]: load best model at {0}'.format(self.best_epoch)
            self.reload = True
        else:
            pass


class LRSchedularByEpoch(LRSchedular):
    def __init__(self, optimizer, decay=0.3, model=None,
                 threshold=10, checkpoint=None, init=30, restart=None):
        super(LRSchedularByEpoch, self).__init__(optimizer, decay, model,
                 threshold, checkpoint, init, restart)

    def step(self, val):
        self.epoch_counter += 1
        self.global_counter += 1
        self.reload = False
        self.current_lr = max(self.get_lr(), 1.e-5)

        print '-----> history best: {0}, past epoch {1}, lr {2}'.format(self.best,
                                                                               self.epoch_counter,
                                                                               self.current_lr)
        if self.epoch_counter >= self.threshold and self.epoch_counter >= self.init:
            change_LR(self.optim, self.decay, info=True, lr=self.current_lr)
            self.epoch_counter = 0
            self.init = 0
        else:
            pass


def change_LR(optimizer, decay, info=False, lr=None):
    for group in optimizer.param_groups:
        if lr is None:
            lr = group['lr']
        group['lr'] = lr*decay
        if info:
            print 'change LR from {0} to {1}'.format(lr, lr*decay)
    return None