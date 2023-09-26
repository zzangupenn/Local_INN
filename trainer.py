import torch
import torch.nn as nn
import numpy as np
import os


class Trainer():
    def __init__(self, exp_name, max_epoch, goal_loss, device,
                 initial_rl, rl_schedule_epoch, rl_schedule_gamma = 0.5, lr_method = 'step',
                 instability_recover = False,
                 instable_thresh = 3, instable_gamma = 0.5, use_mix_precision = 0):
        
        self.rl_schedule_epoch = rl_schedule_epoch + [max_epoch]
        self.rl_schedule_gamma = rl_schedule_gamma
        self.rl_schedule = np.ones_like(self.rl_schedule_epoch) * initial_rl
        for ind in range(len(self.rl_schedule)):
            self.rl_schedule[ind] = self.rl_schedule[ind] * rl_schedule_gamma ** (ind + 1)
        self.instable_thresh = instable_thresh
        self.instable_gamma = instable_gamma
        self.max_epoch = max_epoch
        self.goal_loss = goal_loss
        self.exp_name = exp_name
        self.device = device
        self.instability_recover = instability_recover
        self.use_mix_precision = use_mix_precision
        self.mix_precision_relax_cnt = 2
        self.lr_method = lr_method
        
        self.lr = initial_rl
        self.lr_schedule = initial_rl
        self.epoch = 0
        self.best_pos_loss = float('inf')
        self.best_ori_loss = float('inf')
        self.best_epoch_info = np.ones(7) * float('inf')
        self.prev_epoch_info = np.ones(7) * float('inf')
        self.current_schedule = 0
        self.path = 'results/' + exp_name + '/'
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            
    
    
    def is_done(self):
        if self.epoch >= self.max_epoch - 1 or \
           self.best_epoch_info[2] <= self.goal_loss and self.best_epoch_info[2] > 0: # train_reverse loss   
            return True
        else:
            return False
        
    
    def get_lr(self):

        if self.lr_method == 'step':
            if self.epoch == self.rl_schedule_epoch[self.current_schedule]:
                if self.lr > self.rl_schedule[self.current_schedule]:
                    self.lr = self.rl_schedule[self.current_schedule]
                    self.current_schedule += 1
                    print('Change LR to', self.lr)
        elif self.lr_method == 'linear':
            if self.epoch == self.rl_schedule_epoch[self.current_schedule]:
                self.current_schedule += 1
            if self.lr > self.rl_schedule[self.current_schedule]:
                self.lr += (self.rl_schedule[self.current_schedule] - self.lr) / (self.rl_schedule_epoch[self.current_schedule] - self.epoch)
        elif self.lr_method == 'exponential':
            if self.epoch == 0:
                return self.lr
            factor_value = self.rl_schedule_gamma ** (1/self.rl_schedule_epoch[0])
            self.lr_schedule *= factor_value
            if self.lr > self.lr_schedule:
                self.lr = self.lr_schedule
        elif self.lr_method == 'exponential_stop':
            if self.epoch > 0 and self.epoch <= self.rl_schedule_epoch[0]:
                factor_value = self.rl_schedule_gamma ** (1/self.rl_schedule_epoch[0])
                self.lr *= factor_value

        return self.lr
        
        
    def save_model(self, model, model_epoch_info, save_name, path = ""):
        # save the model
        filename = path + self.exp_name + '_model_' + save_name + '.pt'
        torch.save(model.state_dict(), filename)

        filename = path + self.exp_name + '_model_' + save_name + '.npy'
        np.save(filename, model_epoch_info)
        
        
    def load_model(self, model, model_name, path = ''):
        if path.split('.')[-1] == 'pt':
            filename = path
        else:
            filename = path + self.exp_name + '_model_' + model_name + '.pt'
        print('Load from model:', filename)
        model.load_state_dict(torch.load(filename))
        model.to(self.device)
        return model

    def continue_train_load(self, model, path = '', transfer=0, transfer_exp_name=""):
        model_name = 'best'
        if path.split('.')[-1] == 'pt':
            filename = path
            info_filename = path.split('.')[-2] + '.npy'
        else:
            if transfer:
                filename = path + transfer_exp_name + '_model_' + model_name + '.pt'
                info_filename = path + transfer_exp_name + '_model_' + model_name + '.npy'
            else:
                filename = path + self.exp_name + '_model_' + model_name + '.pt'
                info_filename = path + self.exp_name + '_model_' + model_name + '.npy'
        if os.path.exists(filename):
            print('Load from model:', filename)
            model.load_state_dict(torch.load(filename))
            model.to(self.device)
            
            self.best_epoch_info = np.load(info_filename)
            self.prev_epoch_info = self.best_epoch_info.copy()
            self.best_pos_loss = self.get_best_merit(self.best_epoch_info, rank=1)
            self.best_ori_loss = self.get_best_merit(self.best_epoch_info, rank=2)
            
            for past_epoch in range(int(self.best_epoch_info[3].copy()) + 1):
                self.epoch = past_epoch
                _ = self.get_lr()
            self.epoch = int(self.best_epoch_info[3].copy()) + 1

            if transfer:
                self.best_pos_loss = np.float('inf')
                self.best_ori_loss = np.float('inf')
                self.best_epoch_info[0:3] = np.float('inf')

            return model
        else:
            return None
        
    
    def get_best_merit(self, epoch_info, rank=0):
        if rank == 0:
            return epoch_info[6] + epoch_info[5]
        elif rank == 1:
            return epoch_info[6]
        elif rank == 2:
            return epoch_info[5]
    
    
    def step(self, model, epoch_info, step_use_mix):
        path = self.path
        self.save_model(model, epoch_info, 'last', path)
        return_text = ''
        use_mix_precision = self.use_mix_precision
        
        if self.get_best_merit(epoch_info, 1) < self.best_pos_loss:
            self.best_pos_loss = self.get_best_merit(epoch_info, 1).copy()
            self.save_model(model, epoch_info, 'best_pos', path)
            return_text = 'best_pos'
        
        if self.get_best_merit(epoch_info, 2) < self.best_ori_loss:
            self.best_ori_loss = self.get_best_merit(epoch_info, 2).copy()
            self.save_model(model, epoch_info, 'best_ori', path)
            return_text = 'best_ori'
        
        if self.get_best_merit(self.best_epoch_info) > self.get_best_merit(epoch_info):
            self.best_epoch_info = epoch_info.copy()
            self.save_model(model, epoch_info, 'best', path)
            return_text = 'best'
        
        if self.instability_recover and (np.sum(epoch_info[0:2]) > np.sum(self.best_epoch_info[0:2]) * self.instable_thresh \
                                         or any(np.isnan(epoch_info))):
            # detect instable epoch
            print('Instable epoch detected.')
            model = self.load_model(model, 'best', path)
            self.lr *= self.instable_gamma
            print('Change LR to', self.lr)
            self.epoch = int(self.best_epoch_info[3].copy()) + 1
            if any(np.isnan(epoch_info)) and step_use_mix and self.use_mix_precision:
                use_mix_precision = 0
                self.mix_precision_relax_cnt = 5
            return_text = 'instable'
        else:
            self.prev_epoch_info = epoch_info.copy()
            self.epoch += 1
        
        if self.mix_precision_relax_cnt > 0 and self.use_mix_precision:
            use_mix_precision = 0
            self.mix_precision_relax_cnt -= 1
        
        return model, return_text, use_mix_precision == 1
