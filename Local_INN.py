import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import os, sys

import time
from utils.torch_utils import PositionalEncoding, PositionalEncoding_torch
from utils.utils import ConfigJSON, DataProcessor, DrivableCritic

# FrEIA imports
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

if len(sys.argv) > 2:
    EXP_NAME = sys.argv[1]
    CUDA_NAME = sys.argv[2]
elif len(sys.argv) > 1:
    EXP_NAME = sys.argv[1]
    CUDA_NAME = 'cuda'
else:
    print('Please give scene and exp name.')
DATA_DIR = 'levine/'
DATAFILE = 'levine360_100k.npz'

INSTABILITY_RECOVER = 1
USE_MIX_PRECISION_TRAINING = 0
CONTINUE_TRAINING = 0
TRANSFER_TRAINING = 0
TRANSFER_EXP_NAME = ''
RE_PROCESS_DATA = 1

BATCHSIZE = int(200)
N_DIM = int(60)
COND_DIM = 6
COND_OUT_DIM = 12
LR = 5e-4
COND_NOISE = [0.2, 0.2, 15 / 180 * np.pi] # m, deg
SCAN_NOISE = 0.005


def main():
    from trainer import Trainer
    from tqdm import trange
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('results/tensorboard/' + EXP_NAME)
    device = torch.device(CUDA_NAME)
    
    class F110Dataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = torch.from_numpy(data).type('torch.FloatTensor').to(device)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]
    
    print("EXP_NAME", EXP_NAME)
    if not os.path.exists('results/' + EXP_NAME + '/'):
            os.mkdir('results/' + EXP_NAME + '/')
    with open(__file__, "r") as src:
        with open('results/' + EXP_NAME + '/' + EXP_NAME + '.py', "w") as tgt:
            tgt.write(src.read())
    with open('results/' + EXP_NAME + '/' + EXP_NAME + '.txt', "a") as tgt:
        tgt.writelines(EXP_NAME + '\n')
            
    if os.path.exists('train_data.npz') and not RE_PROCESS_DATA:
        total_data = np.load('train_data.npz')
        c = ConfigJSON()
        c.load_file('train_data.json')
        c.save_file('results/' + EXP_NAME + '/' + EXP_NAME + '.json')
    else:
        data_dir = DATA_DIR
        total_data = np.load(data_dir + DATAFILE)['data_record']
        print('Data loaded.')
        
        dp = DataProcessor()
        c = ConfigJSON()
        drivable_critic = DrivableCritic(DATA_DIR, 'levine_slam_dark.yaml')
        normalize_params = drivable_critic.get_normalize_params()
        total_data[:, 0], c.d['normalization_x'] = dp.data_normalize(total_data[:, 0])
        total_data[:, 1], c.d['normalization_y'] = dp.data_normalize(total_data[:, 1])
        total_data[:, 2] = dp.two_pi_warp(total_data[:, 2])
        c.d['normalization_theta'] = list(normalize_params[2])
        c.d['normalization_laser'] = list(normalize_params[3])
        total_data[:, 3:] = dp.runtime_normalize(total_data[:, 3:], normalize_params[3])
        c.save_file('results/' + EXP_NAME + '/' + EXP_NAME + '.json')
        c.save_file('train_data.json')
        
        ## Augment the data
        def scan_ind_warp(inds, num):
            inds[np.where(inds >= num)] = inds[np.where(inds >= num)] - num
            return inds
        
        total_scans = total_data[:, 3:]
        print(total_scans.shape)
        scan_len = 360
        augmented_data = []
        for ind_0 in trange(total_data.shape[0]):
            randints = np.concatenate([np.array([0]), np.random.choice(np.arange(10, 270), 9, replace=False)])
            for ind_1 in randints:
                new_scan_inds = scan_ind_warp(np.arange(ind_1 + 45, ind_1 + 270 + 45), scan_len)
                new_angle = (np.pi * 2 / scan_len) * ind_1 + total_data[ind_0, 2]
                new_data = np.concatenate([total_data[ind_0, :2], np.array([new_angle]), total_scans[ind_0, new_scan_inds]])
                augmented_data.append(new_data)
        total_data = np.array(augmented_data)
        total_data[:, 2] = dp.two_pi_warp(total_data[:, 2]) # rewarp the theta
        total_data[:, 2] = dp.runtime_normalize(total_data[:, 2], normalize_params[2]) # theta is normalized after augmenting
        
        augmented_data = []
        positional_encoding = PositionalEncoding(L = 10)
        for ind_0 in trange(total_data.shape[0]):
            encoded_data = []
            for k in range(3):
                if k == 2:
                    sine_part, cosine_part = positional_encoding.encode_even(total_data[ind_0, k])
                else:
                    sine_part, cosine_part = positional_encoding.encode(total_data[ind_0, k])
                encoded_data.append(sine_part)
                encoded_data.append(cosine_part)
            encoded_data = np.array(encoded_data)
            encoded_data = encoded_data.flatten('F')
            
            new_data = np.concatenate([encoded_data,
                                    total_data[ind_0, 3:], 
                                    total_data[ind_0, :3]])
            augmented_data.append(new_data)
        total_data = np.array(augmented_data)
        np.save('total_data.npy', total_data)
        

    print(total_data.shape)
    train_data = total_data[0:total_data.shape[0] - 5000]
    test_data = total_data[total_data.shape[0] - 5000:]
    p_encoding_t = PositionalEncoding_torch(1, device)
    
    cond_noise = np.array(COND_NOISE) # m, rad
    cond_noise[0] /= drivable_critic.x_in_m
    cond_noise[1] /= drivable_critic.y_in_m
    cond_noise[2] /= np.pi * 2
    cond_noise = torch.from_numpy(cond_noise).type('torch.FloatTensor').to(device)
    
    train_set = F110Dataset(train_data)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCHSIZE, shuffle=True, drop_last=False)
    test_set = F110Dataset(test_data)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCHSIZE, shuffle=True, drop_last=False)
    l1_loss = torch.nn.L1Loss()
    
    ### Training
    trainer = Trainer(EXP_NAME, 500, 0.0001, device,
                      LR, [300], 0.1, 'exponential',
                      INSTABILITY_RECOVER, 3, 0.99, USE_MIX_PRECISION_TRAINING)

    model = Local_INN(device)
    model.to(device)
    if CONTINUE_TRAINING:
        ret = trainer.continue_train_load(model, path='results/' + EXP_NAME + '/', transfer=TRANSFER_TRAINING)
        if ret is not None:
            print('Continue Training')
            model = ret
            
    if TRANSFER_TRAINING:
        ret = trainer.continue_train_load(model, path='results/' + TRANSFER_EXP_NAME + '/', transfer=TRANSFER_TRAINING, transfer_exp_name=TRANSFER_EXP_NAME)
        if ret is not None:
            print('Transfer Training')
            model = ret
    
    current_lr = LR
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
    optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})
    n_hypo = 20
    epoch_time = 0
    mix_precision = False
    scaler = GradScaler(enabled=mix_precision)
    
    while(not trainer.is_done()):
        epoch = trainer.epoch
        epoch_info = np.zeros(7)
        epoch_info[3] = epoch
        epoch_time_start = time.time()

        if USE_MIX_PRECISION_TRAINING:
            scaler = GradScaler(enabled=mix_precision)
        
        trainer_lr = trainer.get_lr()
        if trainer_lr != current_lr:
            current_lr = trainer_lr
            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.param_groups[1]['lr'] = current_lr
            optimizer.param_groups[2]['lr'] = current_lr
            optimizer.param_groups[3]['lr'] = current_lr
            
        model.train()
        model.vae.encoder.train()
        model.vae.decoder.train()
        for data in train_loader:
            optimizer.zero_grad()
            with autocast(enabled=mix_precision):
                
                x_hat_gt = data[:, :60]
                cond = data[:, 60+270:60+270+3]
                y_gt = data[:, 60:60+270]
                # y_gt += y_gt * torch.zeros_like(y_gt, device=device).normal_(0., SCAN_NOISE)
                cond += torch.zeros_like(cond, device=device).normal_(0., 1.) * cond_noise
                cond = p_encoding_t.batch_encode(cond.round(decimals=1))
                
                y_hat_vae = torch.zeros_like(x_hat_gt, device=device)
                y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
                y_hat_inn, _ = model(x_hat_gt, cond)
                y_inn = model.vae.decoder.forward(y_hat_inn[:, :-6])
                # y_vae = model.vae.decoder.forward(y_hat_vae[:, :-6])
                
                vae_kl_loss = model.vae.encoder.kl * 0.0001
                # vae_recon_loss = l1_loss(y_vae, y_gt)
                inn_recon_loss = l1_loss(y_inn, y_gt)
                y_hat_inn_loss = l1_loss(y_hat_inn[:, :-6], y_hat_vae[:, :-6])
                loss_forward = vae_kl_loss + inn_recon_loss + y_hat_inn_loss
                epoch_info[0] += loss_forward.item()
                epoch_info[1] += inn_recon_loss.item()
            scaler.scale(loss_forward).backward(retain_graph=True)
            with autocast(enabled=mix_precision):
                y_hat_vae[:, -6:] = 0
                x_hat_0, _ = model.reverse(y_hat_vae, cond)
                loss_reverse = l1_loss(x_hat_0[:, :12], x_hat_gt[:, :12])
                
                batch_size = y_gt.shape[0]
                z_samples = torch.cuda.FloatTensor(n_hypo, batch_size, 6, device=device).normal_(0., 1.)
                y_hat = y_hat_vae[None, :, :54].repeat(n_hypo, 1, 1)
                y_hat_z_samples = torch.cat((y_hat, z_samples), dim=2).view(-1, 60)
                cond = cond[None].repeat(n_hypo, 1, 1).view(-1, COND_DIM)
                x_hat_i = model.reverse(y_hat_z_samples, cond)[0].view(n_hypo, batch_size, 60)
                x_hat_i_loss = torch.mean(torch.min(torch.mean(torch.abs(x_hat_i[:, :, :12] - x_hat_gt[:, :12]), dim=2), dim=0)[0])
                loss_reverse += x_hat_i_loss
                epoch_info[2] += loss_reverse.item()
                
            scaler.scale(loss_reverse).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
            
            scaler.step(optimizer)
            scaler.update()
            
        epoch_info[:3] /= len(train_loader)
        
        # testing
        model.eval()
        model.vae.encoder.eval()
        model.vae.decoder.eval()
        epoch_posit_err = []
        epoch_orient_err = []
        with torch.no_grad():
            for data in test_loader:
                with autocast(enabled=mix_precision):
                    x_hat_gt = data[:, :60]
                    x_gt = data[:, 60+270:60+270+3]
                    cond = data[:, 60+270:60+270+3]
                    y_gt = data[:, 60:60+270]
                    # y_gt += y_gt * torch.zeros_like(y_gt, device=device).normal_(0., SCAN_NOISE)
                    cond += torch.zeros_like(cond, device=device).normal_(0., 1.) * cond_noise
                    cond = p_encoding_t.batch_encode(cond.round(decimals=1))
                    
                    y_hat_vae = torch.zeros_like(x_hat_gt, device=device)
                    y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
                    x_hat_0, _ = model.reverse(y_hat_vae, cond)
                    
                    result_posit = torch.zeros((x_hat_gt.shape[0], 2)).to(device)
                    result_posit[:, 0] = dp.de_normalize(p_encoding_t.batch_decode(x_hat_0[:, 0], x_hat_0[:, 1]), c.d['normalization_x'])
                    result_posit[:, 1] = dp.de_normalize(p_encoding_t.batch_decode(x_hat_0[:, 2], x_hat_0[:, 3]), c.d['normalization_y'])
                    gt_posit = torch.zeros((x_hat_gt.shape[0], 2)).to(device)
                    gt_posit[:, 0] = dp.de_normalize(x_gt[:, 0], c.d['normalization_x'])
                    gt_posit[:, 1] = dp.de_normalize(x_gt[:, 1], c.d['normalization_y'])
                    epoch_posit_err.append(torch.median(torch.norm(result_posit[:, 0:2] - gt_posit[:, 0:2], dim=1)))
                    
                    result_angles = torch.zeros((x_hat_gt.shape[0], 1)).to(device)
                    result_angles[:, 0] = p_encoding_t.batch_decode_even(x_hat_0[:, 4], x_hat_0[:, 5])
                    orient_err = torch.abs(result_angles - x_gt[:, 2]) * 2 * np.pi
                    epoch_orient_err.append(torch.median(orient_err))

        epoch_time = (time.time() - epoch_time_start)
        remaining_time = (trainer.max_epoch - epoch) * epoch_time / 3600
        
        epoch_info[4] = torch.median(torch.stack(epoch_posit_err))
        epoch_info[5] = torch.median(torch.stack(epoch_orient_err))

        model, return_text, mix_precision = trainer.step(model, epoch_info, mix_precision)
        if return_text == 'instable':
            optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
            optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})
        
        
        writer.add_scalar("INN/0_forward", epoch_info[0], epoch)
        writer.add_scalar("INN/1_recon", epoch_info[1], epoch)
        writer.add_scalar("INN/2_reverse", epoch_info[2], epoch)
        writer.add_scalar("INN/3_T_pos", epoch_info[4], epoch)
        writer.add_scalar("INN/4_T_yaw", epoch_info[5], epoch)
        writer.add_scalar("INN/5_LR", current_lr, epoch)
        writer.add_scalar("INN/6_remaining(h)", remaining_time, epoch)
        
        text_print = "Epoch {:d}".format(epoch) + \
            ' |f {:.5f}'.format(epoch_info[0]) + \
            ' |recon {:.5f}'.format(epoch_info[1]) + \
            ' |r {:.5f}'.format(epoch_info[2]) + \
            ' |T_pos {:.5f}'.format(epoch_info[4]) + \
            ' |T_yaw {:.5f}'.format(epoch_info[5]) + \
            ' |h_left {:.1f}'.format(remaining_time) + \
            ' | ' + return_text
        print(text_print)
        with open('results/' + EXP_NAME + '/' + EXP_NAME + '.txt', "a") as tgt:
            tgt.writelines(text_print + '\n')
    writer.flush()
    

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(270, 270)
        self.linear2 = nn.Linear(270, 54)
        self.linear3 = nn.Linear(270, 54)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
        return z
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(54, 270)
        self.linear2 = nn.Linear(270, 270)
        
    def forward(self, z):
        z = nn.functional.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class Local_INN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = self.build_inn()
        
        
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for param in self.trainable_parameters:
            param.data = 0.05 * torch.randn_like(param)
        self.cond_net = self.subnet_cond(COND_OUT_DIM)
        self.vae = VAE()
        
    def subnet_cond(self, c_out):
        return nn.Sequential(nn.Linear(COND_DIM, 256), nn.ReLU(),
                             nn.Linear(256, COND_OUT_DIM))
        
    def build_inn(self):

        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, 1024), nn.ReLU(),
                                    nn.Linear(1024, c_out))

        nodes = [InputNode(N_DIM, name='input')]
        cond = ConditionNode(COND_OUT_DIM, name='condition')
        for k in range(6):
            nodes.append(Node(nodes[-1],
                                GLOWCouplingBlock,
                                {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                                conditions=cond,
                                name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                                PermuteRandom,
                                {'seed': k},
                                name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))
            
        return ReversibleGraphNet(nodes + [cond], verbose=False).to(self.device)
    
    def forward(self, x, cond):
        return self.model(x, self.cond_net(cond))
    
    def reverse(self, y_rev, cond):
        return self.model(y_rev, self.cond_net(cond), rev=True)
    

if __name__ == '__main__':
    main()
