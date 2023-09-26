import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from tqdm import tqdm 
import random


def main():
    """
    main entry point
    """
        
    with open('levine_slam_dark.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)

    laptime = 0.0
    start = time.time()
    map_origin = conf.origin
    cnt = 0
    data_record = []
    pos_record = []
    with tqdm(total = conf.sample_num) as pbar:
        while cnt < conf.sample_num:
            sample_pos = [random.uniform(map_origin[0], 50+map_origin[0]),
                        random.uniform(map_origin[1], 40+map_origin[1]),
                        random.uniform(0, np.pi*2)]
            obs, step_reward, done, info = env.reset(np.array([sample_pos]))
            # env.render(mode='human')
            if not done:
                pos_record.append(sample_pos)
                data_record.append(np.concatenate([np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]]), np.array(obs['scans'][0])[np.arange(0, 1440, 4)]]))
                # print(cnt)
                cnt += 1
                pbar.update(1)
            
    data_record = np.array(data_record)
    print(data_record.shape)
    np.savez_compressed(conf.save_filename, data_record=data_record)
    pos_record = np.array(pos_record)

        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()
