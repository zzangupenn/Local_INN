import numpy as np
import matplotlib.pyplot as plt
import yaml
from argparse import Namespace

with open('levine_slam_dark.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

pos_record = np.load(conf.save_filename)['data_record']

ind = np.random.randint(0, pos_record.shape[0])
plt.plot(pos_record[:, 0], pos_record[:, 1], '.', markersize=1)
plt.plot(pos_record[ind, 3:] * np.cos(np.arange(0, 6.28, 6.28/360) + pos_record[ind, 2] + 3.14) + pos_record[ind, 0], 
         pos_record[ind, 3:] * np.sin(np.arange(0, 6.28, 6.28/360) + pos_record[ind, 2] + 3.14) + pos_record[ind, 1], '.', markersize=1)
plt.plot(pos_record[ind, 0], pos_record[ind, 1], '.r', markersize=5)
plt.show()


# plt.plot(pos_record[:, 0], pos_record[:, 1], '.', markersize=1)
# plt.show()