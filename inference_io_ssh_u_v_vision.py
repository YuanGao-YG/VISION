import xarray as xr
import torch as pt
from torch import nn
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F
import os
from model.vision import VISION
import os.path as osp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import h5py


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

high1 = 0
high2 = 512
width1 = 0
width2 = 512

class testDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = './data/KD48_demo.h5'
        self.data_file = h5py.File(self.data_path, 'r')
        self.mean = np.load('./data/mean.npy')
        self.std = np.load('./data/std.npy')
        self.size = 50

    def __getitem__(self, index):
        
        data = self.data_file['fields'][index, 0:1, high1:high2, width1:width2]
        data = np.nan_to_num(data, nan=0)
        ssh = data[0]
        ssh = (ssh - self.mean[0, 0, :, :])/(self.std[0, 0, :, :])

        data_u = self.data_file['fields'][index, 1:2, high1:high2, width1:width2]
        data_u = np.nan_to_num(data_u, nan=0)
        u = data_u[0]
        u = (u - self.mean[0, 1, :, :])/(self.std[0, 1, :, :])

        data_v = self.data_file['fields'][index, 2:3, high1:high2, width1:width2]
        data_v = np.nan_to_num(data_v, nan=0)
        v = data_v[0]
        v = (v - self.mean[0, 2, :, :])/(self.std[0, 2, :, :])

        data_w_20 = self.data_file['fields'][index, 3:4, high1:high2, width1:width2]
        data_w_20 = np.nan_to_num(data_w_20, nan=0)
        w_20 = data_w_20[0]
        w_20 = (w_20 - self.mean[0, 3, :, :])/(self.std[0, 3, :, :])

        data_w_40 = self.data_file['fields'][index, 4:5, high1:high2, width1:width2]
        data_w_40 = np.nan_to_num(data_w_40, nan=0)
        w_40 = data_w_40[0]
        w_40 = (w_40 - self.mean[0, 4, :, :])/(self.std[0, 4, :, :])

        data_w_60 = self.data_file['fields'][index, 5:6, high1:high2, width1:width2]
        data_w_60 = np.nan_to_num(data_w_60, nan=0)
        w_60 = data_w_60[0]
        w_60 = (w_60 - self.mean[0, 5, :, :])/(self.std[0, 5, :, :])

        data_b = self.data_file['fields'][index, 8:9, high1:high2, width1:width2]
        data_b = np.nan_to_num(data_b, nan=0)
        b = data_b[0]
        b = (b - self.mean[0, 8, :, :])/(self.std[0, 8, :, :])

        return np.stack((ssh, u, v, b, w_20, w_40, w_60), axis=0)

    def __len__(self):
        return self.size

    def __del__(self):
        self.data_file.close()


testdataset = testDataset()
testloader=Data.DataLoader(
    dataset=testdataset,
    batch_size=1,
    shuffle=False, 
    num_workers=0
)

model = VISION().cuda()
checkpoint_path = './checkpoint_VISION/best_mse.pt'
ckpt = pt.load(checkpoint_path, map_location='cpu')
state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)

model.eval()

folder_path = './result'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

output_path = osp.join(folder_path, 'results_io_ssh_u_v_vision.h5')
if os.path.exists(output_path):
    os.remove(output_path)

N = len(testdataset)
H = 512
W = 512
f_out = h5py.File(output_path, 'w')
dset_pred = f_out.create_dataset('predicted', shape=(N, 3, H, W), dtype='float32')
dset_true = f_out.create_dataset('ground_truth', shape=(N, 3, H, W), dtype='float32')


buffer_preds = []
buffer_trues = []
buffer_indices = []

batch_size_to_save = 1
current_count = 0

with pt.no_grad():
    num = 0
    for data in tqdm(testloader, desc="Loading data"):
        xbatch = data[:, 0:3, :, :].cuda().float()
        ybatch = data[:, 4:7, :, :].cuda().float()
        out = model(xbatch)
        print(out.shape)

        mse = pt.mean((ybatch - out) ** 2)
        print(num, mse)

        preds_np = out.detach().cpu().numpy().astype(np.float32)
        trues_np = ybatch.detach().cpu().numpy().astype(np.float32)

        buffer_preds.append(preds_np)
        buffer_trues.append(trues_np)
        buffer_indices.append(num)

        if len(buffer_preds) == batch_size_to_save:
            preds_block = np.concatenate(buffer_preds, axis=0)
            trues_block = np.concatenate(buffer_trues, axis=0)
            indices_block = buffer_indices

            dset_pred[indices_block, 0, :, :] = preds_block[:, 0, :, :]
            dset_true[indices_block, 0, :, :] = trues_block[:, 0, :, :]
            
            dset_pred[indices_block, 1, :, :] = preds_block[:, 1, :, :]
            dset_true[indices_block, 1, :, :] = trues_block[:, 1, :, :]
            
            dset_pred[indices_block, 2, :, :] = preds_block[:, 2, :, :]
            dset_true[indices_block, 2, :, :] = trues_block[:, 2, :, :]

            buffer_preds.clear()
            buffer_trues.clear()
            buffer_indices.clear()

        num += 1

    if len(buffer_preds) > 0:
        preds_block = np.concatenate(buffer_preds, axis=0)  
        trues_block = np.concatenate(buffer_trues, axis=0)  
        indices_block = buffer_indices

        dset_pred[indices_block, 0, :, :] = preds_block[:, 0, :, :]
        dset_true[indices_block, 0, :, :] = trues_block[:, 0, :, :]
        
        dset_pred[indices_block, 1, :, :] = preds_block[:, 1, :, :]
        dset_true[indices_block, 1, :, :] = trues_block[:, 1, :, :]
        
        dset_pred[indices_block, 2, :, :] = preds_block[:, 2, :, :]
        dset_true[indices_block, 2, :, :] = trues_block[:, 2, :, :]

        buffer_preds.clear()
        buffer_trues.clear()
        buffer_indices.clear()

f_out.close()
print("Results successfully saved to HDF5 file.")
