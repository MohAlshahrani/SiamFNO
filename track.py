import torch
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from tracker import track_sequence_folder
from models import SiamFCAlexNet, SiamFNO
import csv

checkpoint_path = "C:\\Users\\MohAl\\repos\\SiamFNO\\checkpoints\\epoch_001.pth"
checkpoint = torch.load(checkpoint_path, weights_only=False)
model = SiamFNO()
model.load_state_dict(checkpoint["model_state"], strict=False)

# seq_list = os.listdir('/content/drive/MyDrive/Training/2D/')
# for seq in seq_list:
#     trajectory, img_files = track_sequence_folder(model, f'/content/drive/MyDrive/Training/2D/{seq}', (263.31, 205.31))
#     # trajectory, img_files = track_sequence_folder(model, '/content/drive/MyDrive/train/MED-01-1', (238.14, 140.75))
#     # have to use sequence name
#     filename = f'trajectory_{seq}.csv'
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(trajectory)

device = "cuda" if torch.cuda.is_available() else "cpu"

trajectory, img_files = track_sequence_folder(model, r'C:\Users\MohAl\repos\SiamFNO\Training\2D\ETH-01-1', device=device, init_coord=(263.31, 205.31))
filename = 'trajectory_ETH-01-1.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(trajectory)