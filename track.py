import torch
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from tracker import track_sequence_folder
from models import SiamFCAlexNet
import csv
checkpoint_path = "/content/drive/MyDrive/SiamFC/checkpoints/epoch_050.pth"
checkpoint = torch.load(checkpoint_path)
# model = SiamFC()
model = SiamFCAlexNet()
model.load_state_dict(checkpoint["model_state"])

trajectory, img_files = track_sequence_folder(model, '/content/drive/MyDrive/Training/2D/ETH-01-1', (263.31, 205.31))
# trajectory, img_files = track_sequence_folder(model, '/content/drive/MyDrive/train/MED-01-1', (238.14, 140.75))
# have to use sequence name
filename = 'trajectory.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(trajectory)