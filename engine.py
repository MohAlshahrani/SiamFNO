import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from utils import create_gaussian_response, create_ce_target

def train_siamfc(model, loader, epochs=5, template_size=127, search_size=255, sigma=2, device='cuda', checkpoint_dir="/content/drive/MyDrive/SiameseTracker/checkpoints"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    for e in range(epochs):
        model.train()
        epoch_loss = 0
        for t, s, disp in tqdm(loader, desc=f"Epoch {e+1}/{epochs}"):
            t, s, disp = t.to(device), s.to(device), disp.to(device)
            resp = model(t, s)
            resp_norm = (resp - resp.mean()) / resp.std()
            B, _, H, W = resp.shape
            gt = create_gaussian_response(disp, H, search_size, template_size, sigma, device)
            loss = F.mse_loss(resp_norm, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {e+1}: Loss={epoch_loss/len(loader):.4f}")
        
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{e+1:03d}.pth")
        avg_loss = epoch_loss / len(loader)
        checkpoint = {
            "epoch": e + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "avg_loss": avg_loss,
        }

        torch.save(checkpoint, ckpt_path)
        print(f" Saved checkpoint: {ckpt_path}\n")
    return model

def train_siamfc_logits(model, loader, epochs=5, template_size=127, search_size=255, sigma=2, device='cuda', checkpoint_dir="/content/drive/MyDrive/SiameseTracker/checkpoints"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    for e in range(epochs):
        model.train()
        epoch_loss = 0
        for t, s, disp in tqdm(loader, desc=f"Epoch {e+1}/{epochs}"):
            t, s, disp = t.to(device), s.to(device), disp.to(device)
            resp = model(t, s)
            resp_norm = (resp - resp.mean()) / resp.std()
            # print(resp.size())
            B, _, H, W = resp.shape
            logits = resp_norm.view(B, H*W)     # flatten
            # CE target (index)
            target = create_ce_target(disp, H, search_size, template_size, device)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {e+1}: Loss={epoch_loss/len(loader):.4f}")
        
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{e+1:03d}.pth")
        avg_loss = epoch_loss / len(loader)
        checkpoint = {
            "epoch": e + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "avg_loss": avg_loss,
        }

        torch.save(checkpoint, ckpt_path)
        print(f" Saved checkpoint: {ckpt_path}\n")
    return model
