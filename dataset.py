import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import siamfc_crop_and_resize

# class CLUSTLandmarkDataset(Dataset):
#     def __init__(self, root_dir, resolution_map, template_size=127, search_size=255):
#         """
#         Args:
#             root_dir: root path containing subfolders like ETH-01-1, MED-01-1, etc.
#             resolution_map: dict mapping {sequence_name: pixel_resolution_mm}
#             template_size, search_size: patch dimensions in pixels
#         """
#         self.template_size = template_size
#         self.search_size = search_size
#         self.resolution_map = resolution_map
#         self.samples = []

#         seq_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
#                            if os.path.isdir(os.path.join(root_dir, d))])
        
#         for seq_dir in seq_dirs:
#             seq_name = os.path.basename(seq_dir)
#             if seq_name not in resolution_map:
#                 print(f"[Warning] No resolution for {seq_name}, skipping...")
#                 continue

#             pixel_res = resolution_map[seq_name]
#             img_dir = os.path.join(seq_dir, "Data")
#             ann_dir = os.path.join(seq_dir, "Annotation")
#             if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
#                 continue
          
#             # Load all frame images
#             img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))])
#             frame_ids = [int(os.path.splitext(f)[0]) for f in img_files]
#             img_map = {fid: os.path.join(img_dir, f) for fid, f in zip(frame_ids, img_files)}
            
#             # Each annotation file corresponds to one landmark target
#             for ann_file in sorted(os.listdir(ann_dir)):
#                 if not ann_file.endswith('.txt'):
#                     continue
#                 ann_path = os.path.join(ann_dir, ann_file)
#                 ann = np.loadtxt(ann_path, dtype=float)
#                 if ann.ndim == 1:  # handle single-line annotation files
#                     ann = ann[None, :]
#                 ann = ann[np.argsort(ann[:, 0])]  # sort by frame id

#                 # Build template-search pairs for consecutive annotated frames
#                 for i in range(1, len(ann)):
#                     fid_prev, x_mm_prev, y_mm_prev = ann[i - 1]
#                     fid_curr, x_mm_curr, y_mm_curr = ann[i]

#                     x_prev, y_prev = x_mm_prev, y_mm_prev
#                     x_curr, y_curr = x_mm_curr, y_mm_curr

#                     if fid_prev not in img_map or fid_curr not in img_map:
#                         continue

#                     disp = (x_curr - x_prev, y_curr - y_prev)

#                     self.samples.append({
#                         "template_path": img_map[fid_prev],
#                         "search_path": img_map[fid_curr],
#                         "template_center": (x_prev, y_prev),
#                         "search_center": (x_prev, y_prev),  # crop around old pos
#                         "gt_disp": disp,
#                         "seq_name": seq_name,
#                         "resolution": pixel_res
#                     })

#         print(f"Loaded {len(self.samples)} training pairs from {len(seq_dirs)} sequences.")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         s = self.samples[idx]
#         img_t = cv2.imread(s["template_path"], cv2.IMREAD_GRAYSCALE)
#         img_s = cv2.imread(s["search_path"], cv2.IMREAD_GRAYSCALE)

#         # bbox_t = (s["template_center"][0], s["template_center"][1], 50, 50) 
#         # bbox_s = (s["search_center"][0], s["search_center"][1], 50, 50)
#         # t_patch, _ = siamfc_crop_and_resize(img_t, bbox_t)
#         # _, s_patch = siamfc_crop_and_resize(img_s, bbox_s)

#         #############################################################################
#         ## if working with FNO, we should pass the whole frames instead of patches.##
#         #############################################################################
#         t_patch, _ = img_t
#         _, s_patch = img_s

        
#         t_patch = torch.tensor(t_patch, dtype=torch.float32).unsqueeze(0) / 255.0
#         s_patch = torch.tensor(s_patch, dtype=torch.float32).unsqueeze(0) / 255.0
#         gt_disp = torch.tensor(s["gt_disp"], dtype=torch.float32)

#         return t_patch, s_patch, gt_disp

## No Rsolution mapping is required ###
#TODO: adjust the location values returend by the dataloader to be pixel coords of the object in template image. 
class CLUSTLandmarkDataset(Dataset):
    def __init__(self, root_dir, template_size=127, search_size=255):
        """
        Args:
            root_dir: root path containing subfolders like ETH-01-1, MED-01-1, etc.
            template_size, search_size: patch dimensions in pixels
        """
        self.template_size = template_size
        self.search_size = search_size

        self.samples = []

        seq_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))])
        
        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            img_dir = os.path.join(seq_dir, "Data")
            ann_dir = os.path.join(seq_dir, "Annotation")
            if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
                continue
          
            # Load all frame images
            img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))])
            frame_ids = [int(os.path.splitext(f)[0]) for f in img_files]
            img_map = {fid: os.path.join(img_dir, f) for fid, f in zip(frame_ids, img_files)}
            
            # Each annotation file corresponds to one landmark target
            for ann_file in sorted(os.listdir(ann_dir)):
                if not ann_file.endswith('.txt'):
                    continue
                ann_path = os.path.join(ann_dir, ann_file)
                ann = np.loadtxt(ann_path, dtype=float)
                if ann.ndim == 1:  # handle single-line annotation files
                    ann = ann[None, :]
                ann = ann[np.argsort(ann[:, 0])]  # sort by frame id

                # Build template-search pairs for consecutive annotated frames
                for i in range(1, len(ann)):
                    
                    fid_prev, x_mm_prev, y_mm_prev = ann[i - 1] # preceeding frame id, x_mm, y_mm
                    fid_curr, x_mm_curr, y_mm_curr = ann[i] # current frame id, x_mm, y_mm

                    x_prev, y_prev = int(x_mm_prev), int(y_mm_prev) # convert mm to pixel coords (assuming 1mm = 1 pixel for now, adjust if needed)
                    x_curr, y_curr = int(x_mm_curr), int(y_mm_curr) # convert mm to pixel coords (assuming 1mm = 1 pixel for now, adjust if needed)

                    if fid_prev not in img_map or fid_curr not in img_map:
                        continue

                    disp = (x_curr - x_prev, y_curr - y_prev) # calculate displacement in pixel coords

                    self.samples.append({
                        "template_path": img_map[fid_prev],
                        "search_path": img_map[fid_curr],
                        "template_center": (x_prev, y_prev),
                        # "search_center": (x_prev, y_prev),  # crop around old pos
                        "search_center": (x_curr, y_curr),  # crop around old pos
                        "gt_disp": disp,
                        # "object_location": (x_prev, y_prev), # location of the object in the template image (pixel coords)
                        "seq_name": seq_name,
                    })

        print(f"Loaded {len(self.samples)} training pairs from {len(seq_dirs)} sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_t = cv2.imread(s["template_path"], cv2.IMREAD_GRAYSCALE)
        img_s = cv2.imread(s["search_path"], cv2.IMREAD_GRAYSCALE)

        ## uncomment the following lines if you want to crop patches around the target ##

        bbox_t = (s["template_center"][0], s["template_center"][1], 50, 50) 
        bbox_s = (s["search_center"][0], s["search_center"][1], 50, 50)
        t_patch, _ = siamfc_crop_and_resize(img_t, bbox_t)
        _, s_patch = siamfc_crop_and_resize(img_s, bbox_s)

        ####################################################################################
        ## if working with FNO, we should pass the whole frames instead of cropped frames.##
        ####################################################################################
        
        template_img = torch.tensor(t_patch, dtype=torch.float32).unsqueeze(0) / 255.0
        search_img = torch.tensor(s_patch, dtype=torch.float32).unsqueeze(0) / 255.0
        disp = torch.tensor(s["gt_disp"], dtype=torch.float32)
        object_location = torch.tensor(s["template_center"], dtype=torch.float32)

        # return template_img, search_img, object_location, disp
        return template_img, search_img, disp