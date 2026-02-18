import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def get_subwindow(im, pos, model_sz, original_sz):
    """
    Extracts a square subwindow from the image.
    - pos: (cx, cy)
    - model_sz: output size in pixels (e.g., 127 or 255)
    - original_sz: the size of the crop in the original image (float)
    """
    if isinstance(pos, tuple):
        cx, cy = pos
    else:
        cx, cy = pos[0], pos[1]

    # If original_sz is a float -> round and pad
    sz = original_sz
    if isinstance(sz, float):
        sz = int(np.round(sz))

    # Coordinates in input image
    xs = np.floor(cx) + np.arange(sz) - np.floor(sz/2)
    ys = np.floor(cy) + np.arange(sz) - np.floor(sz/2)
    
    xs = xs.astype(int)
    ys = ys.astype(int)

    # Handle out-of-bound indexing (padding)
    pads = 0
    left_pad = max(0, -xs.min())
    top_pad = max(0, -ys.min())
    right_pad = max(0, xs.max() - (im.shape[1]-1))
    bottom_pad = max(0, ys.max() - (im.shape[0]-1))

    if any([left_pad, top_pad, right_pad, bottom_pad]):
        # pad image using average color
        avg_color = np.mean(im)
        im_padded = np.pad(im, 
                           ((top_pad, bottom_pad), (left_pad, right_pad)),
                           mode='constant', constant_values=avg_color)
    else:
        im_padded = im

    # Shift coordinates into padded image
    xs = xs + left_pad
    ys = ys + top_pad

    # Crop
    out = im_padded[ys[:, None], xs] # (sz, sz)

    # Resize to model size (127 or 255)
    out = cv2.resize(out.astype(np.float32), (model_sz, model_sz))
    return out

def siamfc_crop_and_resize(img, bbox, template_size=127, search_size=255, context_amount=0.1):
    """
    bbox = (cx, cy, w, h) (pixel units)
    Returns:
      template_patch (127x127)
      search_patch (255x255)
      z_crop_size (float) (the amount of original image used for template crop)
      x_crop_size (float) (the amount of original image used for search crop)
    """
    cx, cy, w, h = bbox

    # --------- 1. Compute context padded size ---------
    context = context_amount * (w + h)
    s_z = np.sqrt((w + context) * (h + context)) # template crop size
    
    scale_z = template_size / s_z

    # --------- 2. Search area enlarged by same factor ---------
    s_x = s_z * (search_size / template_size)

    # --------- 3. Extract patches ---------
    z_patch = get_subwindow(img, (cx, cy), template_size, s_z)
    x_patch = get_subwindow(img, (cx, cy), search_size, s_x)

    return z_patch, x_patch

def crop_patch(img, center, size):
    """Crop square patch centered at (x,y)"""
    x, y = map(int, center)
    half = size // 2
    H, W = img.shape
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(W, x + half), min(H, y + half)
    patch = np.zeros((size, size), dtype=img.dtype)
    patch_y1, patch_y2 = half - (y - y1), half + (y2 - y)
    patch_x1, patch_x2 = half - (x - x1), half + (x2 - x)
    patch[patch_y1:patch_y2, patch_x1:patch_x2] = img[y1:y2, x1:x2]
    return patch

import torch

def crop_patch_feat(feat_map: torch.Tensor, center, size: int):
    """
    Crop square patch centered at (x, y) from a 2D tensor.
    feat_map: (H, W) tensor
    center: (x, y)
    size: int, patch size
    """
    x, y = map(int, center)
    half = size // 2
    H, W = feat_map.shape[-2], feat_map.shape[-1]

    # bounds in source feature map
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(W, x + half), min(H, y + half)

    # allocate patch on same device/dtype
    patch = torch.zeros(size, size, dtype=feat_map.dtype, device=feat_map.device)

    # bounds in destination patch
    patch_y1, patch_y2 = half - (y - y1), half + (y2 - y)
    patch_x1, patch_x2 = half - (x - x1), half + (x2 - x)

    patch[patch_y1:patch_y2, patch_x1:patch_x2] = feat_map[y1:y2, x1:x2]
    return patch

def create_gaussian_response(disp, response_size, search_size, template_size, sigma, device):
    """
    disp: Bx2 tensor of (dx, dy) in pixels in search patch
    response_size: H=W of correlation map
    """
    B = disp.size(0)
    H = W = response_size
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    y_grid = y_grid.float()
    x_grid = x_grid.float()
    
    # Compute stride per pixel in response map
    stride = (search_size - template_size) / (H - 1)
    
    # Map displacement to response map coordinates
    gt_x = (W-1)/2 + disp[:,0]/stride
    gt_y = (H-1)/2 + disp[:,1]/stride
    gt_x = gt_x[:,None,None] # Bx1x1
    gt_y = gt_y[:,None,None]
    
    # Gaussian map
    response = torch.exp(-((x_grid[None,:,:]-gt_x)**2 + (y_grid[None,:,:]-gt_y)**2)/(2*sigma**2))
    response = response.unsqueeze(1) # Bx1xHxW
    return response

def create_ce_target(disp, response_size, search_size, template_size, device):
    """
    disp: Bx2 (dx, dy) displacement in pixel coords
    return: B (class indices)
    """
    B = disp.size(0)
    H = W = response_size

    stride = (search_size - template_size) / (H - 1)

    # GT peak in response-map coordinates
    gt_x = (W - 1)/2 + disp[:,0] / stride
    gt_y = (H - 1)/2 + disp[:,1] / stride

    # round to nearest grid cell
    gt_x = torch.round(gt_x).long().clamp(0, W-1)
    gt_y = torch.round(gt_y).long().clamp(0, H-1)

    # convert (y,x) to flat index: idx = y*W + x
    gt_index = gt_y * W + gt_x

    return gt_index.to(device) # B

def get_peak_coords(response_map, search_center, template_size, search_size):
    resp = response_map.squeeze().detach().cpu().numpy()
    r_y, r_x = np.unravel_index(np.argmax(resp), resp.shape)
    H, W = resp.shape
    cx = (W - 1) / 2
    cy = (H - 1) / 2
    stride = (search_size - template_size)/(H-1)
    offset_x = (r_x - cx)*stride
    offset_y = (r_y - cy)*stride
    x = search_center[0] + offset_x
    y = search_center[1] + offset_y
    return x, y

def get_peak_coords_new(response_map, search_center, template_size, search_size, up_factor=16):
    """
    response_map: torch tensor of shape [1, 1, H, W]
    search_center: (x_center, y_center)
    """
    # convert to numpy
    resp = response_map.squeeze().detach().cpu().numpy() # shape HxW
    H, W = resp.shape

    # Upsample (for smoother peak) ===
    resp_up = cv2.resize(resp, None, fx=up_factor, fy=up_factor,
                         interpolation=cv2.INTER_CUBIC)

    Hu, Wu = resp_up.shape

    # Find peak on upsampled map ===
    r_y, r_x = np.unravel_index(np.argmax(resp_up), resp_up.shape)

    # Convert back to displacement ===
    # original stride in image space
    stride = (search_size - template_size) / (H - 1)

    # shrink coordinates back to the coarse 17x17 grid location
    # but keep the fractional offset
    x_lowres = r_x / up_factor
    y_lowres = r_y / up_factor

    cx = (W - 1) / 2
    cy = (H - 1) / 2

    offset_x = (x_lowres - cx) * stride
    offset_y = (y_lowres - cy) * stride

    # Final image coordinates ===
    x = search_center[0] + offset_x
    y = search_center[1] + offset_y

    return x, y

def load_sequence_images(seq_folder):
    img_dir = os.path.join(seq_folder, "Data")
    img_files = sorted(
        [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".bmp", ".tif"))]
    )
    return img_files

def create_cosine_window(size, device):
    """
    size: int, response map size (17, 31, etc.)
    """
    h = np.hanning(size)
    w = np.hanning(size)
    window = np.outer(h, w)
    window = window / window.max()
    return torch.tensor(window, dtype=torch.float32, device=device) # HxW

def apply_cosine_window(response, window, eta=0.25):
    """
    response: Bx1xHxW (torch tensor)
    window: HxW (torch tensor)
    """
    return (1 - eta) * response + eta * window[None, None, :, :]

def visualize_tracking(img_files, trajectory, radius=4, pause_time=0.01):
    for i, (x, y) in enumerate(trajectory):
        plt.figure(figsize=(6,6))
        frame = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)

        # show frame
        if frame.ndim == 2:
            plt.imshow(frame, cmap='gray')
        else:
            plt.imshow(frame)
        
        # plot tracked location
        plt.scatter([x], [y], s=80, c='red', marker='o')

        plt.title(f"Frame {i}")
        plt.axis('off')
        plt.show()
        plt.pause(pause_time)
        plt.close()
