import torch
import cv2
from utils import (
    crop_patch, get_peak_coords, load_sequence_images, 
    create_cosine_window, apply_cosine_window, 
    siamfc_crop_and_resize, get_peak_coords_new
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def track_sequence(model, frames, init_coord, template_size=127, search_size=255, device=device):
    model.eval()
    prev_center = init_coord
    trajectory = [init_coord]
    with torch.no_grad():
        for i in range(1, len(frames)):
            template = crop_patch(frames[i-1], prev_center, template_size)
            search = crop_patch(frames[i], prev_center, search_size)
            t = torch.tensor(template, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)/255.
            s = torch.tensor(search, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)/255.
            resp = model(t, s)
            x, y = get_peak_coords(resp, prev_center, template_size, search_size)
            trajectory.append((x,y))
            prev_center = (x,y)
    return trajectory

def track_sequence_folder(
    model,
    seq_folder,
    init_coord,
    template_size=127,
    search_size=255,
    eta = 0.5,
    device=device):
    """
    seq_folder: path to sequence folder with 'images/'
    init_coord: (x, y) landmark coordinate in **pixels** on frame 0
    """
    # Load frames
    img_files = load_sequence_images(seq_folder)
    N = len(img_files)
    
    print(f"Tracking {N} frames from: {seq_folder}")
    model.to(device)
    model.eval()
    print(f"Initial coordinate: {init_coord}")
    prev_center = init_coord
    trajectory = [init_coord]

    with torch.no_grad():
        dummy_t = torch.zeros(1,1,template_size,template_size,device=device)
        dummy_s = torch.zeros(1,1,search_size,search_size,device=device)
        resp_dummy = model(dummy_t, dummy_s)
        H = resp_dummy.shape[-1]
        window = create_cosine_window(H, device=device)
    

    frame_prev = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)
    bbox_t = (init_coord[0], init_coord[1], 55, 55) 
    # keep the template fixed
    template_np, _ = siamfc_crop_and_resize(frame_prev, bbox_t)

    print("Starting tracking...")
    print(f'Frame 0: {trajectory[0]}   (Initial position)')
    with torch.no_grad():
        for i in range(1, N):
            # crop template from previous frame
            frame_curr = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)

            bbox_s = (prev_center[0], prev_center[1], 55, 55)
            _, search_np = siamfc_crop_and_resize(frame_curr, bbox_s)

            # convert to torch
            t = torch.tensor(template_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.
            s = torch.tensor(search_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.
            
            # # # forward pass
            resp = model(t, s)

            resp = apply_cosine_window(resp, window, eta=eta)

            # get peak in frame coordinates
            x_new, y_new = get_peak_coords_new(resp, prev_center, template_size, search_size, 8)
            # x_new_new, y_new_new = get_peak_coords_new(resp_new, prev_center, template_size, search_size, 8)
            print(f'Frame {i}: ({x_new:.2f}, {y_new:.2f})') 
                #   (new: {x_new_new:.2f}, {y_new_new:.2f})')
            trajectory.append((x_new, y_new))
            prev_center = (x_new, y_new)
            frame_prev = frame_curr

    return trajectory, img_files
