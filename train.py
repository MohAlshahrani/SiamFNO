from torch.utils.data import DataLoader
from dataset import CLUSTLandmarkDataset
from models import SiamFCAlexNet
from engine import train_siamfc

# ---------------------------
# Example usage for training
# ---------------------------
resolution_map = {
    "ETH-01-1": 0.4,
    "ETH-01-2": 0.41,
    "ETH-02-1": 0.42,
    "ETH-03-1": 0.28,
    "ETH-04-1": 0.4,
}

if __name__ == "__main__":
    dataset = CLUSTLandmarkDataset("/content/drive/MyDrive/Training/2D/", resolution_map)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # model = SiamFC()
    model = SiamFCAlexNet()
    
    train_siamfc(model, loader, epochs=50, checkpoint_dir = "/content/drive/MyDrive/SiamFC/checkpoints")
