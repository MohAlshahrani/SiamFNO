from torch.utils.data import DataLoader
from dataset import CLUSTLandmarkDataset
from models import SiamFCAlexNet, SiamFNO
from engine import train_siamfno

# ---------------------------
# Example usage for training
# ---------------------------
# resolution_map = {
#     "ETH-01-1": 0.4,
#     "ETH-01-2": 0.41,
#     "ETH-02-1": 0.42,
#     "ETH-03-1": 0.28,
#     "ETH-04-1": 0.4,
# }

if __name__ == "__main__":
    dataset = CLUSTLandmarkDataset("C:\\Users\\MohAl\\repos\\SiamFNO\\Training\\2D")
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # model = SiamFC()
    model = SiamFNO()
    
    train_siamfno(model, loader, epochs=50, checkpoint_dir = "C:\\Users\\MohAl\\repos\\SiamFNO\\checkpoints")
