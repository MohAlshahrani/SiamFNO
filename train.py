from torch.utils.data import DataLoader
from dataset import CLUSTLandmarkDataset
from models import SiamFCAlexNet, SiamFNO
from engine import train_siamfno

if __name__ == "__main__":

    dataset = CLUSTLandmarkDataset("C:\\Users\\MohAl\\repos\\SiamFNO\\Training\\2D")

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = SiamFNO()
    
    train_siamfno(model, loader, epochs=50, checkpoint_dir = "C:\\Users\\MohAl\\repos\\SiamFNO\\checkpoints")
