import argparse
from training.train import train
from models.unet_model import UNet
from training.dataset import get_dataloader
from models.alignment import align_face
import torch
from torchvision.utils import save_image  # Import save_image function
from models.sampling import sample  # Import the sample function

def main():
    parser = argparse.ArgumentParser(description="Makeup Transfer with Diffusion Model")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(T=1000, num_labels=1, ch=64, ch_mult=[1, 2, 4, 8], num_res_blocks=2, dropout=0.1).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    if args.mode == 'train':
        dataloader = get_dataloader(args.data_dir, args.batch_size)
        train(model, dataloader, args.epochs, args.lr, device)
    elif args.mode == 'test':
        test_dataloader = get_dataloader(args.data_dir, args.batch_size, mode='test')
        test(model, test_dataloader, device)

def test(model, test_dataloader, device):
    model.eval()
    for i, (face, makeup) in enumerate(test_dataloader):
        face = face.to(device)
        makeup = makeup.to(device)

        with torch.no_grad():
            generated = sample(model, makeup, device)

        # Save or display results
        save_image(generated, f'results/generated_{i}.png', normalize=True)
        save_image(torch.cat([face, makeup, generated], dim=3), f'results/comparison_{i}.png', normalize=True)

# Update main.py to include test mode
# Inside the main function in main.py, add:

if __name__ == '__main__':
    main()
