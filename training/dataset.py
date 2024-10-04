import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from models.alignment import align_face


class MakeupDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.face_dir = os.path.join(data_dir, 'face')
        self.makeup_dir = os.path.join(data_dir, 'makeup')
        self.image_names = os.listdir(self.face_dir)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        face = Image.open(os.path.join(self.face_dir, image_name)).convert('RGB')
        makeup = Image.open(os.path.join(self.makeup_dir, image_name)).convert('RGB')

        face = align_face(face)
        makeup = align_face(makeup)

        face = self.transform(face)
        makeup = self.transform(makeup)

        return face, makeup


def get_dataloader(data_dir, batch_size):
    dataset = MakeupDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)