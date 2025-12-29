import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NavigationDataset(Dataset):
    def __init__(self, root_dir, split='data'):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.split_dir, 'images')
        self.annot_dir = os.path.join(self.split_dir, 'annotations')
        self.annot_files = sorted([f for f in os.listdir(self.annot_dir) if f.endswith('.json')])
        
        # Simple Vocabulary
        self.vocab = {"<pad>": 0, "go": 1, "to": 2, "the": 3, "red": 4, "green": 5, "blue": 6, 
                      "circle": 7, "square": 8, "triangle": 9}
        self.max_len = 7

        self.transform = transforms.Compose([transforms.ToTensor()])

    def text_to_tensor(self, text):
        tokens = text.lower().replace('.', '').split()
        indices = [self.vocab.get(t, 0) for t in tokens]
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices[:self.max_len], dtype=torch.long)

    def __len__(self):
        return len(self.annot_files)

    def __getitem__(self, idx):
        annot_path = os.path.join(self.annot_dir, self.annot_files[idx])
        with open(annot_path, 'r') as f:
            data = json.load(f)

        img_path = os.path.join(self.image_dir, data['image_file'])
        image = self.transform(Image.open(img_path).convert('RGB'))
        text_tensor = self.text_to_tensor(data['text'])

        if 'path' in data:
            path = torch.tensor(data['path'], dtype=torch.float32) / 128.0
        else:
            path = torch.zeros((10, 2), dtype=torch.float32)

        return image, text_tensor, path

def get_dataloader(root_dir, split='data', batch_size=32, shuffle=True):
    return DataLoader(NavigationDataset(root_dir, split), batch_size=batch_size, shuffle=shuffle)