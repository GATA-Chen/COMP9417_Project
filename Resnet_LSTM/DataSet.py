import json
import os
import warnings
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer

warnings.filterwarnings('ignore')


class ImageDataset(Dataset):
    def __init__(self, data_dirs, mode=0):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.eos_token = self.tokenizer.sep_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_dirs = data_dirs
        self.transform = [
            transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
            transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
        ][mode]
        self.image_files = []
        self.json_data = {}
        for i in data_dirs:
            image_files_dir = [os.path.join(i, f) for f in os.listdir(i) if f.endswith('.webp')]
            self.image_files.extend(image_files_dir)
            json_file = [f for f in os.listdir(i) if f.endswith('.json')]
            json_file = json_file[0]
            with open(os.path.join(i, json_file)) as f:
                json_data = json.load(f)
            for key in json_data.keys():
                self.json_data[key] = json_data[key]['p']

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        prompt = self.json_data[img_name]
        prompt = self.tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        input_ids = prompt['input_ids']
        attention_mask = prompt['attention_mask'].squeeze()
        return image, input_ids, attention_mask


def get_dataloader(data_location, batch_size):
    data_set = sorted([i for i in [os.path.join(data_location, item) for item in os.listdir(data_location)]
                       if os.path.isdir(i)])
    train_number = int(len(data_set) * 0.8)
    return [DataLoader(ImageDataset(data_set[0:train_number], mode=0), batch_size=batch_size, shuffle=True),
            DataLoader(ImageDataset(data_set[train_number:], mode=1), batch_size=batch_size, shuffle=True)]
