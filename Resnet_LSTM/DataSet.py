import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import GPT2Tokenizer
from datasets import load_dataset
from config import *


warnings.filterwarnings('ignore')


class ImageDataset(Dataset):
    def __init__(self, mode=0):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_set = TRAIN_DATA_SET if mode == 0 else VALID_DATA_SET
        self.dataset = load_dataset(DATA_PATH, self.data_set)['train']
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx]['image'])

        prompt = self.dataset[idx]['prompt']
        prompt = self.tokenizer.encode_plus(prompt, return_tensors='pt', padding='max_length', max_length=512,
                                            truncation=True)
        return image, prompt['input_ids'], torch.tensor(prompt['attention_mask'])


def get_dataloader():
    return [DataLoader(ImageDataset(0), batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(ImageDataset(1), batch_size=BATCH_SIZE, shuffle=True)]
