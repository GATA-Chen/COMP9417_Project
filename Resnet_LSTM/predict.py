import csv
import re
import os
import glob
import torch
import json

from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os.path import join
from torchvision.models import resnet50
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from model import Resnet_LSTM
from itertools import groupby
from difflib import SequenceMatcher

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


def remove_repeated_words(sentence):
    return ' '.join(word for word, _ in groupby(sentence.split()))


class ImageDataset(Dataset):
    def __init__(self, path, preprocess):
        self.images = []
        self.image_names = []
        self.preprocess = preprocess
        for file in glob.glob(join(path, '*')):
            filename = os.path.basename(file)
            if 'png' not in filename:
                continue
            image = Image.open(file).convert('RGB')

            self.images.append(image)
            self.image_names.append(filename)

    def __getitem__(self, item):
        image = self.images[item]
        image = self.preprocess(image)
        return image, self.image_names[item]

    def __len__(self) -> int:
        return len(self.images)


def load_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = 0
    resnet_model = resnet50(pretrained=True)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = Resnet_LSTM(resnet_model, gpt2_model, 1024)
    model.load_state_dict(torch.load(model_path, map_location='mps'))
    model.to(device)

    return model


def generate_and_save_predictions(model, dataloader):
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    with open('../DataSet_test/part-000001.json', 'r') as f:
        data = json.load(f)

    image = list()
    sentences = list()
    original = list()
    similarity = list()

    with torch.no_grad():
        for images, image_names in tqdm(dataloader):
            images = images.to(device)
            attention_mask = torch.ones(images.size(0), 200).to(device)
            output = model(images, attention_mask)
            predicted_ids = output.argmax(dim=-1)
            for i in range(predicted_ids.size(0)):
                tokens = predicted_ids[i].tolist()
                pre = tokenizer.decode(tokens).encode("ascii", errors="ignore").decode()
                pre = re.sub(",+", ",", pre)
                pre = remove_repeated_words(re.sub(" +", " ", pre))
                matcher = SequenceMatcher(None, pre, data[image_names[i]]["p"])
                image.append(image_names[i])
                sentences.append(pre)
                original.append(data[image_names[i]]["p"])
                similarity.append(matcher.quick_ratio() * 100)

    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Generated Sentence", "Original Sentence", "Similarity Score"])
        for i in range(len(image)):
            writer.writerow([image[i], sentences[i], original[i], "{:.2f}%".format(similarity[i])])


def main():
    model = load_model('checkModel.pth')

    dataset = ImageDataset('../DataSet_test/',
                           transforms.Compose([transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    generate_and_save_predictions(model, dataloader)


if __name__ == '__main__':
    main()
