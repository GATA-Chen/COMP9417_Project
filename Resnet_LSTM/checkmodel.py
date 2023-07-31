from torchvision.transforms import transforms
import os
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os.path import join
import PIL.Image
import torch
from torchvision.models import resnet50
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from model import Resnet_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


class ImageDataset(Dataset):
    def __init__(self, path, preprocess):
        self.images = []
        self.image_names = []
        self.preprocess = preprocess
        for file in glob.glob(join(path, '*')):
            image = PIL.Image.open(file).convert('RGB')
            filename = os.path.basename(file)

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
    resnet_model = resnet50(pretrained=True)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = Resnet_LSTM(resnet_model, gpt2_model, 1024)
    model.load_state_dict(torch.load(model_path, map_location='mps'))
    model.to(device)

    return model


def generate_and_save_predictions(model, dataloader, output_file):
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    with open(output_file, 'w') as f:
        with torch.no_grad():
            for images, image_names in tqdm(dataloader):
                images = images.to(device)
                attention_mask = torch.ones(images.size(0), 512).to(device)
                output = model(images, attention_mask)
                predicted_ids = output.argmax(dim=-1)
                for i in range(predicted_ids.size(0)):
                    tokens = predicted_ids[i].tolist()
                    sentence = tokenizer.decode(tokens)
                    f.write(f'{image_names[i]}: {sentence}\n')


def main():
    model = load_model('checkModel.pth')

    dataset = ImageDataset('Test',
                           transforms.Compose([transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    generate_and_save_predictions(model, dataloader, '1.txt')


if __name__ == '__main__':
    main()
