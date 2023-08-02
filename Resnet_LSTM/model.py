import warnings
import torch
import transformers
from tqdm import tqdm
from torch import nn
from torchvision import models
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import *
from DataSet import get_dataloader

warnings.filterwarnings('ignore')


class Resnet_LSTM(nn.Module):
    def __init__(self, resnet_model, gpt2_model, embedding_dim):
        super().__init__()
        self.resnet = resnet_model
        self.fc = nn.Linear(1000, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, gpt2_model.config.hidden_size, batch_first=True)
        self.gpt2 = gpt2_model

    def forward(self, images, attention_mask):
        resnet_output = self.resnet(images)
        resnet_output = self.fc(resnet_output)
        lstm_output, _ = self.lstm(resnet_output.unsqueeze(1))
        lstm_output = lstm_output.expand(-1, 512, -1)
        gpt2_output = self.gpt2(inputs_embeds=lstm_output, attention_mask=attention_mask)
        return gpt2_output.logits


def train(model, dataloader, optimizer, device, gpt2_model, scheduler):
    model.train()
    total_loss = 0
    for images, input_ids, attention_mask in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        optimizer.zero_grad()
        output = model(images, attention_mask)
        loss = nn.CrossEntropyLoss()(output.view(-1, gpt2_model.config.vocab_size), input_ids.view(-1),
                                     ignore_index=0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, device, gpt2_model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, input_ids, attention_mask in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = model(images, attention_mask)
            loss = nn.CrossEntropyLoss()(output.view(-1, gpt2_model.config.vocab_size), input_ids.view(-1),
                                         ignore_index=0)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    resnet_model = models.resnet50(pretrained=True)

    train_dataloader, valid_dataloader = get_dataloader()

    model = Resnet_LSTM(resnet_model, gpt2_model, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = 8000, num_training_steps = len(train_dataloader) * EPOCH
    )

    for epoch in range(EPOCH):
        train_loss = train(model, train_dataloader, optimizer, device, gpt2_model, scheduler)
        valid_loss = validate(model, valid_dataloader, device, gpt2_model)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')

        print('Saving checkpoint at epoch {}'.format(epoch + 1))
        torch.save(model.state_dict(), "checkModel.pth")


if __name__ == '__main__':
    train_model()
