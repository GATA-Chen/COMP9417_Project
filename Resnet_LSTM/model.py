import torch
import torch.nn as nn
from torchvision.models import resnet50
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from DataSet import get_dataloader

EPOCH = 10
BATCH_SIZE = 4
DATA_PATH = '../DataSet'


class Resnet_LSTM(nn.Module):
    def __init__(self, resnet_model, bert_model):
        super().__init__()
        self.vit = resnet_model
        self.fc = nn.Linear(1000, 512)
        self.lstm = nn.LSTM(512, bert_model.config.hidden_size, batch_first=True)
        self.bert = bert_model

    def forward(self, images, attention_mask):
        vit_output = self.vit(images)
        vit_output = self.fc(vit_output)
        lstm_output, _ = self.lstm(vit_output.unsqueeze(1))
        lstm_output = lstm_output.expand(-1, 512, -1)
        bert_output = self.bert(inputs_embeds=lstm_output, attention_mask=attention_mask)
        return bert_output.logits


def train(model, dataloader, optimizer, device, bert_model):
    model.train()
    total_loss = 0
    for images, input_ids, attention_mask in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        optimizer.zero_grad()
        output = model(images, attention_mask)
        loss = nn.CrossEntropyLoss()(output.view(-1, bert_model.config.vocab_size), input_ids.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, device, bert_model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, input_ids, attention_mask in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = model(images, attention_mask)
            loss = nn.CrossEntropyLoss()(output.view(-1, bert_model.config.vocab_size), input_ids.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    resnet_model = resnet50(pretrained=True)
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    trainDataloader, validDataloader = get_dataloader(DATA_PATH, BATCH_SIZE)

    model = Resnet_LSTM(resnet_model, bert_model).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(EPOCH):
        train_loss = train(model, trainDataloader, optimizer, device, bert_model)
        valid_loss = validate(model, validDataloader, device, bert_model)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')

        if epoch % 2 == 1:
            print('saving checkpoint at epoch {}'.format(epoch + 1))
            torch.save(model.state_dict(), r"checkModel.pth")


if __name__ == '__main__':
    train_model()
