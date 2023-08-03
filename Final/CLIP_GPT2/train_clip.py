from typing import Tuple
import clip
import os
import json
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from skimage import io
from PIL import Image
from torch import nn
from loguru import logger
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader


def image_encoder(dataset, mode, f3):
    model = clip.available_models()
    print(model)
    m = int(input("Enter the corresponding index of the above array to select the coding model: "))
    clip_model, preprocess = clip.load(model[m], device=device, jit=False)
    idx2embed = {}
    prompts = list()
    with tqdm(range(dataset.num_rows['train']), desc='Embedding and split prompts') as pbar:
        for i in pbar:
            # pbar.set_postfix({'The image with id %d is being encoded' % i})
            image = io.imread(str(dataset.data['train']['image'][0][1]))
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                clip_embed = clip_model.encode_image(image).cpu()
            idx2embed[i] = clip_embed
            prompts.append(str(dataset.data['train']['prompt'][i]))
            pbar.update()
        pbar.close()
    logger.info('num of image embedding and prompts:{}'.format(len(idx2embed)))
    if mode == "train":
        with open(f3, 'wb') as f:
            pickle.dump([prompts, idx2embed], f)
    else:
        with open(f3, 'wb') as f:
            pickle.dump([prompts, idx2embed], f)
    return prompts, idx2embed


class makeDataset(Dataset):
    def __init__(self, len_prefix, normalization, datasets, filename, f2):
        self.normalization = normalization
        self.prefix_length = len_prefix
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.embed_list = list()
        # data_path = r"DataSet"
        path = r'/kaggle/working/encoder_for_train_5k.pkl'
        if os.path.isfile(f2):
            with open(f2, 'rb') as f:
                self.embed_list, self.text_list, self.max_seq_len = pickle.load(f)
            logger.info('num of data:{}'.format(len(self.embed_list)))
        else:
            self.text_list = list()
            self.embed_list = list()
            max_seq_len = 0
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    prompts, idx2embd = pickle.load(f)
            else:
                prompts, idx2embd = image_encoder(datasets, "train")
            with tqdm(range(len(prompts)), desc='Embedding prompts') as pbar:
                for i in pbar:
                    text_encoder = torch.tensor(self.tokenizer.encode(prompts[i]), dtype=torch.int64)
                    self.text_list.append(text_encoder)
                    self.embed_list.append(idx2embd[i].squeeze(0).float())
                    max_seq_len = max(max_seq_len, self.text_list[-1].shape[0])
                    pbar.update()
                pbar.close()
            with open(f2, 'wb') as f:
                pickle.dump([self.embed_list, self.text_list, max_seq_len], f)
        all_len = torch.tensor([len(self.text_list[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def pad_tokens(self, item: int):
        tokens = self.text_list[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.text_list[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.text_list[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __len__(self) -> int:
        return len(self.text_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        embed = self.embed_list[index]
        tokens, mask = self.pad_tokens(index)
        if self.normalization:
            embed = embed.float()
            embed = embed / embed.norm(2, -1)
        return embed, tokens, mask


def evaluate(model, dev_loader):
    model.eval()
    #     logger.info("Running evaluation")
    eval_loss = 0.0
    with torch.no_grad():
        for data in dev_loader:
            embed_list, text_list, mask_list = data
            embed_list = embed_list.to(device).float()
            text_list = text_list.to(device, dtype=torch.int64)
            mask_list = mask_list.to(device)
            logits = model(embed_list, text_list, mask_list)
            logits = logits[:, model.prefix_len - 1: -1]
            #             shift_logits = logits[..., p.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = text_list.flatten()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), shift_labels, ignore_index=0)
            #             shift_logits = logits[..., model.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
            #             shift_labels = text_list.view(-1)
            #             loss = F.cross_entropy(shift_logits, shift_labels)
            loss = loss.mean()
            eval_loss += loss

    return eval_loss


def train(model, train_loader, dev_loader, optimizer, scheduler, f3):
    model.train()
    logger.info("start training")
    loss_tt = list()
    for epoch in range(p.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            model.zero_grad()
            step = epoch * len(train_loader) + batch_idx + 1
            embed_list, text_list, mask_list = data
            embed_list = embed_list.to(device).float()
            text_list = text_list.to(device, dtype=torch.int64)
            mask_list = mask_list.to(device)
            logits = model(embed_list, text_list, mask_list)
            logits = logits[:, model.prefix_len - 1: -1]
            #             # shift_logits = logits[..., p.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = text_list.flatten()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), shift_labels, ignore_index=0)
            #             shift_logits = logits[..., model.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
            #             shift_labels = text_list.view(-1)
            #             loss = F.cross_entropy(shift_logits, shift_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        #             if step % 10 == 0:
        dev_loss = evaluate(model, dev_loader)
        #             logger.info('test loss at step {} is {}'.format(step, dev_loss.item()))
        model.train()
        #             logger.info('train loss at step {} is {}'.format(step, loss))
        loss_tt.append((step, loss.item(), dev_loss.item()))
        logger.info('test loss at epoch {} is {}'.format(epoch + 1, dev_loss.item()))
        logger.info('train loss at epoch {} is {}'.format(epoch + 1, loss))
        if (epoch + 1) % 2 == 0:
            logger.info('saving checkpoint at epoch {}'.format(epoch + 1))
            torch.save(model.state_dict(), r"/kaggle/working/checkModel" + str(epoch + 1) + " .pth")
    with open(f3, 'w') as f:
        for i in loss_tt:
            f.write(str(i[0]) + '\t' + str(i[1]) + '\t' + str(i[2]) + '\n')


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    def __init__(self, prefix_len, clip_size,
                 finetune_gpt2, constant_len=10):
        super(ClipCaptionModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        #         logger.info('succeed to load pretrain gpt2 model')
        self.prefix_size = self.gpt2.transformer.wte.weight.shape[1]
        self.prefix_len = prefix_len
        self.clip_project = MLP((clip_size, (self.prefix_size * prefix_len) // 2, self.prefix_size * prefix_len))
        self.finetune_gpt2 = finetune_gpt2

    def forward(self, clip_embeds, caption_ids, mask):
        # caption_inputs_embeds:[bs, caption_len, prefix_size]
        caption_embeds = self.gpt2.transformer.wte(caption_ids)
        # prefix_embeds:[bs, prefix_len, prefix_size]
        prefix_embeds = self.clip_project(clip_embeds).view(-1, self.prefix_len, self.prefix_size)
        # embedding_cat:[bs, prefix_len+caption_len, prefix_size]
        embedding_cat = torch.cat((prefix_embeds, caption_embeds), dim=1)
        out = self.gpt2(inputs_embeds=embedding_cat, attention_mask=mask)
        # logits:[bs, prefix_len+caption_len, prefix_size]
        logits = out.logits
        return logits

    def parameters(self, recurse: bool = True):
        if self.finetune_gpt2:
            return super(ClipCaptionModel, self).parameters()
        else:
            return self.clip_project.parameters()

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.finetune_gpt2:
            self.gpt2.eval()
        return self


def main(filename, f2, f3):
    torch.cuda.empty_cache()
    model = ClipCaptionModel(p.prefix_len, p.img_size, p.tune).to(device)
    dataset = makeDataset(p.prefix_len, p.isNormalize, None, filename, f2)
    test_size = int(p.test_split * len(dataset))
    train_dataset, dev_dataset = torch.utils.data.random_split(dataset,
                                                               [len(dataset) - test_size,
                                                                test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True)
    dev_loader = 0
    dev_loader = DataLoader(dev_dataset, batch_size=p.batch_size, shuffle=True)
    t_total = len(train_dataloader) * p.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=p.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
    )

    train(model, train_dataloader, dev_loader, optimizer, scheduler, f3)


p = {
    "prefix_len": 10,
    "img_size": 512,
    "tune": True,
    "isNormalize": True,
    "batch_size": 32,
    "test_split": 0.2,
    "epochs": 40,
    "lr": 2e-5
}


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


p = DotDict(p)

torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
main(r'/kaggle/input/models/train_for_dataset_10k.pkl', r'/kaggle/working/encoder_for_train_10k.pkl', r'loss_10k.txt')
