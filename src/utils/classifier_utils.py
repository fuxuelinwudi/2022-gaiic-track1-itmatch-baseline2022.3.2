# coding:utf-8

import re
import os
import torch
import pickle
import random
import shutil
import scipy.stats
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from src.model.models import BertForSequenceClassification


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(42)


def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


def save_model(model, tokenizer, saving_path):
    os.makedirs(saving_path, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(saving_path)
    tokenizer.save_vocabulary(saving_path)


def batch2cuda(args, batch):
    return {item: value.to(args.device) for item, value in list(batch.items())}


def build_bert_inputs(inputs, text, label_list, feature, tokenizer):
    inputs_dict = tokenizer.encode_plus(text, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)

    input_ids = inputs_dict['input_ids']
    token_type_ids = inputs_dict['token_type_ids']
    attention_mask = inputs_dict['attention_mask']

    inputs['input_ids'].append(input_ids)
    inputs['token_type_ids'].append(token_type_ids)
    inputs['attention_mask'].append(attention_mask)
    inputs['feature'].append(feature)
    for i in range(1, 13 + 1):
        inputs[f'label{i}'].append(label_list[i - 1])


class TrackOneDataset(Dataset):
    def __init__(self, data_dict):
        super(TrackOneDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index],
            self.data_dict['feature'][index],
            self.data_dict['label1'][index],
            self.data_dict['label2'][index],
            self.data_dict['label3'][index],
            self.data_dict['label4'][index],
            self.data_dict['label5'][index],
            self.data_dict['label6'][index],
            self.data_dict['label7'][index],
            self.data_dict['label8'][index],
            self.data_dict['label9'][index],
            self.data_dict['label10'][index],
            self.data_dict['label11'][index],
            self.data_dict['label12'][index],
            self.data_dict['label13'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class TrackOneCollator:
    def __init__(self, max_seq_len, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list, attention_mask_list, feature_list,
                         label1_list, label2_list, label3_list, label4_list, label5_list, label6_list,
                         label7_list, label8_list, label9_list, label10_list, label11_list, label12_list, label13_list,
                         max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])

            # pad
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)

            # cut
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)

        label1 = torch.tensor(label1_list, dtype=torch.long)
        label2 = torch.tensor(label2_list, dtype=torch.long)
        label3 = torch.tensor(label3_list, dtype=torch.long)
        label4 = torch.tensor(label4_list, dtype=torch.long)
        label5 = torch.tensor(label5_list, dtype=torch.long)
        label6 = torch.tensor(label6_list, dtype=torch.long)
        label7 = torch.tensor(label7_list, dtype=torch.long)
        label8 = torch.tensor(label8_list, dtype=torch.long)
        label9 = torch.tensor(label9_list, dtype=torch.long)
        label10 = torch.tensor(label10_list, dtype=torch.long)
        label11 = torch.tensor(label11_list, dtype=torch.long)
        label12 = torch.tensor(label12_list, dtype=torch.long)
        label13 = torch.tensor(label13_list, dtype=torch.long)

        feature = torch.tensor(feature_list, dtype=torch.float)

        return input_ids, token_type_ids, attention_mask, feature, \
               label1, label2, label3, label4, label5, label6, \
               label7, label8, label9, label10, label11, label12, label13

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list, feature_list, \
        label1_list, label2_list, label3_list, label4_list, label5_list, label6_list, \
        label7_list, label8_list, label9_list, label10_list, label11_list, label12_list, \
        label13_list = list(zip(*examples))

        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask, feature, \
        label1, label2, label3, label4, label5, label6, label7, label8, label9, label10, label11, label12, label13 = \
            self.pad_and_truncate(input_ids_list, token_type_ids_list, attention_mask_list, feature_list,
                                  label1_list, label2_list, label3_list, label4_list, label5_list, label6_list,
                                  label7_list, label8_list, label9_list, label10_list, label11_list, label12_list,
                                  label13_list, max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'feature': feature,
            'label1': label1,
            'label2': label2,
            'label3': label3,
            'label4': label4,
            'label5': label5,
            'label6': label6,
            'label7': label7,
            'label8': label8,
            'label9': label9,
            'label10': label10,
            'label11': label11,
            'label12': label12,
            'label13': label13
        }

        return data_dict


def load_data(args, tokenizer):
    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    dev_cache_pkl_path = os.path.join(args.data_cache_path, 'dev.pkl')

    with open(train_cache_pkl_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(dev_cache_pkl_path, 'rb') as f:
        dev_data = pickle.load(f)

    collate_fn = TrackOneCollator(args.max_seq_len, tokenizer)

    train_dataset = TrackOneDataset(train_data)
    dev_dataset = TrackOneDataset(dev_data)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_dataloader, dev_dataloader


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class FGM:
    def __init__(self, args, model):
        self.model = model
        self.backup = {}
        self.emb_name = args.emb_name
        self.epsilon = args.epsilon

    def attack(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, args, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = args.epsilon
        self.emb_name = args.emb_name
        self.alpha = args.alpha

    def attack(self, is_first_attack=False):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


def build_model_and_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    return tokenizer, model


def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    bert_param_optimizer = list(model.bert.named_parameters())
    other_param_optimizer = list(model.classifier1.named_parameters()) + list(model.classifier2.named_parameters()) + \
                            list(model.classifier3.named_parameters()) + list(model.classifier4.named_parameters()) + \
                            list(model.classifier5.named_parameters()) + list(model.classifier6.named_parameters()) + \
                            list(model.classifier7.named_parameters()) + list(model.classifier8.named_parameters()) + \
                            list(model.classifier9.named_parameters()) + list(model.classifier10.named_parameters()) +\
                            list(model.classifier11.named_parameters()) + list(model.classifier12.named_parameters()) + \
                            list(model.classifier13.named_parameters()) + list(model.fuse_layer.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},

        {'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay, 'lr': args.other_learning_rate},
        {'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.use_lookahead:
        optimizer = Lookahead(optimizer, 5, 1)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)

    return optimizer, scheduler


def evaluation(args, val_dataloader, model):
    metric = {}
    preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, preds13, \
    label1, label2, label3, label4, label5, label6, label7, label8, label9, label10, label11, label12, label13 = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    val_loss = 0.

    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            batch_cuda = batch2cuda(args, batch)
            loss, logits1, logits2, logits3, logits4, logits5, logits6, logits7, logits8, \
            logits9, logits10, logits11, logits12, logits13 = model(**batch_cuda)[:14]
            val_loss += loss.item()

            preds1.extend([i for i in torch.argmax(torch.softmax(logits1, -1), 1).cpu().numpy().tolist()])
            preds2.extend([i for i in torch.argmax(torch.softmax(logits2, -1), 1).cpu().numpy().tolist()])
            preds3.extend([i for i in torch.argmax(torch.softmax(logits3, -1), 1).cpu().numpy().tolist()])
            preds4.extend([i for i in torch.argmax(torch.softmax(logits4, -1), 1).cpu().numpy().tolist()])
            preds5.extend([i for i in torch.argmax(torch.softmax(logits5, -1), 1).cpu().numpy().tolist()])
            preds6.extend([i for i in torch.argmax(torch.softmax(logits6, -1), 1).cpu().numpy().tolist()])
            preds7.extend([i for i in torch.argmax(torch.softmax(logits7, -1), 1).cpu().numpy().tolist()])
            preds8.extend([i for i in torch.argmax(torch.softmax(logits8, -1), 1).cpu().numpy().tolist()])
            preds9.extend([i for i in torch.argmax(torch.softmax(logits9, -1), 1).cpu().numpy().tolist()])
            preds10.extend([i for i in torch.argmax(torch.softmax(logits10, -1), 1).cpu().numpy().tolist()])
            preds11.extend([i for i in torch.argmax(torch.softmax(logits11, -1), 1).cpu().numpy().tolist()])
            preds12.extend([i for i in torch.argmax(torch.softmax(logits12, -1), 1).cpu().numpy().tolist()])
            preds13.extend([i for i in torch.argmax(torch.softmax(logits13, -1), 1).cpu().numpy().tolist()])

            label1.extend([i for i in batch_cuda['label1'].cpu().numpy().tolist()])
            label2.extend([i for i in batch_cuda['label2'].cpu().numpy().tolist()])
            label3.extend([i for i in batch_cuda['label3'].cpu().numpy().tolist()])
            label4.extend([i for i in batch_cuda['label4'].cpu().numpy().tolist()])
            label5.extend([i for i in batch_cuda['label5'].cpu().numpy().tolist()])
            label6.extend([i for i in batch_cuda['label6'].cpu().numpy().tolist()])
            label7.extend([i for i in batch_cuda['label7'].cpu().numpy().tolist()])
            label8.extend([i for i in batch_cuda['label8'].cpu().numpy().tolist()])
            label9.extend([i for i in batch_cuda['label9'].cpu().numpy().tolist()])
            label10.extend([i for i in batch_cuda['label10'].cpu().numpy().tolist()])
            label11.extend([i for i in batch_cuda['label11'].cpu().numpy().tolist()])
            label12.extend([i for i in batch_cuda['label12'].cpu().numpy().tolist()])
            label13.extend([i for i in batch_cuda['label13'].cpu().numpy().tolist()])

    avg_val_loss = val_loss / len(val_dataloader)

    acc1 = accuracy_score(y_true=label1, y_pred=preds1)
    acc2 = accuracy_score(y_true=label2, y_pred=preds2)
    acc3 = accuracy_score(y_true=label3, y_pred=preds3)
    acc4 = accuracy_score(y_true=label4, y_pred=preds4)
    acc5 = accuracy_score(y_true=label5, y_pred=preds5)
    acc6 = accuracy_score(y_true=label6, y_pred=preds6)
    acc7 = accuracy_score(y_true=label7, y_pred=preds7)
    acc8 = accuracy_score(y_true=label8, y_pred=preds8)
    acc9 = accuracy_score(y_true=label9, y_pred=preds9)
    acc10 = accuracy_score(y_true=label10, y_pred=preds10)
    acc11 = accuracy_score(y_true=label11, y_pred=preds11)
    acc12 = accuracy_score(y_true=label12, y_pred=preds12)
    acc13 = accuracy_score(y_true=label13, y_pred=preds13)

    avg_val_loss, acc1, acc2, acc3, acc4, acc5, acc6, \
    acc7, acc8, acc9, acc10, acc11, acc12, acc13 = round(avg_val_loss, 4), round(acc1, 4), round(acc2, 4), \
                                                   round(acc3, 4), round(acc4, 4), round(acc5, 4), round(acc6, 4), \
                                                   round(acc7, 4), round(acc8, 4), round(acc9, 4), round(acc10, 4), \
                                                   round(acc11, 4), round(acc12, 4), round(acc13, 4)

    metric['acc1'], metric['acc2'], metric['acc3'], metric['acc4'], metric['acc5'], metric['acc6'], \
    metric['acc7'], metric['acc8'], metric['acc9'], metric['acc10'], metric['acc11'], metric['acc12'],\
    metric['acc13'], metric['avg_val_loss'] = acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, \
                                              acc9, acc10, acc11, acc12, acc13, avg_val_loss

    return metric


def make_dirs(path_list):
    for i in path_list:
        os.makedirs(os.path.dirname(i), exist_ok=True)
