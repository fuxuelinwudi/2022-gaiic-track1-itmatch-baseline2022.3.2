# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class FuseLayer(nn.Module):
    def __init__(self, image_hidden_size, text_hidden_size):
        super().__init__()

        self.image_hidden_size = image_hidden_size
        self.text_hidden_size = text_hidden_size
        self.highway1 = nn.Sequential(
            nn.Linear(self.image_hidden_size, self.text_hidden_size),
            nn.Dropout(0.1)
        )
        self.highway2 = nn.Sequential(
            nn.Linear(self.text_hidden_size * 2, self.text_hidden_size),
            nn.Dropout(0.1)
        )

    def forward(self, text_feature, image_feature):

        image_feature = self.highway1(image_feature)
        concat_feature = torch.cat([image_feature, text_feature], 1)

        fuse_feature = self.highway2(concat_feature)

        return fuse_feature


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Sequential(
            nn.Dropout(0.1)
        )
        self.fuse_layer = FuseLayer(2048, 768)
        self.classifier1 = nn.Linear(config.hidden_size, 2)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.classifier3 = nn.Linear(config.hidden_size, 2)
        self.classifier4 = nn.Linear(config.hidden_size, 2)
        self.classifier5 = nn.Linear(config.hidden_size, 2)
        self.classifier6 = nn.Linear(config.hidden_size, 2)
        self.classifier7 = nn.Linear(config.hidden_size, 2)
        self.classifier8 = nn.Linear(config.hidden_size, 2)
        self.classifier9 = nn.Linear(config.hidden_size, 2)
        self.classifier10 = nn.Linear(config.hidden_size, 2)
        self.classifier11 = nn.Linear(config.hidden_size, 2)
        self.classifier12 = nn.Linear(config.hidden_size, 2)
        self.classifier13 = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, feature=None,
                label1=None, label2=None, label3=None, label4=None, label5=None, label6=None, label7=None,
                label8=None, label9=None, label10=None, label11=None, label12=None, label13=None):

        sequence_out, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        pooled_output = self.dropout(pooled_output)
        last_feature = self.fuse_layer(pooled_output, feature)

        logits1 = self.classifier1(last_feature)
        logits2 = self.classifier2(last_feature)
        logits3 = self.classifier3(last_feature)
        logits4 = self.classifier4(last_feature)
        logits5 = self.classifier5(last_feature)
        logits6 = self.classifier6(last_feature)
        logits7 = self.classifier7(last_feature)
        logits8 = self.classifier8(last_feature)
        logits9 = self.classifier9(last_feature)
        logits10 = self.classifier10(last_feature)
        logits11 = self.classifier11(last_feature)
        logits12 = self.classifier12(last_feature)
        logits13 = self.classifier13(last_feature)

        outputs = (logits1,) + (logits2,) + (logits3,) + (logits4,) + (logits5,) + (logits6,) \
                  + (logits7,) + (logits8,) + (logits9,) + (logits10,) + (logits11,) + (logits12,) + (logits13,) + \
                  (last_feature,)
        if label1 is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss1 = loss_fct(logits1.view(-1, self.num_labels), label1.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.num_labels), label2.view(-1))
            loss3 = loss_fct(logits3.view(-1, self.num_labels), label3.view(-1))
            loss4 = loss_fct(logits4.view(-1, self.num_labels), label4.view(-1))
            loss5 = loss_fct(logits5.view(-1, self.num_labels), label5.view(-1))
            loss6 = loss_fct(logits6.view(-1, self.num_labels), label6.view(-1))
            loss7 = loss_fct(logits7.view(-1, self.num_labels), label7.view(-1))
            loss8 = loss_fct(logits8.view(-1, self.num_labels), label8.view(-1))
            loss9 = loss_fct(logits9.view(-1, self.num_labels), label9.view(-1))
            loss10 = loss_fct(logits10.view(-1, self.num_labels), label10.view(-1))
            loss11 = loss_fct(logits11.view(-1, self.num_labels), label11.view(-1))
            loss12 = loss_fct(logits12.view(-1, self.num_labels), label12.view(-1))
            loss13 = loss_fct(logits13.view(-1, self.num_labels), label13.view(-1))

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + \
                   loss9 + loss10 + loss11 + loss12 + loss13

            outputs = (loss,) + outputs

        return outputs
