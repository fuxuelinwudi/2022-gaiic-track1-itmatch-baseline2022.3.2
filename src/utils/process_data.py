# coding:utf-8

import json

coarse_train_path = '../../raw_data/train_coarse_sample.txt'  # 1000 - don't have key attr
fine_train_path = '../../raw_data/train_fine_sample.txt'  # 500 - have key attr


"""
process coarse train data
"""

with open(coarse_train_path, 'r', encoding='utf-8') as f:
    texts = []
    lines = f.readlines()
    for line_id, line in enumerate(lines):
        data = json.loads(line)
        """
        'img_name', 'title', 'key_attr', 'match', 'feature'
        """
        img_name = data['img_name']
        title = data['title']
        key_attr = data['key_attr']
        match = data['match']
        feature = data['feature']

        label = match['图文']

        sample = {}
        sample['text'] = title
        sample['label'] = label
        sample['feature'] = feature

        texts.append(sample)

out_coarse_train_path = '../../raw_data/train_coarse_sample.json'
with open(out_coarse_train_path, 'w', encoding='utf-8') as f:
    for line_id, text in enumerate(texts):
        f.write("%s\n" % text)


print('*' * 50)

"""
process fine train data

all key attr : 
    '裤门襟', '类别', '领型', '裤长', '裤型', '裙长', '穿着方式', 
    '闭合方式', '衣长', '袖长', '版型', '鞋帮高度'

加 '图文' ， 共 12 + 1 种
"""

label_list = ['裤门襟', '类别', '领型', '裤长', '裤型', '裙长',
              '穿着方式', '闭合方式', '衣长', '袖长', '版型',
              '鞋帮高度', '图文']
label2id = {l: i for i, l in enumerate(label_list)}

sort_label_list = sorted(label_list)
sort_label_list_save_path = '../../raw_data/sort_label_list.txt'
with open(sort_label_list_save_path, 'w', encoding='utf-8') as f:
    f.write(' '.join(sort_label_list))

with open(fine_train_path, 'r', encoding='utf-8') as f:
    texts = []
    lines = f.readlines()
    for line_id, line in enumerate(lines):
        data = json.loads(line)
        """
        'img_name', 'title', 'key_attr', 'match', 'feature'
        """
        img_name = data['img_name']
        title = data['title']
        key_attr = data['key_attr']
        match = data['match']
        feature = data['feature']

        # print('图片名称 : ', img_name)
        # print('商品标题 : ', title)
        # print('关键属性 : ', key_attr)
        # print('关键属性匹配情况 : ', match)

        matched_label = list(key_attr.keys())
        for i in label_list:
            if i not in matched_label:
                match[i] = 0

        sample, new_match, labels = {}, {}, []
        for i in sorted(match):
            new_match[i] = match[i]

        for i in new_match.values():
            labels.append(str(i))

        sample["text"] = title
        sample["label"] = labels
        sample["feature"] = feature

        texts.append(sample)

out_fine_train_path = '../../raw_data/train_fine_sample.json'
out_fine_dev_path = '../../raw_data/dev_fine_sample.json'
with open(out_fine_train_path, 'w', encoding='utf-8') as f:
    for line_id, text in enumerate(texts):
        if line_id < 400:
            text = json.dumps(text)
            f.write("%s\n" % text)
with open(out_fine_dev_path, 'w', encoding='utf-8') as f:
    for line_id, text in enumerate(texts):
        if 400 <= line_id < 500:
            text = json.dumps(text)
            f.write("%s\n" % text)

