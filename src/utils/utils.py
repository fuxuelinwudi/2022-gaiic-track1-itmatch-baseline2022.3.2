import csv
import hashlib
import json
import os
import pickle
import time
from datetime import timedelta, datetime
from glob import iglob
from mmap import mmap, ACCESS_READ

import numpy as np
import yaml


def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


def load_big_file(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        m = mmap(f.fileno(), 0, access=ACCESS_READ)
        tmp = 0
        for i, char in enumerate(m):
            if char == b"\n":
                yield m[tmp:i + 1].decode()
                tmp = i + 1


def load_file(fp: str, sep: str = None, name_tuple=None):
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if sep:
            if name_tuple:
                return map(name_tuple._make, [line.strip().split(sep) for line in lines])
            else:
                return [line.strip().split(sep) for line in lines]
        else:
            return lines


def load_json_file(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(line.strip(), encoding="utf-8") for line in f.readlines()]


def save_json_file(list_data, fp):
    with open(fp, "w", encoding="utf-8") as f:
        for data in list_data:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write("{}\n".format(json_str))
        f.flush()


def load_csv(fp, is_tsv: bool = False):
    dialect = 'excel-tab' if is_tsv else 'excel'
    with open(fp, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)


def load_csv_tuple(fp, name_tuple):
    return map(name_tuple._make, csv.reader(open(fp, "r", encoding="utf-8")))


def load_pkl(fp):
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data, fp):
    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def load_yaml(fp):
    return yaml.load(open(fp, "r", encoding='utf-8'),
                     Loader=yaml.SafeLoader)


def save_yaml(data, fp):
    yaml.dump(
        data,
        open(fp, "w", encoding="utf-8"),
        allow_unicode=True,
        default_flow_style=False)


def word2id(c2id: dict, word: str):
    word_id = list()

    for i, c in enumerate(word):
        if c in c2id:
            word_id.append(c2id[c])
        else:
            word_id.append(0)
    return word_id


def id2word(id2c: dict, ids: list):
    word = ""
    for i in ids:
        if i in id2c:
            word += id2c[i]
        else:
            return ""
    return word


def calculate_distance(vector1, vector2):
    cosine_distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))  # 余弦夹角
    # euclidean_distance = np.sqrt(np.sum(np.square(vector1 - vector2)))  # 欧式距离
    return cosine_distance


def split_data(data_set, batch_size):
    batch_list = []
    data_size = len(data_set)
    count = (data_size + batch_size - 1) // batch_size
    for i in range(count):
        last_index = data_size if (i + 1) * batch_size > data_size else (i + 1) * batch_size
        res = data_set[i * batch_size:last_index]
        batch_list.append(res)
    return batch_list


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def format_data(t: datetime):
    return t.strftime("%Y-%m-%d %H:%M:%S")


def md5(s: str):
    m = hashlib.md5()
    m.update(s.encode("utf-8"))
    return m.hexdigest()


def scan_fp(ph):
    path_list = []
    for p in os.listdir(ph):
        fp = os.path.join(ph, p)
        if os.path.isfile(fp):
            path_list.append(fp)
        elif os.path.isdir(fp):
            path_list.extend(scan_fp(fp))
    return path_list


def scan_fp_iglob(ph):
    return list(filter(lambda x: os.path.isfile(x), iglob(f"{ph}/**", recursive=True)))
