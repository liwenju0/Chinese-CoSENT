"""
@file   : run_cosent_ATEC.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-07
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from config import set_args
from model import Model
from torch.utils.data import DataLoader
from utils import l2_normalize, compute_corrcoef, set_seed
from transformers.models.bert import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_helper import CustomDataset, collate_fn, pad_to_maxlen, load_data, load_test_data
import functools


def get_sent_id_tensor(s_list, max_len=128):
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    for s in s_list:
        inputs = tokenizer.encode_plus(text=s, text_pair=None, add_special_tokens=True, return_token_type_ids=True)
        input_ids.append(pad_to_maxlen(inputs['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(inputs['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(inputs['token_type_ids'], max_len=max_len))
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


if __name__ == '__main__':
    set_seed()
    args = set_args()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    model = Model(args.pretrained_model_path)
    model.load_state_dict(torch.load(args.output_dir + "/best_model.bin"))

    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    while True:
        try:
            sents = []
            all_a_vecs = []
            all_b_vecs = []
            for i in range(2):
                s = input("请输入句子{}".format(i+1))
                sents.append(s)
            s1, s2 = sents[0], sents[1]
            input_ids, input_mask, segment_ids = get_sent_id_tensor([s1, s2], max_len=128)
            if torch.cuda.is_available():
                input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()

            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=input_mask,
                               token_type_ids=segment_ids, encoder_type='fist-last-avg')

            all_a_vecs.append(output[0].cpu().numpy())
            all_b_vecs.append(output[1].cpu().numpy())

            all_a_vecs = np.array(all_a_vecs)
            all_b_vecs = np.array(all_b_vecs)
            a_vecs = l2_normalize(all_a_vecs)
            b_vecs = l2_normalize(all_b_vecs)
            sims = (a_vecs * b_vecs).sum(axis=1)

            print("二者语义相似度为：", sims[0])
        except Exception as e:
            print(e)
