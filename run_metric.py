import os
import json
import torch
from collections import defaultdict
from tqdm import trange
from pytorch_pretrained_bert import BertTokenizer, BertModel
from bert_score.utils import collate_idf
from bert_score.utils import get_idf_dict, get_bert_embedding


def get_idf_dict_from_samples(samples):
    caps_list = []
    for i in range(len(samples)):
        for cap_type in ['refs', 'cand']:
            for j in range(len(samples[str(i)][cap_type])):
                caps_list.append(samples[str(i)][cap_type][j])

    return get_idf_dict(caps_list, tokenizer)


def compute_hiddens_and_idf(samples, name, path='./example/', save=True):
    if os.path.exists(path + name + '.pt'):
        samples = torch.load(path + name + '.pt', map_location=device)
    else:
        idf_dict = defaultdict(lambda: 1.)
        idf_dict[101], idf_dict[102] = 0, 0  # [SEP] and [CLS] to 0
        print('---> computing contextual embeddings')
        for i in trange(len(samples)):
            for cap_type in ['refs', 'cand']:
                samples[str(i)][cap_type + '_hid'] = []
                for j in range(len(samples[str(i)][cap_type])):
                    cap = samples[str(i)][cap_type][j]
                    samples[str(i)][cap_type + '_hid'].append(
                        get_bert_embedding([cap], model, tokenizer, idf_dict, batch_size=1,
                                           device=device)[0])

        print('---> computing idf weights')
        idf_dict = get_idf_dict_from_samples(samples)
        for i in trange(len(samples)):
            for cap_type in ['refs', 'cand']:
                samples[str(i)][cap_type + '_idf'] = []
                for j in range(len(samples[str(i)][cap_type])):
                    cap = samples[str(i)][cap_type][j]
                    _, padded_idf, _, _ = collate_idf([cap], tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                      idf_dict, device=device)
                    samples[str(i)][cap_type + '_idf'].append(padded_idf[0])

        if save: torch.save(samples, path + name + '.pt')
    return samples


def matrix_search_mismatch(ref_hid, cand_hid, THRESHOLD=0.0, idfs=None):
    ref_hid.div_(torch.norm(ref_hid, dim=-1).unsqueeze(-1))
    cand_hid.div_(torch.norm(cand_hid, dim=-1).unsqueeze(-1))
    cand_hid.to(device)
    ref_hid.to(device)
    sim = torch.bmm(cand_hid, ref_hid.transpose(1, 2))
    sim = sim[:, 1:-1, 1:-1]
    R_masks = list()
    for r in range(0, sim.shape[2]):
        tmp = 0.
        for g in range(0, sim.shape[1]):
            tmp = max(torch.sum(sim[:, g:g + 1, r:r + 1]), tmp)
        R_masks.append(1 if tmp > THRESHOLD else 0)
    return R_masks


def matrix_compute_rm_stop_word(ref_hid, cand_hid, THRESHOLD=0.0, tokens_R=None, tokens_C=None):
    ref_hid.div_(torch.norm(ref_hid, dim=-1).unsqueeze(-1))
    cand_hid.div_(torch.norm(cand_hid, dim=-1).unsqueeze(-1))
    cand_hid.to(device)
    ref_hid.to(device)
    sim = torch.bmm(cand_hid, ref_hid.transpose(1, 2))
    sim = sim[:, 1:-1, 1:-1]

    R = 0.
    R_count = 0.
    for r in range(0, sim.shape[2]):
        if tokens_R[r] in stop_word_list:
            continue
        tmp = 0.
        for g in range(0, sim.shape[1]):
            if tokens_C[g] not in stop_word_list:
                tmp = max(torch.sum(sim[:, g:g + 1, r:r + 1]), tmp)
        if tmp > THRESHOLD:
            R += tmp
            R_count += 1.

    return R / R_count if R != 0.0 else 0.0


def matrix_compute_full(ref_hid, cand_hid, THRESHOLD=0.0, idfs_ref=None):
    ref_hid.div_(torch.norm(ref_hid, dim=-1).unsqueeze(-1))
    cand_hid.div_(torch.norm(cand_hid, dim=-1).unsqueeze(-1))
    cand_hid.to(device)
    ref_hid.to(device)
    sim = torch.bmm(cand_hid, ref_hid.transpose(1, 2))
    sim = sim[:, 1:-1, 1:-1]
    score_full, idf_full = 0.0, 0.0

    for r in range(0, sim.shape[2]):
        tmp = 0.
        for g in range(0, sim.shape[1]):
            if float(sim[:, g:g + 1, r:r + 1]) > tmp:
                tmp = float(sim[:, g:g + 1, r:r + 1])
        if tmp > THRESHOLD:
            score_full += tmp * (idfs_ref[r + 1].item())
            idf_full += idfs_ref[r + 1].item()
    return (score_full / idf_full) if idf_full != 0. else 0.0


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def collate_idf_tokenized(arr, numericalize, idf_dict,
                          pad="[PAD]"):
    tmp_a = list()
    for a in arr:
        tmp_a.append(["[CLS]"] + a + ["[SEP]"])
    arr = tmp_a
    arr = [numericalize(a) for a in arr]
    idf_weights = [[idf_dict[i] for i in a] for a in arr]
    pad_token = numericalize([pad])[0]
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    return padded_idf


def get_stop_word_list(file_path='./stop_word_list.txt'):
    stop_word_list = []
    with open(file_path, 'r') as file:
        for line in file:
            stop_word_list.append(line.strip())
    return stop_word_list


print('---> loading BERT model')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert)
model = BertModel.from_pretrained(bert)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:8]])

name = 'flickr'
with open('example/'+name+'.json', 'r') as file:
    samples = json.load(file)
samples = compute_hiddens_and_idf(samples, name, save=False)
idf_dict = get_idf_dict_from_samples(samples)
stop_word_list = get_stop_word_list()
THRESHOLD = 0.4

if 'mismatch' not in samples['0'].keys():
    print('---> detecting mismatches')
    for i in trange(len(samples)):
        samples[str(i)]['mismatch'] = []
        for curr_idx in range(len(samples[str(i)]['refs'])):
            current_record = {}
            for next_idx in range(len(samples[str(i)]['refs'])):
                if next_idx != curr_idx:
                    R_masks = matrix_search_mismatch(samples[str(i)]['refs_hid'][next_idx],
                                                     samples[str(i)]['refs_hid'][curr_idx],
                                                     THRESHOLD=0.4, idfs=samples[str(i)]['refs_idf'][next_idx])
                    current_record[next_idx] = R_masks
            samples[str(i)]['mismatch'].append(current_record)

    torch.save(samples, './example/' + name + '.pt')

print('---> computing scores')

for i in trange(len(samples)):
    tokens_cand = tokenizer.tokenize(samples[str(i)]['cand'][0])
    idfs_cand = collate_idf_tokenized([tokens_cand], tokenizer.convert_tokens_to_ids,
                                      idf_dict).reshape(-1)
    R_full = 0.
    # combine and avoid overlap-combination
    for next_idx in range(len(samples[str(i)]['refs'])):
        hidden_ref = samples[str(i)]['refs_hid'][next_idx]
        tokens_ref = tokenizer.tokenize(samples[str(i)]['refs'][next_idx])

        for left_idx in range(len(samples[str(i)]['refs'])):
            if left_idx != next_idx:
                tokens_left = tokenizer.tokenize(samples[str(i)]['refs'][left_idx])
                marks_base = samples[str(i)]['mismatch'][next_idx][left_idx]
                for k in range(len(marks_base)):
                    if marks_base[k] == 0:
                        duplicate = False
                        for check_idx in range(len(samples[str(i)]['refs'])):
                            if check_idx < left_idx and check_idx != next_idx:
                                marks = samples[str(i)]['mismatch'][check_idx][left_idx]
                                if marks[k] == 1: duplicate = True

                        if not duplicate:
                            hidden_ref = torch.cat(
                                [hidden_ref, samples[str(i)]['refs_hid'][left_idx][0][k].reshape(1, 1, -1)], 1)
                            tokens_ref.append(tokens_left[k])

        idfs_ref = collate_idf_tokenized([tokens_ref], tokenizer.convert_tokens_to_ids,
                                         idf_dict).reshape(-1)

        score = matrix_compute_full(hidden_ref, samples[str(i)]['cand_hid'][0], THRESHOLD,
                                    idfs_ref)

        score_rm = matrix_compute_rm_stop_word(hidden_ref, samples[str(i)]['cand_hid'][0], THRESHOLD, tokens_ref,
                                               tokens_cand)
        # BERTScore uses 'max' as the pooling function
        R_full += score_rm * score
    samples[str(i)]['metric_result'] = float(R_full) / len(samples[str(i)]['refs'])

for i in range(len(samples)):
    print('Sample' + str(i), 'Score: ', round(samples[str(i)]['metric_result'], 2))

# # to reproduce our results on Flickr dataset
# # 1. modify the value 'name' to 'flickr'
# # 2. uncomment these lines
# from scipy.stats import kendalltau
#
# sim_scores = []
# expert_scores = []
# for i in range(len(samples)):
#     sim_scores.append(samples[str(i)]['metric_result'])
#     expert_scores.append(samples[str(i)]['score'])
#
# Kendallta2, p_value = kendalltau(sim_scores, expert_scores)
# print(Kendallta2, p_value)
