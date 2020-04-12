from collections import defaultdict
from tqdm import trange
import time
import json


def precook(s, n=4):
    words = s.split()
    counts = defaultdict(int)

    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        counts[ngram] += 1
    return (len(words), counts)


def combined(ref, cand,stop_words=None):
    if stop_words != None:
        ref = clear_stop_words(ref,stop_words)
        cand = clear_stop_words(cand,stop_words)
    ref_len, ref_counts = precook(ref, 1)
    cand_len, cand_counts = precook(cand, 1)
    l = ref.split()
    for (ngram, count) in cand_counts.items():
        if ref_counts.get(ngram, 0) == 0:
            l.append(ngram[0])
    return ' '.join(l)

def clear_stop_words(stc,stop_words):
    new_stc = ''
    for word in stc.split():
        if word not in stop_words:
            new_stc =  new_stc + ' ' + word
    return new_stc.strip()


def get_simple_score(refs, cand,stop_words=None):
    ref = refs
    if stop_words != None:
        ref = clear_stop_words(ref,stop_words)
        cand = clear_stop_words(cand,stop_words)
    ref_len, ref_counts = precook(ref, 1)
    cand_len, cand_counts = precook(cand, 1)
    correct = 0
    for (ngram, count) in ref_counts.items():
        correct += min(cand_counts.get(ngram, 0), 1)
    if len(ref.split()) == 0:
        return 0
    return correct / len(ref.split())



def get_stop_word_list(file_path='./stop_word_list.txt'):
    stop_word_list = []
    with open(file_path, 'r') as file:
        for line in file:
            stop_word_list.append(line.strip())
    return stop_word_list



with open('example/samples.json', 'r') as file:
    samples = json.load(file)
stop_word_list = get_stop_word_list()
sim_scores = []


st_time = time.time()
for i in trange(len(samples)):

    scores = 0
    cand = samples[str(i)]['cand'][0]
    base_ref = ''
    flag = False
    for next_idx in range(len(samples[str(i)]['refs'])):
        next_ref = samples[str(i)]['refs'][next_idx]
        base_ref = combined(base_ref, next_ref)

    scores += get_simple_score(base_ref, cand, stop_words=stop_word_list)
    samples[str(i)]['metric_result'] = scores

for i in range(len(samples)):
    print('Sample' + str(i), 'Score: ', round(samples[str(i)]['metric_result'], 2))





