
import torch
import numpy as np

import nltk
import evaluate

from torchtext.data.metrics import bleu_score
from lexical_diversity import lex_div as ld

nltk.download("punkt", quiet=True)
metric_rouge = evaluate.load("rouge")

from typing import List



def flemmatize(sentence: str):
    return ld.flemmatize(sentence)


def attr(sentence_tok: List[str]):
    ttr = 0
    for sentence in sentence_tok:
        ttr += ld.ttr(sentence)
    return ttr/len(sentence_tok)


def compute_div(sentence: str):
    results = {}
    sentence_tok = flemmatize(sentence)
    _attr = attr(sentence_tok)
    results['attr'] = _attr
    results['mattr'] = ld.mattr(sentence_tok, window_length=35)
    results['mtld'] = ld.mtld(sentence_tok)
    results['hdd'] = ld.hdd(sentence_tok)
    return results


def join(examples: List[str]):
    examples = ' '.join(examples)
    return examples


def get_metrics(yp, y, vocab):
    
    # pad all the tokens after
    # the <eos> token
    for i in range(len(yp)):
        idx_yp_i = torch.where(yp[i] == 2)[0]
        if len(idx_yp_i) > 0:
            if len(yp[i]) > idx_yp_i[0]+1:
                yp[i][idx_yp_i[0]+1:] = 0

    yp = [p[p!=0.0]for p in yp]
    y = [p[p!=0.0]for p in y]
    
    candidate_corpus = [vocab.decode(p.tolist()) for p in yp]
    references_corpus = [[vocab.decode(p.tolist())] for p in y]
    metrics = {'bleu': bleu_score(candidate_corpus, references_corpus)}

    candidate_corpus = [vocab.untokenize(vocab.decode(p.tolist())) for p in yp]
    references_corpus = [vocab.untokenize(vocab.decode(p.tolist())) for p in y]
    metrics.update(metric_rouge.compute(predictions=candidate_corpus, references=references_corpus, use_stemmer=True))
    
    _results = {'attr': [], 'mattr': [], 'mtld': [], 'hdd': []}
    for idx, s in enumerate(candidate_corpus):
        if len(s) == 0:
            print("empty sentence")
            print(candidate_corpus)
        else:
            temp = compute_div(s)
            for key in temp:
                _results[key].append(temp[key])

    for key in _results:
        metrics[key] = sum(_results[key])/len(_results[key])
        metrics[key + "_std"] = np.std(_results[key])
        
    return metrics
  
  
def compute_metrics(y, yp, x, vocab):
    
    # pad all the tokens after
    # the <eos> token
    for i in range(len(yp)):
        idx_yp_i = torch.where(yp[i] == 2)[0]
        if len(idx_yp_i) > 0:
            if len(yp[i]) > idx_yp_i[0]+1:
                yp[i][idx_yp_i[0]+1:] = 0

    yp = [p[p!=0.0]for p in yp]
    y = [p[p!=0.0]for p in y]
    
    candidate_corpus = [vocab.decode(p.tolist()).split(' ') for p in yp]
    references_corpus = [[vocab.decode(p.tolist()).split(' ')] for p in y]
    metrics = {'bleu': bleu_score(candidate_corpus, references_corpus)}

    candidate_corpus = [vocab.decode(p.tolist()) for p in yp]
    references_corpus = [vocab.decode(p.tolist()) for p in y]
    question_corpus = [vocab.decode(p.tolist()) for p in x]
    metrics.update(metric_rouge.compute(predictions=candidate_corpus, references=references_corpus, use_stemmer=True))
    
    _results = {'attr': [], 'mattr': [], 'mtld': [], 'hdd': []}
    for idx, s in enumerate(candidate_corpus):
        if len(s) != 0:
            temp = compute_div(s)
            for key in temp:
                _results[key].append(temp[key])

    for key in _results:
        metrics[key] = sum(_results[key])/len(_results[key])
        metrics[key + "_std"] = np.std(_results[key])

    return metrics, [(q, a, a_yp) for q, a, a_yp in zip(question_corpus, references_corpus, candidate_corpus)]
