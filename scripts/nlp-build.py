import sys
import os
import torch
import random
import pandas as pd
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset



#############################################
# Utils.
#############################################


def get_full_data(dataset, split):
    _i, _questions, _answers, _main_category = [], [], [], []
    for ids, q, top_a, answers, main_category in tqdm(zip(dataset[split]["id"], dataset[split]["question"],dataset[split]["answer"], dataset[split]["nbestanswers"], dataset[split]["main_category"])):
        for a in answers:
            _i.append(ids)
            _questions.append(q)
            _answers.append(a)
            _main_category.append(main_category)
    return {"id": _i, "question": _questions, "answer": _answers, "main_category": _main_category, "split": [split]*len(_i)}


def save_dataset():
    dataset = load_dataset("yahoo_answers_qa")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dict_train = get_full_data(dataset, "train")
    dict_test = get_full_data(dataset, "test")

    df_train = pd.DataFrame(dict_train)
    df_test = pd.DataFrame(dict_test)

    full_df = pd.concat([df_train, df_test], axis=0)
    full_df.to_csv("../dataset/yahoo_answers_qa.csv", index=False)

    _list_sentences = []
    for q, a in zip(full_df["question"], full_df["answer"]):
        _list_sentences.append(f"{q} {a}")
        
    write_file = open("../dataset/yahoo_answers_qa.txt", "w")
    write_file.write("\n".join(_list_sentences))
    write_file.close()


def train_tokenizer():
    tk_cfg = dict(
        input = "../dataset/yahoo_answers_qa.txt",
        input_format = "text",
        model_prefix = "yahooqa_tok",
        model_type = "bpe",
        normalization_rule_name = "identity",
        remove_extra_whitespaces = False,
        seed_sentencepiece_size = 1000000,
        shuffle_input_sentence = True,
        character_coverage = 0.98,
        byte_fallback = True,
        split_digits = True,
        split_by_unicode_script = True,
        split_by_whitespace = True,
        split_by_number = True,
        add_dummy_prefix = True,
        allow_whitespace_only_pieces = True,
        unk_id = 3,
        bos_id = 1,
        eos_id = 2,
        pad_id = 0,
        num_threads = os.cpu_count(),
    )
    spm.SentencePieceTrainer.train(**tk_cfg)
    
    
    
def save_idx_validation_from_train():
    sm = spm.SentencePieceProcessor(model_file='yahooqa_tok.model')
    train = torch.load('../dataset/yahoo_answers_qa_train.pt')
    sentences = [(sm.decode(q.tolist()), sm.decode(a.tolist()), idx) for idx, (q, a) in enumerate(train)]
    questions, answers, idx = zip(*sentences)
    questions = list(questions)
    answers = list(answers)
    idx = list(idx)
    unique_pairs = {}
    for q, a, i in zip(questions, answers, idx):
        if q not in unique_pairs:
            unique_pairs[q] = (a, i)
    questions, answers = unique_pairs.keys(), unique_pairs.values()
    answers, idx = zip(*answers)
    random.seed(42)
    idx_sampled = random.sample(idx, int(len(questions)*0.1) )
    torch.save(idx_sampled, '../dataset/yahoo_answers_qa_valid_idx_train.pt')
    

def save_encoded_sentences():
    
    vocab = spm.SentencePieceProcessor(model_file="yahooqa_tok.model")
    full_df = pd.read_csv("../dataset/yahoo_answers_qa.csv")
    full_df.dropna(inplace=True)
    _list_enc_sentences_train = []
    _list_enc_sentences_test = []
    for q, a, split in tqdm(zip(full_df["question"], full_df["answer"], full_df["split"])):
        enc_q = torch.tensor([vocab.bos_id()] + vocab.encode(q) + [vocab.eos_id()], dtype=torch.int16)
        enc_a = torch.tensor([vocab.bos_id()] + vocab.encode(a) + [vocab.eos_id()], dtype=torch.int16)
        if len(enc_q) < 128 and len(enc_a) < 128:
            if split == "train":
                _list_enc_sentences_train.append((enc_q, enc_a))
            else:
                _list_enc_sentences_test.append((enc_q, enc_a))
                
    torch.save(_list_enc_sentences_train, "../dataset/yahoo_answers_qa_train.pt")
    torch.save(_list_enc_sentences_test, "../dataset/yahoo_answers_qa_test.pt")
    
    
    
#############################################
# Main.
#############################################

    
if __name__ == "__main__":
    save_dataset()
    train_tokenizer()
    save_idx_validation_from_train()
    save_encoded_sentences()
    