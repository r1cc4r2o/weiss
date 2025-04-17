from glob import glob
import os
import sys

from natsort import natsorted
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger
from tqdm import tqdm
RDLogger.DisableLog("rdApp.*")

import numpy as np


def get_stats(valid_mols, unique_mols, novel_mols, similar_mols):
    all_valid, all_similar, all_unique, all_novel = [], [], [], []
    for k in valid_mols:
        n_samples = len(valid_mols[k])
        n_valid = np.sum(valid_mols[k])
        n_unique = len(unique_mols[k]) / max(1, n_valid)
        n_novel = len(novel_mols[k]) / max(1, len(unique_mols[k]))
        n_similar = np.sum(similar_mols[k]) / max(1, n_samples)
        all_valid.append(n_valid / n_samples)
        all_similar.append(n_similar)
        all_unique.append(n_unique)
        all_novel.append(n_novel)
    
    return (f"V={np.mean(all_valid):.2f}±{np.std(all_valid):.2f} - " 
            f"N={np.mean(all_novel):.2f}±{np.std(all_novel):.2f} - "
            f"U={np.mean(all_unique):.2f}±{np.std(all_unique):.2f} - "
            f"S={np.mean(all_similar):.2f}±{np.std(all_similar):.2f}")
    
    
if __name__ == "__main__":
    
    train_sources = set([x.strip() for x in open("../dataset/test_source.txt")]
                        + [x.strip() for x in open("../dataset/test_target.txt")])

    fps, can_smilies = {}, {}
    valid, novel, unique, similar = {}, {}, {}, {}
    line_number = 0

    for result_file in natsorted(glob("../results/*_t=*_gpus=*_gpuid=*.csv")):
        pbar = tqdm(open(result_file), total=14245000)
        for line in pbar:
            stop = False
            src, gen = line.strip().split(",")[:2]

            if src not in valid:
                valid[src] = []
                novel[src] = set()
                unique[src] = set()
                similar[src] = []

            gen_mol, gen_can = None, None
            
            if gen in can_smilies:
                gen_mol, gen_can = can_smilies[gen]
            else:
                try:
                    gen_mol = Chem.MolFromSmiles(gen)
                    gen_can = Chem.MolToSmiles(gen_mol)
                    can_smilies[gen] = (gen_mol, gen_can)
                except:
                    gen_mol, gen_can = None, None
            if gen_mol is None:
                valid[src].append(0)
                stop = True
            else:
                valid[src].append(1)

            if not stop:
                if src not in fps:
                    fps[src] = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(src), 2)
                if gen_can not in fps:
                    fps[gen_can] = AllChem.GetMorganFingerprint(gen_mol, 2)

                if (gen_can not in train_sources) and (gen_can not in unique[src]):
                    novel[src].add(gen_can)

                #if gen_can not in unique[src]:
                tanimoto = TanimotoSimilarity(fps[src], fps[gen_can]) >= 0.5
                similar[src].append(tanimoto)

                unique[src].add(gen_can)
            line_number += 1
            
            if (line_number % 100000) == 0:
                # empty cache
                fps = {}
                can_smilies = {}    
                print(len(valid), len(unique), len(novel), len(similar))
                pbar.set_description(get_stats(valid, unique, novel, similar))
                    
        if not os.path.exists(f"../results/table_02"):
            os.makedirs(f"../results/table_02")
        with open(f"../results/table_02/table_02.txt", "a+") as f:
            f.write(result_file.split("/")[-1] + ' ' + get_stats(valid, unique, novel, similar) + "\n")