import pandas as pd
from tqdm import tqdm
import numpy as np

from src.setup import setup_table_04

def load_csv(file_path):
    lines = open(file_path, 'r')
    full_df = []
    header = lines.readline().strip().split('Ϟ')
    for line in tqdm(lines):
        line = line.strip().split('Ϟ')
        if len(line) == len(header):
            full_df.append(line)
    return pd.DataFrame(full_df, columns=header)


args = setup_table_04()

df_gt = load_csv(args.path_df_gt)
df_generated = load_csv(args.path_df_generated)
df_gt = df_gt[df_gt['x'] == 'What is the meaning of life?']
df_generated = df_generated[df_generated['x'] == 'What is the meaning of life?']


dataset_nlls = {}
for _, method in df_gt.groupby("method"):
    name = method["method"].values[0]
    if name not in dataset_nlls:
        dataset_nlls[name] = {}
    for _, x in tqdm(method.groupby("x")):
        s = str(x["x"].values[0]).strip()
        if s not in dataset_nlls[name]:
            dataset_nlls[name][s] = []
        dataset_nlls[name][s] +=  (x["nlls"].values.astype(np.float32)).tolist()
        

generated_nlls = {}
for _, method in df_generated.groupby(["method", "temperature"]):
    model_name = str(method["method"].values[0])
    temp = str(method["temperature"].values[0])
    generated_nlls[(model_name, temp)] = {}
    for _, x in tqdm(method.groupby("x")):
        s = str(x["x"].values[0]).strip()
        if s not in generated_nlls[(model_name, temp)]:
            generated_nlls[(model_name, temp)][s] = []
        generated_nlls[(model_name, temp)][s] +=  (x["nlls"].values.astype(np.float32)).tolist()


for k in generated_nlls:
    model_name, temp = k
    kl = []
    for source in generated_nlls[k]:
        try:
            X = np.array(generated_nlls[k][source])[:, None].astype(np.float32)
            Y = np.array(dataset_nlls[model_name][source])[None]
            kl += ((X - Y) * np.exp(-Y) ).sum(1).tolist()
        except:
            print('Error')
    kl = np.array(kl)
    print(k, np.sum(kl), np.std(kl))
        