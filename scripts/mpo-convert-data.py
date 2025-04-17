import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.setup import setup_convert_smiles
from src.module.conversion import Conversions
from src.utils import write_to_txt_file_smiles_data



def main(_args):

    print('Preprocessing configuration:')
    print(_args)

    # read the csv files
    csv = pd.read_csv(f'{_args.path_dataset}/{_args._split}.csv')

    dict_smiles = {}
    for id_srcortgt in ['Source_Mol', 'Target_Mol']:
        smiles_set = set(csv[id_srcortgt].tolist())
        
        # convert the smiles to the canonical form
        # and remove explicit hydrogens
        conv = Conversions()
        canon_smiles = conv.batched_convert_to_standardized_smiles(smiles_set)
        
        dict_smiles[id_srcortgt] = smiles_set
        dict_smiles[id_srcortgt + '_canon'] = canon_smiles
        
    print('SMILES conversion done.')
    
    # convert the list to numpy array
    array_source_mols = np.array(csv['Source_Mol'].tolist())
    array_target_mols = np.array(csv['Target_Mol'].tolist())
    del csv

    print('Substituting the source and target molecules with the canonical form...')
    for source_mol, source_mol_canon, target_mol, target_mol_canon in tqdm(zip(dict_smiles['Source_Mol'], dict_smiles['Source_Mol_canon'], dict_smiles['Target_Mol'], dict_smiles['Target_Mol_canon'])):
        
        # find the index where the source mol is the same as the source_mol_canon
        ids = np.where(array_source_mols == source_mol)
        
        # substitute the list_source_mols with the source_mol_canon
        array_source_mols[ids] = source_mol_canon
        
        # find the index where the target mol is the same as the target_mol_canon
        ids = np.where(array_target_mols == target_mol)
        
        # substitute the list_target_mols with the target_mol_canon
        array_target_mols[ids] = target_mol_canon
        
    if not os.path.exists('../dataset'):
        os.makedirs('../dataset')
        print('Created the dataset directory, where the data will be stored.')
        
    print('Substitution done.')
    print('Writing the data to the txt files...')
    print(f'source: ../dataset/{_args._split}_source.csv')
    print(f'target: ../dataset/{_args._split}_target.csv')
    # write the data to the txt files
    write_to_txt_file_smiles_data(f'../dataset/{_args._split}_source.csv', array_source_mols.tolist())
    write_to_txt_file_smiles_data(f'../dataset/{_args._split}_target.csv', array_target_mols.tolist())
    print('Writing done.')


if __name__ == '__main__':
    print('Preprocessing started...')
    args = setup_convert_smiles()
    main(args)
    print('Preprocessing done.')
