


# Define the model name to file name mapping

modelname2file = {
    'Mol2MolLSTM': 'lstm',
    'Mol2MolWEISS': 'weiss',
    'Mol2MolWEISSWithB': 'weissb',
    'Mol2MolVAE': 'vae',
    'Mol2Mol': 'mol2mol'
}

mpomodel2nlpmodel = {
    'Mol2MolWEISS': 'Text2TextWEISS',
    'Mol2Mol': 'Text2Text'
}