from pathlib import Path

import hydra
import pandas as pd

import molbart.modules.util as util
from molbart.models import Chemformer
from molbart.modules.data.base import SimpleReactionListDataModule
from molbart.modules.decoder import DecodeSampler
from molbart.modules.tokenizer import ChemformerTokenizer


def write_predictions(args, smiles, log_lhs, original_smiles):
    num_data = len(smiles)
    beam_width = len(smiles[0])
    beam_outputs = [[[]] * num_data for _ in range(beam_width)]
    beam_log_lhs = [[[]] * num_data for _ in range(beam_width)]

    for b_idx, (smiles_beams, log_lhs_beams) in enumerate(zip(smiles, log_lhs)):
        for beam_idx, (smi, log_lhs) in enumerate(zip(smiles_beams, log_lhs_beams)):
            beam_outputs[beam_idx][b_idx] = smi
            beam_log_lhs[beam_idx][b_idx] = log_lhs

    df_data = {"original_smiles": original_smiles}
    for beam_idx, (outputs, log_lhs) in enumerate(zip(beam_outputs, beam_log_lhs)):
        df_data["prediction_" + str(beam_idx)] = beam_outputs[beam_idx]
        df_data["log_likelihood_" + str(beam_idx)] = beam_log_lhs[beam_idx]

    df = pd.DataFrame(data=df_data)
    df.to_pickle(Path(args.products_path))


def load_bart_model(args):
    print("Building tokeniser...")
    tokeniser = ChemformerTokenizer(filename=args.vocabulary_path)
    print("Finished tokeniser.")

    sampler = DecodeSampler(tokeniser, util.DEFAULT_MAX_SEQ_LEN)

    print("Loading model...")
    model = util.load_bart(args, sampler)
    model.num_beams = args.num_beams
    sampler.max_seq_len = util.DEFAULT_MAX_SEQ_LEN
    print("Finished model.")
    return model


def build_datamodule(args, tokeniser):
    dm = SimpleReactionListDataModule(
        dataset_path=args.reactants_path,
        tokenizer=tokeniser,
        batch_size=args.batch_size,
        max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
    )
    return dm


def build_data_loader(args, tokenizer):
    print("Building data loader...")
    dm = build_datamodule(args, tokenizer)
    dm.setup()
    data_loader = dm.full_dataloader()
    print("Finished loader.")
    return data_loader


@hydra.main(version_base=None, config_path="config", config_name="predict")
def main(args):
    model_args, data_args = util.get_chemformer_args(args)

    kwargs = {
        "config": args,
        "vocabulary_path": args.vocabulary_path,
        "n_gpus": args.n_gpus,
        "model_path": args.model_path,
        "model_args": model_args,
        "data_args": data_args,
        "n_beams": args.n_beams,
        "train_mode": "eval",
    }

    chemformer = Chemformer(
        **kwargs,
        datamodule_type="simple_reaction_list",
    )

    print("Making predictions...")
    smiles, log_lhs, original_smiles = chemformer.predict(dataset=args.dataset_part)
    write_predictions(args, smiles, log_lhs, original_smiles)
    print("Finished predictions.")
    return


if __name__ == "__main__":
    main()
