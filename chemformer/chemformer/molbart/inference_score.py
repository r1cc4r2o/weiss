import hydra

import molbart.modules.util as util
from molbart.models import Chemformer


@hydra.main(version_base=None, config_path="config", config_name="inference_score")
def main(args):
    util.seed_everything(args.seed)

    print("Running model inference and scoring.")

    if args.dataset_type not in [
        "uspto_mixed",
        "uspto_50",
        "uspto_sep",
        "uspto_50_with_type",
        "synthesis",
    ]:
        raise ValueError(f"Unknown dataset: {args.dataset_type}")

    model_args, data_args = util.get_chemformer_args(args)

    kwargs = {
        "config": args,
        "vocabulary_path": args.vocabulary_path,
        "n_gpus": args.n_gpus,
        "model_path": args.model_path,
        "model_args": model_args,
        "data_args": data_args,
        "n_beams": args.n_beams,
        "n_multinomial": args.n_multinomial,
        "train_mode": "eval",
        "sample_unique": args.n_unique_beams is not None,
    }

    chemformer = Chemformer(**kwargs)

    chemformer.score_model(
        n_unique_beams=args.n_unique_beams,
        dataset=args.dataset_part,
        i_chunk=args.i_chunk,
        n_chunks=args.n_chunks,
        output_scores=args.output_score_data,
        output_sampled_smiles=args.output_sampled_smiles,
        output_scores_efficiency=args.output_score_efficiency
    )
    print("Model inference and scoring done.")
    return


if __name__ == "__main__":
    main()
