import hydra

import molbart.modules.util as util
from molbart.models import Chemformer


@hydra.main(version_base=None, config_path="config", config_name="evaluate")
def main(args):
    util.seed_everything(args.seed)

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
    }

    chemformer = Chemformer(**kwargs)

    trainer = util.build_trainer(args)
    print("Evaluating model...")
    results = trainer.test(chemformer.model, datamodule=chemformer.datamodule)

    print(f"Results for model: {args.model_path}")
    print(f"{'Item':<25}Result")
    for key, val in results[0].items():
        print(f"{key:<25} {val:.4f}")
    print("Finished evaluation.")
    return


if __name__ == "__main__":
    main()
