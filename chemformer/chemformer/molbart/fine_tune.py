import time

import hydra

import molbart.modules.util as util
from molbart.models import Chemformer


@hydra.main(version_base=None, config_path="config", config_name="fine_tune")
def main(args):
    util.seed_everything(args.seed)
    print("Fine-tuning CHEMFORMER.")
    model_args, data_args = util.get_chemformer_args(args)

    kwargs = {
        "config": args,
        "vocabulary_path": args.vocabulary_path,
        "n_gpus": args.n_gpus,
        "model_path": args.model_path,
        "model_args": model_args,
        "data_args": data_args,
        "resume_training": args.resume,
    }

    chemformer = Chemformer(**kwargs)

    print("Training model...")
    t0 = time.time()
    chemformer.fit()
    t_fit = time.time() - t0
    print(f"Training complete, time: {t_fit}")
    print("Done fine-tuning.")
    return


if __name__ == "__main__":
    main()
