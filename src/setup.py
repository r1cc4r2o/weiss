
import argparse


#############################################

def setup_build_vocab():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset", type=str, default='../dataset')
    parser.add_argument("--file_name_vocab", type=str, default='vocab.pkl')
    return parser.parse_args()


#############################################


def setup_train_mpo():
    parser = argparse.ArgumentParser( description='Train a model')
    parser.add_argument('--model_name', type=str, default='Mol2MolWEISS')
    parser.add_argument('--path_vocabulary', type=str, default='../dataset/dict_vocab.pt')  
    parser.add_argument('--path_dataset', type=str, default='../dataset/train_test_valid_preprocessed.pt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--_lambda', type=float, default=10.0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_ae_dim', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='Noam')
    return parser.parse_args()


#############################################


def setup_inference_o2o_beamsearch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_lenght", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path_vocabulary", type=str, default='../dataset/dict_vocab.pt')
    parser.add_argument("--base_path_cpk", type=str, default='../checkpoints/Mol2Mol_best.pt')
    return parser.parse_args()

#############################################


def setup_inference_mpo_multinomial():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=float)
    parser.add_argument("--ckpt_path", type=str) 
    parser.add_argument("--gpu_id", type=int, default=0) 
    parser.add_argument("--gpus", type=int, default=1) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--path_vocabulary", type=str, default='../dataset/dict_vocab.pt')
    _args = parser.parse_args()
    return _args


#############################################


def setup_training_nlp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument(  # DDP configs
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--type_model", default="Mol2MolWEISS", type=str, help="type of model")
    parser.add_argument("--lambda_", default=40.0, type=float, help="mmd lambda")
    parser.add_argument("--b", default=16, type=int, help="bottleneck size")
    parser.add_argument("--base_path", default="setup_path", type=str, help="base path")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    args = parser.parse_args()
    return args


#############################################


def setup_inference_multinomial_nlp():
    parser = argparse.ArgumentParser(description='Inference NLP')
    parser.add_argument('--path', type=str, default='', help='Path')
    parser.add_argument('--N', type=int, default=None, help='Number of samples to evaluate, if None all')
    parser.add_argument('--K', type=int, default=10, help='Number of samples to evaluate')
    parser.add_argument('--base_path', type=str, default='base_path', help='Base path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--run_name', type=str, default='Text2Text', help='Run name')
    return parser.parse_args()


#############################################


def setup_table_04():
    parser = argparse.ArgumentParser(description='Table 04')
    parser.add_argument('--path_df_gt', type=str, default=None, help='Path to the ground truth dataframe')
    parser.add_argument('--path_df_generated', type=str, default=None, help='Path to the generated dataframe')
    args = parser.parse_args()
    return args

#############################################

def setup_convert_smiles():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, default='../dataset/data/similarity',help='The path to the dataset')
    parser.add_argument('--_split', type=str, help='The split of the dataset')
    return parser.parse_args()


#############################################


