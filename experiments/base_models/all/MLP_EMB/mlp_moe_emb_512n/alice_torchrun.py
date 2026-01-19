import argparse

from moe_mlp_emb_512 import MoeMLPEmbedded_2x512

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.model_runner import AliceModelRunner

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--training_args_mode', type=str, default="FILE", choices=["FILE", "CMD_LINE"])
parser.add_argument('--training_args_path', type=str)

def main():
    args = parser.parse_args()
    training_args_mode = ArgsMode(args.training_args_mode)
    training_args_path = args.training_args_path
    training_config = TrainingConfig(args_mode = training_args_mode, training_args_path = training_args_path)

    AliceModelRunner.train_distributed(training_config, MoeMLPEmbedded_2x512)

if __name__ == '__main__':
    main()
