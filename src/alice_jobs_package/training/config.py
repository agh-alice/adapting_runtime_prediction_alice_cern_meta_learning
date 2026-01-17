import torch

import os
import json
import random
import argparse
import pkg_resources
from pathlib import Path

from alice_jobs_package.utils import logging, tools, project_config, wandb_logging
from alice_jobs_package.preprocessor import AliceDataPreprocessor

logger = logging.get_logger(__name__)

class TrainingConfig():
    def __init__(self, args_mode: project_config.ArgsMode, training_args_path=None, alice_preprocessor: AliceDataPreprocessor = None):
        
        self.args = TrainingConfig.parse(args_mode, training_args_path)

        self.parse_resume_state()        
        self.setup_cache()
        self.setup_seed()
        self.setup_gpu_config()
        self.setup_alicepreprocesing_config(alice_preprocessor)
        self.validate_config(alice_preprocessor)
        wandb_logging.setup_wandb(self.args)
    
    def parse_resume_state(self):
        if self.args.resume:
            self.args.resume = tools.parse_to_pathlib(self.args.resume)
    
    def setup_cache(self):
        self.results_save_path = tools.parse_to_pathlib(self.args.results_save_path) / f"{self.args.model_name}_cache"
        logger.info(f'Save path: {self.results_save_path}')

        self.model_save_path = self.results_save_path / project_config.ModelCacheSubFolders.SAVED_MODEL.value
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.plot_save_path = self.results_save_path / project_config.ModelCacheSubFolders.SAVED_PLOTS.value
        self.plot_save_path.mkdir(parents=True, exist_ok=True)
        self.history_save_path = self.results_save_path / project_config.ModelCacheSubFolders.SAVED_HISTORY.value
        self.history_save_path.mkdir(parents=True, exist_ok=True)
    
    def setup_seed(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

    def setup_gpu_config(self):
        self.ngpus_per_node = torch.cuda.device_count()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.args.multiprocessing_distributed:
            self.args.gpu = int(os.environ["LOCAL_RANK"])
            self.args.rank = int(os.environ["RANK"])
            self.args.world_size = int(os.environ['WORLD_SIZE'])
            self.args.master_addr = os.environ['MASTER_ADDR']
            self.args.master_port = os.environ['MASTER_PORT']
            self.args.init_method = f'tcp://{self.args.master_addr}:{self.args.master_port}'
            self.device = torch.device('cuda:{}'.format(self.args.gpu))
            if self.args.rank != 0: self.args.verbose = False
        
    def setup_alicepreprocesing_config(self, alice_preprocessor : AliceDataPreprocessor):
        self.column_names = None
        self.num_config = None
        self.cat_config = None
        self.filter_config = None
        self.num_col_numbers = None
        self.cat_col_numbers = None

        if alice_preprocessor:
            self.column_names = list(alice_preprocessor.columns)
            self.num_config = alice_preprocessor.numerical_columns_config
            self.cat_config = alice_preprocessor.categorical_columns_config
            self.filter_config = alice_preprocessor.filter_data_config
            self.num_col_numbers = alice_preprocessor.num_col_numbers
            self.cat_col_numbers = alice_preprocessor.cat_col_numbers

    def validate_config(self, alice_preprocessor : AliceDataPreprocessor):
        if alice_preprocessor and self.args.attn_heads:
            assert len(self.num_col_numbers) % self.args.attn_heads == 0
        
        assert torch.cuda.is_available()

        if self.args.time_series_encoding or self.args.sequence_grouping_column:
            processing_target = project_config.ProcessingTarget(self.args.processing_target)
            assert (processing_target == project_config.ProcessingTarget.MLP_EMBEDINGS or 
                processing_target == project_config.ProcessingTarget.TRANSFORMER_EMBEDINGS)
        
    @staticmethod
    def parse(args_mode: project_config.ArgsMode, training_args_path):
        if args_mode == project_config.ArgsMode.FILE:
            return TrainingConfig._parse_and_validate_args_from_file(training_args_path)
        elif args_mode == project_config.ArgsMode.CMD_LINE:
            return TrainingConfig._parse_args_from_cmd_line()
        else:
            raise ValueError(f"Unknown mode: {args_mode}")

    @staticmethod
    def _parse_and_validate_args_from_file(training_args_path):
        if training_args_path is None:
            training_args_path = pkg_resources.resource_filename(__name__, 'resources/training_config/default_training_args.json')
        else:
            training_args_path = tools.parse_to_pathlib(training_args_path)
            assert training_args_path.is_file() and training_args_path.exists()
        
        logger.info(f'Loading args in Jupyter mode from {training_args_path}')

        with open(training_args_path, 'r') as file:
            training_config = json.load(file)

        # Validate using the argument parser
        parser = TrainingConfig._get_arg_parser()
        parsed_args = parser.parse_args([])
        for key, value in training_config.items():
            if hasattr(parsed_args, key):
                setattr(parsed_args, key, value)
            else:
                raise ValueError(f"Invalid argument '{key}' in configuration file")
        
        return parsed_args

    @staticmethod
    def _parse_args_from_cmd_line():
        logger.info(f'Loading args in script mode')
        parser = TrainingConfig._get_arg_parser()
        return parser.parse_args()

    @staticmethod
    def _get_arg_parser():
        parser = argparse.ArgumentParser(description="Training Configuration Parser")

        # Model name and save path, this will define names of save folders and files
        parser.add_argument("--results_save_path", type=str, default=Path(os.getcwd()), metavar='PATH', help="Path for creating model plots and checkpoint save folders")
        parser.add_argument("--model_name", type=str, default="alice_model", help="Name of model used for naming different experiments in same save path")
        parser.add_argument('--verbose', action='store_true', help="This will cause full training progress bars")
        parser.add_argument('--wandb_entity', type=str, default='Alice-ML', help="Setting this varible will cause to log wandb to specyfic team instead of default one")
        parser.add_argument('--wandb_project', type=str, default=None, help="Setting this varible will cause loggin all metrics and config to wandb. Before using it 'wandbe login' should be called")

        #Checkpoint load, is setuped to some model checkpoint it will resume training, but if seed is not set datasets will have different shuffle
        parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='Path to checkpoint (default: none)')

        # General arguments for running model
        parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
        parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training (default: 1024)")
        parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
        parser.add_argument("--evaluation_frequency", type=int, default=2, help="Frequency (in epochs) for evaluation (default: 2)")
        parser.add_argument("--checkpoint_frequency", type=int, default=2, help="Frequency (in epochs) for saving checkpoints (default: 2)")
        parser.add_argument("--train_valid_split", type=float, default=0.2, help="Train-validation split ratio (default: 0.2)")
        parser.add_argument(
            "--sub_valid_splits", 
            nargs="+",
            type=bool,
            default=[True],
            help="This option allow to divide valid dataset to len(sub_valid_splits) equal subsets, and based on bool values in array, only those with True will end in valid dataset.")
        parser.add_argument("--shuffle_before_split", type=bool, default=True, help="Shuffle data before splitting into train and validation sets (default: True)")
        parser.add_argument("--shuffle_after_split", type=bool, default=True, help="Shuffle data after splitting into train and validation sets (default: True)")
        
        # Optimizer and scheduler arguments for training 
        parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer (default: 0.001)")
        parser.add_argument("--optimizer", type=str, default="adam", choices=["adam"], help="Optimizer to use (default: adam)")
        parser.add_argument("--aggregator", type=str, default="upgrad", choices=["upgrad"], help="Aggregator to use (default: upgrad)")
        parser.add_argument("--scheduler", type=str, default="step", choices=["step"], help="Learning rate scheduler to use (default: step)")
        parser.add_argument("--scheduler_gamma", type=float, default=0.9, help="Gamma value for step scheduler (default: 0.9)")
        parser.add_argument("--scheduler_step", type=int, default=1, help="Step size for step scheduler (default: 1)")
        
        # Loss function and metrics
        parser.add_argument("--loss", type=str, default="huber", choices=["huber"], help="Loss function to use (default: huber)")
        parser.add_argument('--logspace_loss', action='store_true', help="This will cause to parse data to log space before calculating loss.")
        parser.add_argument("--loss_coeff", type=float, default=0.1, help="Loss coeff")
        parser.add_argument("--aux_loss_coeff", type=float, default=0.1, help="Aux loss coeff")
        parser.add_argument("--losses_count", type=int, default=1, help="Help argument, helpful with jacobian descent")
        parser.add_argument(
            "--metrics",
            nargs="+",
            type=str,
            choices=["mse", "rmse", "mae", "mape", "uep"],
            default=["rmse", "mae", "mape"],
            help="List of metrics to evaluate the model (default: ['mse', 'rmse', 'mae', 'mape', 'uep'])",
        )

        #dataset
        parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
        parser.add_argument('--data_path', type=str, default=None, metavar='PATH', help='Path to dataset folder', required=False)
        parser.add_argument('--processing_target', type=str, default=None, choices=["MLP", "MLP_EMBEDINGS", "TRANSFORMER", "TRANSFORMER_EMBEDINGS"], required=False)
        parser.add_argument('--considered_columns_config_path', type=str, default=None, required=False)
        parser.add_argument('--ohe_threshold_config_path', type=str, default=None, required=False)
        parser.add_argument('--filter_data_config_path', type=str, default=None, required=False)
        parser.add_argument(
            '--distinct_split_column',
            nargs="+",
            type=str,
            default=None,
            required=False,
            help="Comma-separated list of column names to use for distinct grouping"
        )
        parser.add_argument('--time_series_encoding', action='store_true')
        parser.add_argument('--sequence_grouping_column', 
                            nargs="+",
                            type=str,
                            default=None,
                            help="Columns by which data will be divided into groups to organize data into sequences. Could be list of str values or single str value")

        # distributed training, only for script use, setuped automaticly
        parser.add_argument('--multiprocessing_distributed', action='store_true')
        parser.add_argument('--gpu', type=int, default=0, help="The gpu on node")
        parser.add_argument('--rank', type=int, default=0, help="The rank of proces globaly")
        parser.add_argument('--world_size', type=int, default=0, help="The amount of gpus and nodes ~ gpu * nodes")
        parser.add_argument('--master_addr', type=int, default=0, help="Ip of master computing node")
        parser.add_argument('--master_port', type=int, default=0, help="Port of master computing node")
        parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

        #embeding config
        parser.add_argument('--embeding_reduction_const', type=int, default=10, help="Category amount of distinct values * by this const will be output embeding dim")
        parser.add_argument('--positional_encoding', type=str, default=None, choices=["sincos"], required=False)

        #Transformer config 
        parser.add_argument('--teacher_forcing', type=bool, default=True, help="The way transforemr will work, if true there will be used masking and will be prediction full sequence")
        parser.add_argument('--sequence_length', type=int, default=16, help="Length of sequence wihle processing in transformers")
        parser.add_argument('--sequence_window_length', type=int, default=2, help="Length of sequence window offset while reorganizing data into sequences")
        parser.add_argument('--encoder_block_num', type=int, default=4, help="Amount of encoder blocks")
        parser.add_argument('--attn_heads', type=int, help="Amount of encoder blocks")

        #MLP config
        parser.add_argument('--dropout_rate', type=float, default=0.3, help="Amount of encoder blocks")
        parser.add_argument('--output_dim', type=int, default=1, help="In case of regression it always should be 1")
        parser.add_argument('--monte_carlo_dropout', type=int, required=False, help="If set while validation and evaluation model will process batch many times with dropot on. This will allow to calculate mean (final result) and std (confidence) of model.")
        
        return parser