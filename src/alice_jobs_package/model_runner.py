import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from torchjd.autojac._mtl_backward import mtl_backward
from torchjd.aggregation._upgrad import UPGrad

import time
import numpy
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.utils import logging, tools, wandb_logging, diagnostics
from alice_jobs_package.training import metrics
from alice_jobs_package.training.history import TrainingHistory, EvalHistory
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.training.losses import RubberHuberLoss, HuberUepSumLoss, HuberUepLoss, normalize_losses, normalize_loss
from alice_jobs_package.config_generator import AliceConfigGenerator
from alice_jobs_package.preprocessor import AliceDataPreprocessor
from alice_jobs_package.models.base_alice_model import BaseAliceModel

logger = logging.get_logger(__name__)

class AliceModelRunner():
    @staticmethod
    def model_provider(training_config : TrainingConfig, model_class : BaseAliceModel) -> BaseAliceModel:
        return model_class(training_config)
    
    @staticmethod
    def scheduler_provider(args, optimizer):
        if args.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        elif args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        else:
            logger.error(f'Scheduler config not recognized, implement to use {args.scheduler}')
            raise BaseException()

    @staticmethod
    def optimizer_provider(optimizer_type: str, *args, **kwargs):
        if optimizer_type == 'adam':
            return torch.optim.Adam(*args, **kwargs)
        else:
            logger.error(f'Optimzer config not recognized, implement to use {optimizer_type}')
            raise BaseException()
        
    @staticmethod
    def aggregator_provider(aggregator_type: str, *args, **kwargs):
        if aggregator_type == 'upgrad':
            return UPGrad(*args, **kwargs)
        else:
            logger.error(f'Aggregator config not recognized, implement to use {aggregator_type}')
            raise BaseException()
    
    @staticmethod
    def loss_provider(arguments, loss_type, *args, **kwargs):
        if loss_type == 'huber':
            return torch.nn.HuberLoss(delta = arguments.loss_coeff, *args, **kwargs)
        elif loss_type == 'rubber':
            return RubberHuberLoss(delta = arguments.loss_coeff, *args, **kwargs)
        elif loss_type == 'huber_uep_sum':
            return HuberUepSumLoss(arguments.loss_coeff, arguments.aux_loss_coeff, *args, **kwargs)
        elif loss_type == 'huber_uep':
            arguments.losses_count = 2
            return HuberUepLoss(arguments.loss_coeff, arguments.aux_loss_coeff, *args, **kwargs)
        else:
            logger.error(f'Loss config not recognized, implement to use {loss_type}')
            raise BaseException()
    
    @staticmethod
    def setup_metrics(metrics_list):
        metrics_ = {}
        for metric in metrics_list:
            if metric == 'mse':
                metrics_[metric] = metrics.mse
            elif metric == 'rmse':
                metrics_[metric] = metrics.rmse
            elif metric == 'mae':
                metrics_[metric] = metrics.mae
            elif metric == 'mape':
                metrics_[metric] = metrics.mape
            elif metric == 'uep':
                metrics_[metric] = metrics.uep
            elif metric == 'smape':
                metrics_[metric] = metrics.smape
            elif metric == 'r2_score':
                metrics_[metric] = metrics.r2_score
            elif metric == 'huber':
                metrics_[metric] = metrics.huber
            else:
                logger.error(f'Metric not recognized, {metric}')
        return metrics_
    
    @staticmethod
    def load_numpy_data(training_config : TrainingConfig):
        args = training_config.args

        processing_target = ProcessingTarget(args.processing_target)
        numerical_columns_config, categorical_columns_config = AliceConfigGenerator.generate_config(data_path=args.data_path, 
                                                                                                    considered_columns_config_path=args.considered_columns_config_path,
                                                                                                    ohe_threshold_config_path=args.ohe_threshold_config_path,
                                                                                                    verbose=args.verbose)
        
        filter_data_config = AliceConfigGenerator._load_filter_data_config(args.filter_data_config_path)

        alice_data_preprocessor = AliceDataPreprocessor(data_path = args.data_path,
                                                        processing_target = processing_target,
                                                        numerical_columns_config = numerical_columns_config,
                                                        categorical_columns_config = categorical_columns_config,
                                                        filter_data_config = filter_data_config) 
        
        X_, y_ = alice_data_preprocessor.preprocess(pandas = True, verbose=args.verbose)

        if True:
            n = int(0.1 * len(X_))
            X_ = X_.iloc[:n].reset_index(drop=True)
            y_ = y_.iloc[:n].reset_index(drop=True)

        training_config.setup_alicepreprocesing_config(alice_data_preprocessor)

        #Making data into sequences
        if processing_target in {ProcessingTarget.TRANSFORMER, ProcessingTarget.TRANSFORMER_EMBEDINGS}:
            X_, y_ = AliceModelRunner.parse_data_to_sequences(training_config, X_, y_)
        else:
            X_, y_ = X_.to_numpy(), y_.to_numpy()
        
        if args.time_series_encoding:
            AliceModelRunner.time_based_feature_encoding(training_config, X_)

        return AliceModelRunner.parse_numpy_to_torch(X_), AliceModelRunner.parse_numpy_to_torch(y_)

    @staticmethod
    def parse_data_to_sequences(training_config : TrainingConfig, X_, y_):
        sequence_column = training_config.args.sequence_grouping_column

        seq_length = training_config.args.sequence_length
        window_step = training_config.args.sequence_window_length

        X_sequences, y_sequences = [], []

        if sequence_column is not None:
            # Ensure the sequence grouping column is treated as an integer
            X_[sequence_column] = X_[sequence_column].astype(int)

            # Group data by the specified column
            grouped = X_.groupby(sequence_column)

            for _, group in tqdm(grouped):
                X_group = group.to_numpy()
                y_group = y_.iloc[group.index].to_numpy()

                if len(X_group) < seq_length:
                    continue  # Skip groups that are too short for sequence creation

                # Create sequences with sliding windows
                X_seq = sliding_window_view(X_group, (seq_length, X_group.shape[1]))[::window_step, 0, :, :]
                y_seq = sliding_window_view(y_group, (seq_length, y_group.shape[1]))[::window_step, 0, :, :]

                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        else:
            # Process the entire dataset as a single sequence
            X_, y_ = X_.to_numpy(), y_.to_numpy()

            if len(X_) >= seq_length:
                X_sequences = sliding_window_view(X_, (seq_length, X_.shape[1]))[::window_step, 0, :, :]
                y_sequences = sliding_window_view(y_, (seq_length, y_.shape[1]))[::window_step, 0, :, :]

        # Stack sequences if grouping was used
        if isinstance(X_sequences, list) and X_sequences:
            X_sequences = numpy.concatenate(X_sequences, axis=0)
            y_sequences = numpy.concatenate(y_sequences, axis=0)

        return numpy.array(X_sequences), numpy.array(y_sequences)

    @staticmethod
    def time_based_feature_encoding(training_config : TrainingConfig, X_seq):
        time_column_to_encode = "startedtimestamp"
        index_of_column_in_data = training_config.column_names.index(time_column_to_encode)
        one_day_in_ms = 86400 * 1000

        # Inverse processing to get timestamps
        if training_config.num_config is not None:
            mean = training_config.num_config[time_column_to_encode]['mean']
            std = training_config.num_config[time_column_to_encode]['std']
        else: 
            raise BaseException('Problem with loading config')

        # Extract the column to be inverted
        if std != 0:
            X_seq[:, :, index_of_column_in_data] *= std
            X_seq[:, :, index_of_column_in_data] += mean
        else:
            X_seq[:, :, index_of_column_in_data] += mean

        # Compute the differences
        diffs = numpy.expand_dims(X_seq[:, -1, index_of_column_in_data], axis=-1) - X_seq[:, :, index_of_column_in_data]

        # Apply threshold using np.maximum (faster than np.where)
        diffs = numpy.where(diffs > one_day_in_ms, one_day_in_ms, diffs)

        # Handle division safely (avoid division by zero)
        denominator = numpy.expand_dims(diffs[:, 0], axis=-1)
        denominator[denominator == 0] = 1.0  # In-place modification

        X_seq[:, :, index_of_column_in_data] = diffs / denominator
    
    @staticmethod
    def prepare_dataloaders(training_config: TrainingConfig, X_: numpy.ndarray | torch.Tensor, y_: numpy.ndarray | torch.Tensor, no_split = False):
        args = training_config.args
        X = AliceModelRunner.parse_numpy_to_torch(X_)
        y = AliceModelRunner.parse_numpy_to_torch(y_)
        dataset = TensorDataset(X, y)

        col = training_config.column_names.index("startedtimestamp")
        data = training_config.num_config["startedtimestamp"]
        std, mean = data['std'], data['mean']
        to_date = lambda v: datetime.fromtimestamp((v.item() * std + mean) / 1000).strftime("%Y-%m-%d %H:%M:%S")

        split_idx = int(len(X) * args.train_valid_split)
        # print("Data range:", to_date(X[0, col]), "|", to_date(X[split_idx, col]), "|", to_date(X[-1, col]))

        if args.seed:
            generator = torch.Generator().manual_seed(args.seed)
        else:
            generator = None

        if no_split:
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle_after_split, generator=generator)

        elif args.distinct_split_column is not None:

            start = time.time()
            # Normalize to list
            if isinstance(args.distinct_split_column, str):
                distinct_cols = [args.distinct_split_column]
            else:
                distinct_cols = list(args.distinct_split_column)

            # Column indices
            time_col_idx = training_config.column_names.index("startedtimestamp")
            group_col_indices = [
                training_config.column_names.index(c) for c in distinct_cols
            ]
            
            # Extract tensors
            group_matrix = X[:, group_col_indices]
            time_col = X[:, time_col_idx]

            unique_groups, group_ids = torch.unique(
                group_matrix,
                dim=0,
                return_inverse=True
            )
            num_groups = unique_groups.size(0)

            # Earliest timestamp per group
            min_time_per_group = torch.full(
                (num_groups,),
                float("inf"),
                device=X.device
            )
            min_time_per_group.scatter_reduce_(
                dim=0,
                index=group_ids,
                src=time_col,
                reduce="amin"
            )

            # Group sizes
            group_counts = torch.bincount(group_ids, minlength=num_groups)

            # Sort groups by earliest timestamp
            sorted_group_ids = torch.argsort(min_time_per_group)
            sorted_group_counts = group_counts[sorted_group_ids]

            # Select train groups by sample count
            train_target = int((1 - args.train_valid_split) * len(dataset))
            cumulative = torch.cumsum(sorted_group_counts, dim=0)
            cutoff_idx = torch.searchsorted(cumulative, train_target)

            train_group_ids = sorted_group_ids[:cutoff_idx + 1]

            # Build masks and datasets
            train_group_mask = torch.zeros(
                num_groups,
                dtype=torch.bool,
                device=X.device
            )
            train_group_mask[train_group_ids] = True
            train_mask = train_group_mask[group_ids]

            train_indices = train_mask.nonzero(as_tuple=True)[0]
            valid_indices = (~train_mask).nonzero(as_tuple=True)[0]

            train_dataset = Subset(dataset, train_indices.tolist())
            valid_dataset = Subset(dataset, valid_indices.tolist())

            diagnostics.post_split_diagnostics(
                train_dataset,
                valid_dataset,
                group_columns_indices=group_col_indices
            )

        else:
            # Default random or ordered split
            if args.shuffle_before_split:
                train_len = int((1 - args.train_valid_split) * len(dataset))
                valid_len = len(dataset) - train_len
                train_dataset, valid_dataset = random_split(
                    dataset, [train_len, valid_len], generator=generator
                )
            else:
                split_index = int((1 - args.train_valid_split) * len(dataset))
                train_dataset = Subset(dataset, list(range(split_index)))
                valid_dataset = Subset(dataset, list(range(split_index, len(dataset))))

        if args.sub_valid_splits is not None:
            total_size = len(valid_dataset)
            left_data_size = total_size
            left_splits = len(args.sub_valid_splits)
            selected_indices = []

            start = 0
            for flag in args.sub_valid_splits:
                sub_size = left_data_size // left_splits
                end = start + sub_size if left_splits != 1 else total_size
                indices = list(range(start, end))

                if flag:
                    selected_indices.extend(indices)

                start = end
                left_data_size -= sub_size
                left_splits -= 1

            valid_dataset = Subset(valid_dataset, selected_indices)

        if args.multiprocessing_distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=args.shuffle_after_split)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, sampler=train_sampler, generator=generator)

            valid_sampler = DistributedSampler(valid_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=args.shuffle_after_split, drop_last=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, sampler=valid_sampler, generator=generator)

        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_after_split, generator=generator)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.shuffle_after_split, generator=generator)

        return train_dataloader, valid_dataloader
    
    @staticmethod
    def train(training_config : TrainingConfig, model_class : BaseAliceModel):
        X_, y_ = AliceModelRunner.load_numpy_data(training_config)
        return AliceModelRunner.train_from_numpy_data(training_config, model_class, X_, y_)

    @staticmethod
    def evaluate(training_config : TrainingConfig, model_class : BaseAliceModel):
        X_, y_ = AliceModelRunner.load_numpy_data(training_config)
        return AliceModelRunner.evaluate_from_numpy_data(training_config, model_class, X_, y_)

    @staticmethod
    def train_from_numpy_data(training_config : TrainingConfig, model_class : BaseAliceModel, X : numpy.ndarray | torch.Tensor, y : numpy.ndarray | torch.Tensor):
        #SETUP ARGS
        args = training_config.args
        if args.verbose: logging.set_verbosity_info()

        #SETUP DATA
        train_dataloader, valid_dataloader = AliceModelRunner.prepare_dataloaders(training_config, X, y)

        start_epoch = 1
        device = training_config.device
        model = AliceModelRunner.model_provider(training_config, model_class).to(device)
        optimizer = AliceModelRunner.optimizer_provider(args.optimizer, model.parameters(), lr = args.learning_rate)
        aggregator = AliceModelRunner.aggregator_provider(args.aggregator)
        scheduler = AliceModelRunner.scheduler_provider(args, optimizer)
        loss_fn = AliceModelRunner.loss_provider(args, args.loss).to(device)
        metrics_ = AliceModelRunner.setup_metrics(args.metrics)
        train_history = TrainingHistory(metrics_)

        ### LOG AMOUNT OF PARAMETERS

        total_params = sum(p.numel() for p in model.parameters())  # All parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Only trainable parameters

        if training_config.args.rank == 0:
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
        
        #SETUP MODEL AND HISTORY
        if args.resume and args.resume.is_file():
            checkpoint = AliceModelRunner.load_checkpoint(args, filepath = args.resume)
            start_epoch = checkpoint['epoch'] + 1 #Becouse checkpoint['epoch'] is number of done epochs
            train_history.load_history_from_json(checkpoint['history'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            aggregator.load_state_dict(checkpoint['aggregator'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            logger.info(f"Checkpoint loaded, continiue training from epoch {start_epoch}")

        #TRAINING
        for epoch in range(start_epoch, args.epochs + 1):
            AliceModelRunner.train_one_epoch(training_config, epoch, model, train_dataloader, optimizer, aggregator, scheduler, loss_fn, metrics_, train_history, device)

            #VALIDATION
            if args.evaluation_frequency != 0 and epoch % args.evaluation_frequency == 0:
                AliceModelRunner.validate_while_training(training_config, epoch, model, valid_dataloader, loss_fn, metrics_, train_history, device)
            
            #CHECKPOINT MODEL AND HISTORY OR SAVE LAST STATE
            if (args.checkpoint_frequency != 0 and epoch % args.checkpoint_frequency == 0) or epoch == args.epochs:
                AliceModelRunner.checkpoint_while_training(training_config, epoch, train_history, model, optimizer, aggregator, scheduler)

            #LOGIN WANDB
            wandb_logging.track_wandb_metrics(args, epoch)

        filename = f'{training_config.args.model_name}'
        history_plot_filename = training_config.plot_save_path / (filename + '_train_plot.png')
        train_history.plot_history(history_plot_filename, plot_train=True, plot_valid=True, evaluation_frequency=args.evaluation_frequency)

        #NEXT EVALUATE BOOTH TRAINLOADERS AND RETURN EVAL HISTORY
        eval_train_history = AliceModelRunner.evaluate_from_dataloader(training_config, model, train_dataloader, loss_fn, metrics_, 'eval_train')
        eval_valid_history = AliceModelRunner.evaluate_from_dataloader(training_config, model, valid_dataloader, loss_fn, metrics_, 'eval_valid')
        wandb_logging.track_wandb_tables(args)
        
        #CLOSE WANDB
        wandb_logging.finish_wandb(args)

        return train_history, eval_train_history, eval_valid_history

    @staticmethod
    def train_one_epoch(training_config : TrainingConfig, epoch : int, model : BaseAliceModel | DistributedDataParallel, dataloader, optimizer, aggregator, scheduler, loss_fn, metrics_, history : TrainingHistory, device):
        args = training_config.args
        epoch_start_time = time.time()

        model.train()
        running_epoch_losses = torch.zeros(args.losses_count)
        batch_count = 0

        all_outputs = []
        all_labels = []

        with tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120, disable=not args.verbose) as pbar:
            for inputs, labels in pbar:

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if not training_config.args.teacher_forcing:
                    labels = labels[:, -1, :]
                
                outputs = model(inputs).view(-1, 1)
                labels = labels.view(-1, 1)

                if args.logspace_loss:
                    outputs = torch.log1p(outputs)
                    labels = torch.log1p(labels)

                loss = loss_fn(outputs, labels)
                
                optimizer.zero_grad()
                
                if args.losses_count == 1:
                    loss.backward()
                else:
                    mtl_backward(losses=loss, features=outputs, aggregator=aggregator)

                optimizer.step()

                all_outputs.append(outputs.clone().detach())
                all_labels.append(labels.clone().detach())

                loss = normalize_loss(loss)
                running_epoch_losses += loss.clone().detach().cpu()
                batch_count += 1

                postfix = {
                    f"avg_loss_{i}": (running_epoch_losses[i] / batch_count).item()
                    for i in range(args.losses_count)
                }

                pbar.set_postfix(postfix)

        epoch_end_time = time.time()
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        all_outputs = torch.cat(all_outputs, dim=0)
        all_outputs = AliceModelRunner.gather_all(training_config, all_outputs).cpu()

        all_labels = torch.cat(all_labels, dim=0)
        all_labels = AliceModelRunner.gather_all(training_config, all_labels).cpu()

        if args.rank == 0:
            ### SAVE TO HISTORY
            history.lr.append(current_lr)
            wandb_logging.save_metric_to_wandb_tracker('learning_rate', current_lr)

            losses_values = loss_fn(all_outputs, all_labels).clone().detach()
            losses_values = normalize_losses(losses_values)
            history.loss_train.append([loss.item() for loss in losses_values])

            for i in range(args.losses_count):
                wandb_logging.save_metric_to_wandb_tracker(
                    f"loss_{i}",
                    losses_values[i].item(),
                )
    
            for metric_name, metric_fn in metrics_.items():
                getattr(history, f'{metric_name}_train').append(metric_fn(all_outputs, all_labels).item())
                wandb_logging.save_metric_to_wandb_tracker(f'{metric_name}_train', metric_fn(all_outputs, all_labels).item())

            ###LOG train metrics
            metrics_log = ''
            for metric_name in metrics_.keys():
                metrics_log += f"{metric_name}: {getattr(history, f'{metric_name}_train')[-1]:.4f} | "

            loss_log = " | ".join(
                f"Loss_{i}: {v:.4f}"
                for i, v in enumerate(history.loss_train[-1])
            )

            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Time: {epoch_end_time - epoch_start_time:.2f} s | "
                f"LR: {current_lr:.6f} | "
                + loss_log
                + " | "
                + metrics_log
            )
                
    @staticmethod
    def validate_while_training(training_config : TrainingConfig, epoch : int, model : BaseAliceModel | DistributedDataParallel, dataloader, loss_fn, metrics_, history : TrainingHistory, device):
        args = training_config.args
        if args.rank == 0: logger.info('')
        valid_start_time = time.time()

        model.eval()
        if args.monte_carlo_dropout: 
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.set_dropout_to_train()
            else:
                model.set_dropout_to_train()

        running_epoch_val_losses = torch.zeros(args.losses_count)
        val_batch_count = 0

        val_all_outputs = []
        val_all_uncertainty = []
        val_all_labels = []
        
        with torch.no_grad():
            with tqdm(dataloader, desc=f"Validating after epoch {epoch}", ncols=120, disable=not args.verbose) as pbar:
                for val_inputs, val_labels in pbar:

                    val_inputs = val_inputs.to(device, non_blocking=True)
                    val_labels = val_labels.to(device, non_blocking=True)

                    if not training_config.args.teacher_forcing:
                        val_labels = val_labels[:, -1, :]
                    
                    if args.monte_carlo_dropout:
                        val_outputs, val_uncertainty = AliceModelRunner.monte_carlo_dropout(args, val_inputs, model)
                    else:
                        val_outputs = model(val_inputs).view(-1, 1)
                        val_uncertainty = torch.zeros_like(val_outputs)
                    
                    val_labels = val_labels.view(-1, 1)

                    val_all_outputs.append(val_outputs)
                    val_all_uncertainty.append(val_uncertainty)
                    val_all_labels.append(val_labels)

                    loss = normalize_loss(loss_fn(val_outputs, val_labels))
                    running_epoch_val_losses += loss.clone().detach().cpu()
                    val_batch_count += 1

                    postfix = {
                        f"avg_loss_{i}": (running_epoch_val_losses[i] / val_batch_count).item()
                        for i in range(args.losses_count)
                    }

                    pbar.set_postfix(postfix)
            
            valid_end_time = time.time()

            ### GATHER ALL OUTPUTS AND LABELS
            val_all_outputs = torch.cat(val_all_outputs, dim=0)
            val_all_outputs = AliceModelRunner.gather_all(training_config, val_all_outputs).cpu()

            val_all_uncertainty = torch.cat(val_all_uncertainty, dim=0)
            val_all_uncertainty = AliceModelRunner.gather_all(training_config, val_all_uncertainty).cpu()
            
            val_all_labels = torch.cat(val_all_labels, dim=0)
            val_all_labels = AliceModelRunner.gather_all(training_config, val_all_labels).cpu()
            
            ### SAVE TO HISTORY
            if args.rank == 0:
                losses_values = loss_fn(val_all_outputs, val_all_labels).clone().detach()
                losses_values = normalize_losses(losses_values)
                history.loss_valid.append([loss.item() for loss in losses_values])

                for i in range(args.losses_count):
                    wandb_logging.save_metric_to_wandb_tracker(
                        f"loss_{i}_valid",
                        losses_values[i].item(),
                    )

                history.uncertainty_valid.append(val_all_uncertainty.mean().item())
                wandb_logging.save_metric_to_wandb_tracker('uncertainty_valid', val_all_uncertainty.mean().item())

                for metric_name, metric_fn in metrics_.items():
                    getattr(history, f'{metric_name}_valid').append(metric_fn(val_all_outputs, val_all_labels).item())
                    wandb_logging.save_metric_to_wandb_tracker(f'{metric_name}_valid', metric_fn(val_all_outputs, val_all_labels).item())
            
                ### LOG METRICS
                metrics_log = ''
                for metric_name in metrics_.keys():
                    metrics_log += f'{metric_name}: {getattr(history, f'{metric_name}_valid')[-1]:.4f} | '

                loss_log = " | ".join(
                    f"Loss_{i}: {v:.4f}"
                    for i, v in enumerate(history.loss_valid[-1])
                )

                logger.info(
                    f"Validation after epoch {epoch} | "
                    f"Time: {valid_end_time - valid_start_time:.2f} s | "
                    + loss_log
                    + " | "
                    + f"Uncertainty: {(history.uncertainty_valid[-1]):.4f}"
                    + " | "
                    + metrics_log
                )

        if args.rank == 0: logger.info('')

    @staticmethod
    def checkpoint_while_training(training_config : TrainingConfig, epoch : int, history : TrainingHistory, model : BaseAliceModel | DistributedDataParallel, optimizer, aggregator, scheduler):
        filename = f'{training_config.args.model_name}_checkpoint_epoch_{epoch}.pth' if epoch != training_config.args.epochs else f'{training_config.args.model_name}.pth'
        model_filename = training_config.model_save_path / filename
        hisotry_filename = tools.ensure_extension((training_config.history_save_path / filename), '.json')

        history.save_history_to_file(hisotry_filename)
        AliceModelRunner.save_checkpoint({
            'epoch': epoch,
            'history': history.dump_history_to_json(),
            'model': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'aggregator' : aggregator.state_dict(),
            'scheduler' : scheduler.state_dict(),
        },
        filepath = model_filename)

        logger.info(f"Checkpoint after epoch {epoch}, {filename}\n")

    @staticmethod
    def evaluate_from_numpy_data(training_config : TrainingConfig, model_class : BaseAliceModel, X, y, checkpoint_path = None, filename_sufix : str = 'eval') -> EvalHistory:
        args = training_config.args

        device = training_config.device

        if checkpoint_path is None:
            checkpoint_path = training_config.model_save_path / f'{args.model_name}.pth'
            logger.info(f'Checkpoint from {checkpoint_path} will be loaded')
            if not checkpoint_path.is_file():
                raise Exception(f'Passed checkpoint_path does not exists and there is no saved final model, pass checkpoint_path')
        
        dataloader = AliceModelRunner.prepare_dataloaders(training_config, X, y, no_split = True)
        model = AliceModelRunner.model_provider(training_config, model_class).to(device)

        checkpoint = AliceModelRunner.load_checkpoint(args, filepath = checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        loss_fn = AliceModelRunner.loss_provider(args, args.loss).to(device)
        metrics_ = AliceModelRunner.setup_metrics(args.metrics)

        return AliceModelRunner.evaluate_from_dataloader(training_config, model, dataloader, loss_fn, metrics_, filename_sufix)
    
    @staticmethod
    def evaluate_from_dataloader(training_config : TrainingConfig, model : BaseAliceModel | DistributedDataParallel, dataloader, loss_fn, metrics_, filename_sufix : str = 'eval') -> EvalHistory:
        args = training_config.args
        device = training_config.device
        eval_start_time = time.time()

        model.eval()
        if args.monte_carlo_dropout: 
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.set_dropout_to_train()
            else:
                model.set_dropout_to_train()
        
        eval_all_outputs = []
        eval_all_uncertainty = []
        eval_all_labels = []
        
        eval_hisotry = EvalHistory(metrics_)

        with torch.no_grad():
            with tqdm(dataloader, desc=f"Evaluating", ncols=120, disable= not args.verbose) as pbar:
                for val_inputs, val_labels in pbar:

                    val_inputs = val_inputs.to(device, non_blocking=True)
                    val_labels = val_labels.to(device, non_blocking=True)

                    if args.monte_carlo_dropout:
                        val_outputs, val_uncertainty = AliceModelRunner.monte_carlo_dropout(args, val_inputs, model)
                    else:
                        val_outputs = model(val_inputs).view(-1, 1)
                        val_uncertainty = torch.zeros_like(val_outputs)

                    val_labels = val_labels.view(-1, 1)

                    eval_all_outputs.append(val_outputs)
                    eval_all_uncertainty.append(val_uncertainty)
                    eval_all_labels.append(val_labels)
            
            eval_end_time = time.time()

            ### GATHER ALL OUTPUTS AND LABELS
            eval_all_outputs = torch.cat(eval_all_outputs, dim=0)
            eval_all_outputs = AliceModelRunner.gather_all(training_config, eval_all_outputs).cpu()

            eval_all_uncertainty = torch.cat(eval_all_uncertainty, dim=0)
            eval_all_uncertainty = AliceModelRunner.gather_all(training_config, eval_all_uncertainty).cpu()
            
            eval_all_labels = torch.cat(eval_all_labels, dim=0)
            eval_all_labels = AliceModelRunner.gather_all(training_config, eval_all_labels).cpu()

            if args.rank == 0:
                #Calcualte metrics
                eval_hisotry.y_pred = eval_all_outputs.squeeze(1).tolist()
                eval_hisotry.y_uncertainty = eval_all_uncertainty.squeeze(1).tolist()
                eval_hisotry.y_true = eval_all_labels.squeeze(1).tolist()

                wandb_logging.parse_and_save_to_wandb_tracker(filename_sufix, eval_all_labels.squeeze(1).tolist(), eval_all_outputs.squeeze(1).tolist())
                losses_values = torch.tensor(loss_fn(eval_all_outputs, eval_all_labels))
                losses_values = normalize_losses(losses_values)
                eval_hisotry.loss_eval.append([loss.item() for loss in losses_values])
                eval_hisotry.uncertainty_eval.append(eval_all_uncertainty.mean().item())

                for metric_name, metric_fn in metrics_.items():
                    getattr(eval_hisotry, f'{metric_name}_eval').append(metric_fn(eval_all_outputs, eval_all_labels).item())

                ### LOG gathered valid metrics on rank 0
                metrics_log = ''
                for metric_name in metrics_.keys():
                    metrics_log += f'{metric_name}: {getattr(eval_hisotry, f'{metric_name}_eval')[-1]:.4f} | '

                loss_log = " | ".join(
                    f"Loss_{i}: {v:.4f}"
                    for i, v in enumerate(eval_hisotry.loss_eval[-1])
                )

                logger.info(
                    f"Evaluation | "
                    f"Time: {eval_end_time - eval_start_time:.2f} s | "
                    + loss_log
                    + " | "
                    + f"Uncertainty: {(eval_hisotry.uncertainty_eval[-1]):.4f}"
                    + " | "
                    + metrics_log
                )

                #SAVE HISTORY TO FILE IN PLOT FOLDER AS ALSO SAVE PLOT
                eval_hisotry_path = training_config.history_save_path / (f'{training_config.args.model_name}_' + filename_sufix + '_hisotry')
                eval_hisotry.save_history_to_file(eval_hisotry_path)
                eval_plot_path = training_config.plot_save_path / (f'{training_config.args.model_name}_' + filename_sufix + '_plot')
                eval_hisotry.plot_history(eval_plot_path)

        return eval_hisotry

    @staticmethod
    def save_checkpoint(state, filepath : Path):
        torch.save(state, filepath)
    
    @staticmethod
    def load_checkpoint(args, filepath : Path):
        if args.multiprocessing_distributed:
            loc = 'cuda:{}'.format(args.gpu)
            return torch.load(args.resume, map_location=loc, weights_only=True)
        else:
            checkpoint = torch.load(filepath, weights_only=True)
            if any(True if "module." in key else False for key in checkpoint['model'].keys()): #Loading distributed model to classical model 
                model = checkpoint["model"]
                parsed_model = {k.replace("module.", ""): v for k, v in model.items()}
                checkpoint['model'] = parsed_model
            return checkpoint

    @staticmethod
    def parse_numpy_to_torch(data: torch.Tensor | numpy.ndarray) -> torch.Tensor:
        if isinstance(data, numpy.ndarray):
            return torch.Tensor(data)
        elif isinstance(data, torch.Tensor):
            return data
        else:
            logger.warning("Passed data is not numpy ndarray, data not passed")
            return torch.Tensor()
    
    @staticmethod
    def monte_carlo_dropout(args, model_input : torch.Tensor, model : BaseAliceModel | DistributedDataParallel):
        predictions = []

        for _ in range(args.monte_carlo_dropout):
            preds = model(model_input).view(-1, 1)
            predictions.append(preds)

        predictions = torch.stack(predictions, dim=0)

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std
        
    ### MULTI GPU

    @staticmethod
    def train_distributed(training_config : TrainingConfig, model_class : BaseAliceModel):
        logger.info("ARGS on rank", training_config.args.rank, training_config.args)
        args = training_config.args

        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank, init_method=args.init_method)
        if args.verbose: logging.set_verbosity_info()

        train_dataloader, valid_dataloader = AliceModelRunner.prepare_distributed_dataloader(training_config)

        # CREATE MODEL AND OTHER STRUCTURES
        torch.cuda.set_device(args.gpu)

        start_epoch = 1
        device = training_config.device
        model = AliceModelRunner.model_provider(training_config, model_class).cuda(args.gpu)
        optimizer = AliceModelRunner.optimizer_provider(args.optimizer, model.parameters(), lr = args.learning_rate)
        aggregator = AliceModelRunner.aggregator_provider('upgrad')
        scheduler = AliceModelRunner.scheduler_provider(args, optimizer)
        loss_fn = AliceModelRunner.loss_provider(args, args.loss).to(device)
        metrics_ = AliceModelRunner.setup_metrics(args.metrics)
        train_history = TrainingHistory(metrics_)

        model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        #SETUP MODEL AND HISTORY
        if args.resume and args.resume.is_file():
            checkpoint = AliceModelRunner.load_checkpoint(args, filepath = args.resume)
            start_epoch = checkpoint['epoch'] + 1 #Becouse checkpoint['epoch'] is number of done epochs
            train_history.load_history_from_json(checkpoint['history'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            aggregator.load_state_dict(checkpoint['aggregator'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            logger.info(f"Checkpoint loaded, continiue training from epoch {start_epoch}")

        #TRAINING
        for epoch in range(start_epoch, args.epochs + 1):
            AliceModelRunner.train_one_epoch(training_config, epoch, model, train_dataloader, optimizer, aggregator, scheduler, loss_fn, metrics_, train_history, device)

            #VALIDATION
            if args.evaluation_frequency != 0 and epoch % args.evaluation_frequency == 0:
                AliceModelRunner.validate_while_training(training_config, epoch, model, valid_dataloader, loss_fn, metrics_, train_history, device)
            
            #CHECKPOINT MODEL AND HISTORY OR SAVE LAST STATE ONLY ON RANK 0
            if args.rank == 0 and ((args.checkpoint_frequency != 0 and epoch % args.checkpoint_frequency == 0) or epoch == args.epochs):
                AliceModelRunner.checkpoint_while_training(training_config, epoch, train_history, model, optimizer, aggregator, scheduler)

            #LOGIN WANDB
            wandb_logging.track_wandb_metrics(args, epoch)

        if args.rank == 0:
            filename = f'{training_config.args.model_name}'
            history_plot_filename = training_config.plot_save_path / (filename + '_train_plot.png')
            train_history.plot_history(history_plot_filename, plot_train=True, plot_valid=True, evaluation_frequency=args.evaluation_frequency)

        #NEXT EVALUATE BOOTH TRAINLOADERS AND RETURN EVAL HISTORY
        AliceModelRunner.evaluate_from_dataloader(training_config, model, train_dataloader, loss_fn, metrics_, 'eval_train')
        AliceModelRunner.evaluate_from_dataloader(training_config, model, valid_dataloader, loss_fn, metrics_, 'eval_valid')
        wandb_logging.track_wandb_tables(args)

        #CLOSE WANDB
        wandb_logging.finish_wandb(args)

        pass

    @staticmethod
    def prepare_distributed_dataloader(training_config : TrainingConfig):
        logger.info(f"Loading Data")

        X_, y_ = AliceModelRunner.load_numpy_data(training_config)

        logger.info(f"Data loaded, preparing dataset")

        train_dataloader, valid_dataloader = AliceModelRunner.prepare_dataloaders(training_config, X_, y_)

        return train_dataloader, valid_dataloader
    
    @staticmethod
    def gather_all(training_config : TrainingConfig, tensor : torch.Tensor):
        args = training_config.args

        if args.rank == 0 and args.multiprocessing_distributed:
            gathered_data = [torch.empty_like(tensor, device=training_config.device) for _ in range(args.world_size)]
            dist.gather(tensor, gather_list=gathered_data, dst=0)
            gathered_tensor = torch.cat(gathered_data, dim=0)
            return gathered_tensor
        
        if args.multiprocessing_distributed:
            dist.gather(tensor, dst=0)
            
        return tensor

    @staticmethod
    def reduce_all(training_config : TrainingConfig, tensor : torch.Tensor):
        if training_config.args.multiprocessing_distributed:
            total = torch.tensor(tensor, device=training_config.device)
            dist.all_reduce(total, dist.ReduceOp.SUM)
        return total