import os
import wandb

_WANDB_TRACKER = {}

@staticmethod 
def get_wandb_tracker():
    global _WANDB_TRACKER
    return _WANDB_TRACKER

@staticmethod
def setup_wandb(args):
    if args.rank == 0 and args.wandb_project:
        wandb_path = os.path.join(args.results_save_path / f"{args.model_name}_cache", "wandb")
        os.makedirs(wandb_path, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_path

        wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.model_name, config=vars(args))

@staticmethod
def finish_wandb(args):
    if args.rank == 0 and args.wandb_project:
        wandb.finish()

@staticmethod
def save_metric_to_wandb_tracker(name: str, metric: float):
    tracker = get_wandb_tracker()
    tracker.setdefault('metrics', {})
    if name not in tracker['metrics']:
        tracker.setdefault(name, 0)
    tracker['metrics'][name] = metric

@staticmethod
def clear_wandb_metrics_tracker():
    tracker = get_wandb_tracker()
    for name in tracker['metrics']:
        tracker['metrics'][name] = 0

@staticmethod
def track_wandb_metrics(args, epoch):
    if args.rank == 0 and args.wandb_project:
        tracker = get_wandb_tracker()

        for name, metric in tracker['metrics'].items():
            wandb.log({f"{name}": metric}, epoch)

        clear_wandb_metrics_tracker()

@staticmethod
def parse_to_wandb_table(y_true, y_pred):
    table = wandb.Table(columns=["y_true", "y_pred"])
    for true, pred in zip(y_true, y_pred):
        table.add_data(true, pred)
    return table

@staticmethod
def save_table_to_wandb_tracker(name, table):
    tracker = get_wandb_tracker()
    tracker.setdefault('tables', {})
    if name not in tracker['tables']:
        tracker.setdefault(name, wandb.Table(columns=["y_true", "y_pred"]))
    tracker['tables'][name] = table

@staticmethod
def clear_wandb_tables_tracker():
    tracker = get_wandb_tracker()
    for name in tracker['tables']:
        tracker['tables'][name] = wandb.Table(columns=["y_true", "y_pred"])

@staticmethod
def parse_and_save_to_wandb_tracker(name, y_true, y_pred):
    table = parse_to_wandb_table(y_true, y_pred)
    save_table_to_wandb_tracker(name, table)

@staticmethod
def track_wandb_tables(args):
    if args.rank == 0 and args.wandb_project:
        tracker = get_wandb_tracker()

        for name, table in tracker['tables'].items():
            wandb.log({name: wandb.plot.scatter(table, "y_true", "y_pred", title=name)})

        clear_wandb_tables_tracker()