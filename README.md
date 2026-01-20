# Alice Jobs Training and Preprocessing Package

This package enables users to preprocess CERN ALICE jobs data and use it for training machine learning models based on PyTorch. The package can be used both in Jupyter notebooks and script-based workflows. However, multi-GPU training is available only in the script-based approach.

## Instalation

The project is available on test.pypi.org and can be installed using the following pip install command:

`pip install -i https://test.pypi.org/simple/ alice-jobs-package`

As the package evolves rapidly, it is recommended to use the `-U` flag to ensure you always have the latest version. If stability is critical for use in scripts (e.g., SLURM sbatch scripts), you can install a specific version as shown below:

`pip install -i https://test.pypi.org/simple/ alice-jobs-package==1.2.7`

## Usage

The package comprises four main components:

### 1.	Data Loader
This component loads, parses, and cleans raw data into a single table, which can then be further processed using other parts of the package. Note that the data itself is not publicly available. The script used to download the data from our internal copy is located in src/tools, but it requires proper credentials, which are available upon request.

```python
from alice_jobs_package.data_loader import AliceDataLoader

data_path = 'path_to_raw_data_folder'
ignore_cache = False
verbose = True

raw_data = AliceDataLoader.load_data(data_path = args.data_path, 
                                    ignore_cache = ignore_cache, 
                                    verbose = verbose)
```

In data_path there should be all needed data in format:
- job_info.csv
- mon_jdls.csv
- trace.csv
- site sonar (folder of files with site-sonar-`timestamp`.out.xz naming)

This process generates three types of files:
1.	joined_site_sonar.npz: Contains the combined data from all site-sonar-<timestamp>.out.xz files.
2.	joined_data.npz: Contains the properly joined and cleaned data derived from the individual input files.
3.	output_data.npz: Contains the processed version of joined_data augmented with aggregated metrics, and is ready for use in subsequent processing steps.

This code currently assumes that all raw files may contain new values but not exacly new categories. However, if new categories do appear, they must be added to the alice_jobs_package/resources/dtypes files. This ensures that the data is correctly typed. Without this declaration, the data might be improperly recognized.

### 2.	Config Generator
Based on the data, this component generates a configuration file. This is necessary because new records may introduce additional columns, requiring updates to the configuration. The configuration maintains the context of the data, enabling the mapping of raw data into specific columns and attributes (referred to as dimensions).

```python
from alice_jobs_package.utils.project_config import *
from alice_jobs_package.config_generator import AliceConfigGenerator

data_path = 'path_to_raw_data_folder'
ignore_cache = False
verbose = True

considered_columns_path = 'path_considered_columns_config'
ohe_threshold_path = 'path_to_ohe_config'

numerical_columns_config, categorical_columns_config = \
AliceConfigGenerator.generate_config(data_path = data_path,
                                    considered_columns = considered_columns_path,
                                    ohe_threshold_config = ohe_threshold_path,
                                    ignore_cache = ignore_cache, 
                                    verbose = verbose)
```

This script relies on two configurations. The first is a dictionary of columns that should be included in the processed output data. Depending on the list where a column is specified, it will be treated as either a categorical or numerical feature. Any other columns from the raw data will be omitted.

The package includes a baseline file with predefined columns in the resources directory. If no file is provided, this baseline file will be used by default.
```json
{
  "categories_columns": [
    "column1"
  ],
  "numerical_columns": [
    "column3"
  ],
}
```

The second file also has a baseline in the resources directory, it is used only in case of One Hot Encoding. It defines the threshold for applying a limit on the number of distinct values allowed for a single processed column (rest will be set as unknown). The structure of this file is as follows:

```json
{
    "threshold_limit" : 100,
    "threshold_value": 20
}
```

The output consists of two configuration files that can be used as input to the preprocessor in the next step.

Additional comments:
1.	Before this step, there is no need to run the Data Loader separately, as it will be executed automatically within the generate_config() function.
2.	Distinct configurations will create separate cache folders, which can be shared and used by multiple users, provided they use the same raw_data_path.

### 3.	Preprocessor
This component processes the data using the generated configuration, transforming it into a format ready for training. Currently, two approaches are supported: one-hot encoding and embeddings.

```python
from alice_jobs_package.utils.project_config import *
from alice_jobs_package.preprocessor import AliceDataPreprocessor

data_path = 'path_to_raw_data_folder'
processing_target = ProcessingTarget.MLP
pandas = True
ignore_cache = False
verbose = True
numerical_columns_config, categorical_columns_config = None, None # From Config generator

alice_data_preprocessor = AliceDataPreprocessor(data_path = data_path,
                                                processing_target = processing_target,
                                                numerical_columns_config = numerical_columns_config,
                                                categorical_columns_config = categorical_columns_config) 

X, y = alice_data_preprocessor.preprocess(pandas = True,
                                          ignore_cache = ignore_cache, 
                                          verbose = verbose)                                                
```

This code snippet, based on previously generated configurations, preprocesses data for one of four targets:

```python
class ProcessingTarget(Enum):
    MLP = 'MLP'
    MLP_EMBEDINGS = 'MLP_EMBEDINGS'
    TRANSFORMER = 'TRANSFORMER'
    TRANSFORMER_EMBEDINGS = 'TRANSFORMER_EMBEDINGS'
```

As output we got two tables data and labels, in format pandas or numpy.

### 4.	Model Runner
This component builds, trains, and validates the model during training, and finally evaluates it.
Validation, in this context, differs from evaluation: validation is performed on test data during training, while evaluation is done on any dataset using the finalized model after training.

Example of usage in jupyter notebook:

```python
from alice_jobs_package.utils.project_config import *
from alice_jobs_package.model_runner import AliceModelRunner
from alice_jobs_package.models.base_mlp import BaseMLP

training_config = TrainingConfig(args_mode = ArgsMode.FILE, training_args_path = None)

train_history, eval_train_history, eval_valid_history = AliceModelRunner.train_from_numpy_data(training_config, BaseMLP, X_, y_)
```

Example of usage in script

```bash
python ./script.py --training_args_mode FILE
```

```python
import argparse

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.model_runner import AliceModelRunner
from alice_jobs_package.models.base_mlp import BaseMLP

def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--training_args_mode', type=str, default="FILE", choices=["FILE", "CMD_LINE"])
    args = parser.parse_args()

    training_args_mode = ArgsMode(args.training_args_mode)
    training_config = TrainingConfig(args_mode = training_args_mode)
    
    train_history, eval_train_history, eval_valid_history = AliceModelRunner.train(training_config, BaseMLP)

if __name__ == '__main__':
    main()
```

In the case of single-GPU training, the output is the training history, which can be used to create plots of metrics and regression results. I recommend studying alice_jobs_package/training/history.py. An important note is that evaluation history differs from training history, but both are generated automatically in the model dictionary.

Example of distributed training
What about multi-GPU training? Data Parallelism (DP) is supported during training, and initial testing has demonstrated an almost linear speedup when utilizing this approach.

```bash
DISTRIBUTED_ARGS="
    --nnodes $SLURM_NNODES \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --rdzv_endpoint $head_node_ip:$rdvz_port 
    --rdzv_id $SLURM_JOB_ID 
    --rdzv-backend c10d
"

srun torchrun $DISTRIBUTED_ARGS ./script.py --training_args_mode FILE
```

```python
import argparse

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.model_runner import AliceModelRunner
from alice_jobs_package.models.base_mlp import BaseMLP

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--training_args_mode', type=str, default="FILE", choices=["FILE", "CMD_LINE"])

def main():
    args = parser.parse_args()
    training_args_mode = ArgsMode(args.training_args_mode)
    training_config = TrainingConfig(args_mode = training_args_mode, training_args_path = '/net/pr2/projects/plgrid/plggalice_ai/tools/alice_research/5_Training/Single_MultiGPU_sbatchOnly/multigpu_training_args.json')

    AliceModelRunner.train_distributed(training_config, BaseMLP)

if __name__ == '__main__':
    main()
```

In the distributed (multi-GPU) case, nothing is returned, but the same plots are created.

To fully understand the training arguments and configuration, refer to alice_jobs_package/training/config.py.

## Experiments (repository scripts)

This repository also contains runnable experiment folders under `experiments/` (separate from the installable package code in `src/alice_jobs_package/`).

### Base model experiments (`experiments/base_models/*`)

There are three “dataset variants”, each with its own considered-columns file:

- **alitrain**: `experiments/base_models/alitrain/considered_columns_alitrain.json`
- **aliprod**: `experiments/base_models/aliprod/considered_columns_aliprod.json`
- **all**: `experiments/base_models/all/considered_columns_all.json`

Each variant contains an `MLP_EMB/mlp_emb_512n/` experiment with:

- **entrypoint**: `alice_torchrun.py`
- **model**: `model_mlp_emb_512.py` (exports `MLPEmbedded512`)
- **example configs**: `experiment_*/considered_columns_config.json`, `filter_data_config.json`, `ohe_threshold_config.json`, `training_config.json`

### Meta-learning experiments (“MAML folder”) (`experiments/meta_learning/*`)

The scripts in `experiments/meta_learning/` implement meta-learning / online few-shot variants of the same `MLPEmbedded512` model interface. **Each of these files can be used as a drop-in replacement for the base-model `model_mlp_emb_512.py`** used in:

- `experiments/base_models/alitrain/MLP_EMB/mlp_emb_512n/`
- `experiments/base_models/aliprod/MLP_EMB/mlp_emb_512n/`
- `experiments/base_models/all/MLP_EMB/mlp_emb_512n/`

#### Drop-in replacement: how to use

The base entrypoint imports locally:

- `from model_mlp_emb_512 import MLPEmbedded512` (see `alice_torchrun.py`)

So the simplest way to swap models is:

1. Pick the dataset variant (`alitrain`, `aliprod`, or `all`) and keep using its **existing configs**, especially the same considered-columns JSON (see paths above).
2. In the chosen `.../MLP_EMB/mlp_emb_512n/` directory, replace the model implementation by copying one of:
   - `experiments/meta_learning/model_mlp_emb_512_reptile_with_few_shot_adaptation.py`
   - `experiments/meta_learning/model_mlp_emb_512_reptile_with_few_shot_adaptation_hostname.py`
   - `experiments/meta_learning/model_mlp_emb_512_reptile_with_few_shot_adaptation_update_global_expert.py`
   - `experiments/meta_learning/model_mlp_emb_512_MAML_reptile_update_gradient.py`
3. Name the copied file **exactly** `model_mlp_emb_512.py` so `alice_torchrun.py` keeps working unchanged.

Notes:

- **Considered columns**: these meta-learning models still consume `training_config.column_names` produced by the same preprocessing pipeline, so you should reuse the same considered-columns file for the dataset variant you’re running.
- **Meta-grouping columns**:
  - The default variants group by `lpmjobtypeid`.
  - The `*_hostname.py` variant prefers the `hostname` column (and falls back to `lpmjobtypeid` if `hostname` is not present).

#### `model_mlp_emb_512_MAML_reptile_update_gradient.py`: switching “MAML mode” with `second_order`

In `experiments/meta_learning/model_mlp_emb_512_MAML_reptile_update_gradient.py`, the training step is **MAML**, and the `second_order` flag controls whether you run full second-order MAML or a reptile:

- **`second_order: true`** → full (second-order) MAML (`create_graph=True`)
- **`second_order: false`** → first-order/ Reptile (`create_graph=False`)

If you want a **Reptile-style** meta-update, use the `model_mlp_emb_512_reptile_with_few_shot_adaptation*.py` scripts instead (they implement the Reptile delta update).

#### Few-shot adaptation parameters (online adaptation)

All meta-learning models expose a small set of hyperparameters controlling the **online few-shot adaptation** behaviour during evaluation/inference. They are provided via `training_config.args` and have sensible defaults:

- **`online_max_support`**: maximum number of support samples kept **per group** in the in-memory buffer (job type or hostname, depending on the script). New samples overwrite the oldest ones once this limit is reached.  
  - Higher → longer “memory” of the recent behaviour for that group, more stable but slightly slower adaptation.
  - Lower → more aggressive focus on very recent behaviour.

- **`online_min_support`**: minimum number of buffered samples required **before** building an adapted expert or updating the global expert.  
  - If the buffer has fewer than this many samples, the model falls back to the **global** expert for that group.

- **`online_inner_steps`**: number of inner-loop gradient steps used when adapting:  
  - In `*_reptile_with_few_shot_adaptation*.py`, this controls how many SGD steps are run when creating the per-group adapted expert (or updating the global expert in the `*_update_global_expert.py` variant).
  - Increasing this usually improves adaptation (up to a point) but makes each adaptation more expensive.

- **`online_inner_lr`**: learning rate used for the few-shot inner loop during evaluation.  
  - Larger → faster adaptation but higher risk of overshooting / instability.
  - Smaller → slower, more conservative adaptation.

- **`online_adaptation_enabled`**: master on/off switch for online few-shot adaptation.  
  - `False` → the model behaves like a standard global model (no per-group adaptation, only the meta-training is used).

- **`online_update_global`** (only in `model_mlp_emb_512_reptile_with_few_shot_adaptation_update_global_expert.py`):  
  - `True` → the **global** expert is continually fine-tuned online using buffered data for each group (“non-frozen” global expert).
  - `False` → keep the global expert frozen and create **ephemeral** adapted experts per group (original Reptile-style few-shot behaviour).

These parameters can be set in your `training_config.json` under `args` to tune how quickly and how aggressively the model adapts to new job types or hosts.