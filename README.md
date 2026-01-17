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
This component loads, parses, and cleans raw data into a single table, which can then be processed further using other parts of the package. Note that the data itself is not downloaded by this package; it must be accessed via external sources.

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

This process will generate two files:
1.	joined_site_sonar.npz: This file contains the combined data from all site-sonar-<timestamp>.out.xz files.
2.	joined_data.npz: This file contains the properly joined and cleaned data from all input files.

This code currently assumes that all raw files may contain new values but not new categories. However, if new categories do appear, they must be added to the alice_jobs_package/resources/dtypes files. This ensures that the data is correctly typed. Without this declaration, the data might be improperly recognized.

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

The second file also has a baseline in the resources directory, it is used in both cases OHE and Embedings. It defines the threshold for applying a limit on the number of distinct values allowed for a single processed column (rest will be set as unknown). The structure of this file is as follows:

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
