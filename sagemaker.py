import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path

# role and local output path
role = 'arn:aws:iam::id:role/role'  # only necessary for running in cloud mode (nice try github)
local_output_path = str(Path('~/sandbox').expanduser())

# Set up hyperparameters
hyperparameters = {
    'epoch': 20,
    'batchsize': 4,
    'learning_rate': 0.001,
    'accumulation_steps': 1,
    'n_fft': 2048,
    'hop_length': 1024,
}


estimator = PyTorch(
    entry_point='sagemaker.py',
    role=role,                      # required for cloud mode
    instance_count=1,
    instance_type='local',          # 'local' for local mode
    framework_version='2.5.1',
    py_version='py312',
    hyperparameters=hyperparameters,
    output_path=local_output_path,  # path for model output
)

# local paths for data
local_train_path = 'file://{}/train'.format(Path('~/data/instrs').expanduser())
local_val_path = 'file://{}/val'.format(Path('~/data/mixes').expanduser())

# start training
estimator.fit({'train': local_train_path, 'validation': local_val_path})

