import re
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

from alice_jobs_package.utils import logging

logger = logging.get_logger(__name__)

@staticmethod
def load_dataframe_from_numpy(data_path: Path, pandas = True) -> pd.DataFrame:
    logger.info(f"Function: Loading dataframe from {data_path}")
    data_path = parse_to_pathlib(data_path)
    if not data_path.exists():
        raise IOError(f'Given path not found {data_path}')
    
    if data_path.suffix != '.npz':
        data_path = data_path.with_suffix('.npz')

    loaded = np.load(data_path, allow_pickle=True)
    loaded_data = loaded['data']
    loaded_columns = loaded['columns']
        
    if not pandas:
        return loaded_data, loaded_columns
    
    return pd.DataFrame(loaded_data, columns = loaded_columns).infer_objects()

@staticmethod
def save_dataframe_to_numpy(save_path: Path, data_frame: pd.DataFrame):
    logger.info(f"Function: Saving dataframe to file {save_path}")
    save_path = parse_to_pathlib(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.suffix != '.npz':
        save_path = save_path.with_suffix('.npz')
    
    np.savez(save_path, 
            data = data_frame.to_numpy(),
            columns = data_frame.columns)

@staticmethod
def save_dataframe_to_numpy_in_parts(save_path: Path, data_frame: pd.DataFrame, n_parts: int):
    logger.info(f"Function: Saving dataframe in {n_parts} parts")
    save_path = parse_to_pathlib(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Get base stem and suffix
    base_stem = save_path.stem
    suffix = save_path.suffix or ".npz"

    # Split into N roughly equal parts
    parts = np.array_split(data_frame, n_parts)

    for i, part in enumerate(parts):
        part_path = save_path.with_name(f"{base_stem}_{i}{suffix}")
        save_dataframe_to_numpy(part_path, part)
        logger.info(f"Saved part {i+1}/{n_parts} to {part_path}")

@staticmethod
def get_hash_from_file_conntent(file_path):
    hash_func = getattr(hashlib, 'sha256')()

    with open(file_path, 'rb') as file:
        while chunk := file.read(4096):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

@staticmethod
def get_hash_from_config_file(config_file: dict) -> str:
    sorted_config = _sort_config(config_file)
    json_string = json.dumps(sorted_config, sort_keys=True, separators=(',', ':'))
    hash_object = hashlib.sha256(json_string.encode('utf-8'))
    return hash_object.hexdigest()

@staticmethod
def combine_hashes(first_hash, sec_hash):
    combined_hash_input = first_hash + sec_hash
    final_hash = hashlib.sha256(combined_hash_input.encode('utf-8')).hexdigest()
    return final_hash

@staticmethod
def _sort_config(data):
    if isinstance(data, dict):
        return {key: _sort_config(value) for key, value in sorted(data.items())}
    elif isinstance(data, list):
        return sorted(_sort_config(item) for item in data)
    else:
        return data

@staticmethod
def parse_to_pathlib(path: str | Path) -> Path:
    return Path(path)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

@staticmethod
def split_dataframe_by_timestamp(data: pd.DataFrame, timestamp_of_split = None, uint='ms', shuffle = False):
    if uint != 'ms':
        raise Exception('For now pass timestamp in unix ms')
    if 'startedtimestamp' not in data.columns:
        raise Exception('There is no startedtimestamp column in given data frame')

    data = data.sort_values(by='startedtimestamp').reset_index(drop=True)

    if timestamp_of_split == None:
        timestamp_of_split = data['startedtimestamp'].median()
        logger.info(f'The date of dvision {pd.to_datetime(timestamp_of_split, unit="ms")}')

    split_before_timestamp = data[data['startedtimestamp'] < timestamp_of_split]
    split_after_timestamp = data[data['startedtimestamp'] >= timestamp_of_split]

    logger.info(f"Split before timestamp is size {split_before_timestamp.shape[0]} and after is {split_after_timestamp.shape[0]}")

    if shuffle:
        return split_before_timestamp.sample(frac=1), split_after_timestamp.sample(frac=1)

    return split_before_timestamp, split_after_timestamp

@staticmethod
def split_dataframe_by_ratio(data: pd.DataFrame, ratio: float, shuffle = False, return_only_index = False):
    if 'startedtimestamp' not in data:
        raise Exception('There is no startedtimestamp column in given data frame')

    data = data.sort_values(by='startedtimestamp').reset_index(drop=True)

    split_index = int(len(data) * ratio)

    split_before_index = data.iloc[:split_index]
    split_after_index = data.iloc[split_index:]
    
    print(f"Split first is size {split_before_index.shape[0]} and second is {split_after_index.shape[0]}")

    if shuffle:
        return split_before_index.sample(frac=1), split_after_index.sample(frac=1)

    if return_only_index:
        return split_before_index.index, split_after_index.index
    
    return split_before_index, split_after_index

@staticmethod
def get_index_of_every_new_day_from_dataframe(data: pd.DataFrame) -> np.array:
    if 'startedtimestamp' not in data:
        raise Exception('There is no startedtimestamp column in given data frame')
    
    data = data.sort_values(by='startedtimestamp').reset_index(drop=True)

    data_ = pd.DataFrame()
    data_['startedtimestamp'] = data.startedtimestamp
    data_['timestamps'] = pd.to_datetime(data_['startedtimestamp'].astype(int), unit='ms')

    daily_indexes = []
    for _, group in data_.groupby(pd.Grouper(key='timestamps', freq='D')):
        if len(group) > 0:
            daily_indexes.append(group.index[0])

    return np.array(daily_indexes)

@staticmethod
def split_np_array_by_indexes(data: np.array, indexes):
    splits = []
    prev_index = 0

    for index in indexes:
        splits.append(data[prev_index:index])
        prev_index = index
    splits.append(data[prev_index:])

    return splits

@staticmethod
def ensure_extension(filepath: Path, extension: str) -> Path:
    if not extension.startswith('.'):
        extension = '.' + extension 

    return filepath.with_suffix(extension)

@staticmethod
def get_files_by_stem_regex(folder_path: str | Path, filename: str | Path) -> list[Path]:
    folder = Path(folder_path)
    filename = Path(filename)
    stem_pattern = filename.stem

    if not folder.exists() or not folder.is_dir():
        # No directory → no files
        return []

    regex = re.compile(stem_pattern)

    matching_files = [
        f
        for f in folder.iterdir()
        if f.is_file() and regex.match(f.stem)
    ]

    return matching_files

@staticmethod
def count_matching_files(folder_path: str | Path, filename: str | Path) -> int:
    folder = Path(folder_path)
    filename = Path(filename)

    matching_files = get_files_by_stem_regex(folder, filename)
    return len(matching_files)
