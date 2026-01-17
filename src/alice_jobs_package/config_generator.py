import json
import math
import pandas as pd
import pkg_resources
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.utils import logging, tools 
from alice_jobs_package.data_loader import AliceDataLoader

logger = logging.get_logger(__name__) 

class AliceConfigGenerator():
    @staticmethod
    def generate_config(data_path: Path, considered_columns_config_path: Path = None, ohe_threshold_config_path: Path = None, ignore_cache: bool = False, verbose: bool = False):
        if verbose is True: logging.set_verbosity_info()
        elif verbose is False: logging.set_verbosity_warning()

        data_path = tools.parse_to_pathlib(data_path)
        if considered_columns_config_path is None:
            considered_columns_config_path = pkg_resources.resource_filename(__name__, 'resources/preprocessing_config/considered_columns.json')
        with open(considered_columns_config_path, 'r') as file:
            considered_columns_config = json.load(file)
        
        if ohe_threshold_config_path is None:
            ohe_threshold_config_path = pkg_resources.resource_filename(__name__, 'resources/preprocessing_config/ohe_threshold_config.json')
        with open(ohe_threshold_config_path, 'r') as file:
            ohe_threshold_config = json.load(file)
        
        file_hash = tools.get_hash_from_file_conntent(file_path=__file__)
        config_hash = tools.get_hash_from_config_file(considered_columns_config | ohe_threshold_config)
        hash_value = tools.combine_hashes(config_hash, file_hash)
        logger.info(f'Hash for this config is {hash_value}')
        hash_value = Path(hash_value)
        cached_folder = data_path / CacheDataFolders.CACHE_DATA_CONFIG_FOLDER.value / hash_value
        
        return AliceConfigGenerator._generate_configs(considered_columns_config = considered_columns_config,
                                                        ohe_threshold_config = ohe_threshold_config,
                                                        data_path = data_path,
                                                        cached_folder = cached_folder,
                                                        hash_value = hash_value,
                                                        ignore_cache = ignore_cache,
                                                        verbose = verbose)

    @staticmethod
    def _generate_configs(considered_columns_config: dict, ohe_threshold_config: dict, data_path: Path, cached_folder: Path, hash_value: Path, ignore_cache: bool, verbose: bool):
        logger.info(f'Generating embeding configs')
        numerical_file = (cached_folder / f'numerical.json')
        categorical_file = (cached_folder / f'categorical.json')

        if not ignore_cache:
            numerical_dict = AliceConfigGenerator._check_cache_for_config(numerical_file)
            categories_dict = AliceConfigGenerator._check_cache_for_config(categorical_file)

            if len(numerical_dict) != 0 and len(categories_dict) != 0:
                return numerical_dict, categories_dict
        
        logger.info(f'Preapring for generating')
        cached_folder.mkdir(parents=True, exist_ok=True)

        raw_csv_subfolder = data_path / RawDataSubFolders.RAW_CSV.value
        mon_jdls_files_count = tools.count_matching_files(raw_csv_subfolder, RawDataFiles.MON_JDLS.value)
        
        arr_numerical_dict = []
        arr_categories_dict = []

        for file_number in range(mon_jdls_files_count):
            numerical_dict, categories_dict = AliceConfigGenerator._generate_config_for_part(
                part = file_number,
                considered_columns_config = considered_columns_config,
                ohe_threshold_config = ohe_threshold_config,
                data_path = data_path,
                cached_folder = cached_folder,
                hash_value = hash_value,
                ignore_cache = ignore_cache,
                verbose = verbose)
            
            arr_numerical_dict.append(numerical_dict)
            arr_categories_dict.append(categories_dict)
        
        all_numerical_dict = AliceConfigGenerator._merge_numerical_dicts(arr_numerical_dict)
        with open(numerical_file, 'w') as file:
            json.dump(all_numerical_dict, file)
            
        all_categories_dict = AliceConfigGenerator._merge_categorical_dicts(arr_categories_dict)
        with open(categorical_file, 'w') as file:
            json.dump(all_categories_dict, file)
        
        return all_numerical_dict, all_categories_dict
    
    @staticmethod
    def _generate_config_for_part(part: int, considered_columns_config: dict, ohe_threshold_config: dict, data_path: Path, cached_folder: Path, hash_value: Path, ignore_cache: bool, verbose: bool):
        logger.info(f'Generating embeding configs part {part}')
        numerical_file = (cached_folder / f'numerical_{part}.json')
        categorical_file = (cached_folder / f'categorical_{part}.json')

        if not ignore_cache:
            numerical_dict = AliceConfigGenerator._check_cache_for_config(numerical_file)
            categories_dict = AliceConfigGenerator._check_cache_for_config(categorical_file)

            if len(numerical_dict) != 0 and len(categories_dict) != 0:
                return numerical_dict, categories_dict
        
        logger.info(f'Preapring for generating part {part}')
        cached_folder.mkdir(parents=True, exist_ok=True)

        df = AliceDataLoader.load_part_of_data(part = part,
                                                data_path = data_path, 
                                                ignore_cache = False, 
                                                verbose = verbose)

        categories_columns = considered_columns_config["categories_columns"]
        categories_dict = AliceConfigGenerator._generate_categorical_config(df, categories_columns, ohe_threshold_config, categorical_file)

        numerical_columns = considered_columns_config["numerical_columns"]
        numerical_dict = AliceConfigGenerator._generate_numerical_config(df, numerical_columns, numerical_file)

        return numerical_dict, categories_dict
    
    @staticmethod
    def _check_cache_for_config(file_path: Path) -> dict:
        if file_path.exists():
            logger.info(f'Found cached file {file_path}')
            with open(file_path, 'r') as file:
                data = json.load(file)
        else:
            logger.warning(f'Cached file not found {file_path}')
            data = {}
        return data
    
    @staticmethod
    def _generate_categorical_config(df: pd.DataFrame, categories_columns: list, ohe_threshold_config: dict, embeding_file: Path):
        logger.info(f'Generating config for categorical columns embedings')

        embedings_encoding_dict = {}

        with tqdm(categories_columns) as pbar:
            for column_name in pbar:
                pbar.set_description(f"Processing {column_name}")
                column = df[column_name].fillna(UNKNOWN_VALUE_NAME)

                #TODO potential problem with list values that in orginal data has Nan as first record.
                # Now it will be string so it is possible this never will be called for notmandatory features

                if isinstance(column[0],list): 
                    unique_values = Counter()
                    for i, row in enumerate(column):
                        if isinstance(row,list):
                            unique_values.update(row)
                        # else:
                        #     logger.warning(f"Nan list value in coulmn with type list, case handeled as UNKNOWN_VALUE_NAME")
                        if i % 100 == 0:
                            pbar.set_description(f"Processing {column_name}, row {i}")

                else:
                    unique_values = Counter(column)
                
                # Filter out values below the threshold
                if len(unique_values) >= ohe_threshold_config['threshold_limit']:
                    filtered_values = {key: count for key, count in unique_values.items() if count >= ohe_threshold_config['threshold_value']}
                else:
                    filtered_values = unique_values
                
                _, unique_strings = pd.factorize(pd.Series(filtered_values.keys()).astype(str))

                encoding_dict = {string: index for index, string in enumerate(unique_strings)}
                if UNKNOWN_VALUE_NAME not in encoding_dict.keys():
                    encoding_dict[UNKNOWN_VALUE_NAME] = len(encoding_dict.keys())
                embedings_encoding_dict[column_name] = encoding_dict

        with open(embeding_file, 'w') as file:
            json.dump(embedings_encoding_dict, file)
        
        return embedings_encoding_dict

    @staticmethod
    def _generate_numerical_config(df: pd.DataFrame, numerical_columns: list, numerical_file: Path):
        logger.info(f'Generating config for numerical columns')

        numerical_dict = {}

        with tqdm(numerical_columns) as pbar:
            for column_name in pbar:
                column = df[column_name]
                median, mean, std, min, max, count = column.median(), column.mean(), column.std(), column.min(), column.max(), column.count()
                numerical_dict[column_name] = {'median':float(median), 'mean':float(mean), 'std':float(std), 'min':float(min), 'max':float(max), 'count': int(count)}
                pbar.set_description(f"Processing {column_name}")

        with open(numerical_file, 'w') as file:
            json.dump(numerical_dict, file)
        
        return numerical_dict
    
    @staticmethod
    def _load_filter_data_config(filter_data_config_path: Path = None):
        if filter_data_config_path is None:
            filter_data_config_path = pkg_resources.resource_filename(__name__, 'resources/preprocessing_config/filter_data_config.json')
        with open(filter_data_config_path, 'r') as file:
            filter_data_config = json.load(file)
        
        return filter_data_config
    
    @staticmethod
    def _merge_numerical_dicts(dicts):
        agg = defaultdict(lambda: {
            "sum": 0.0, "sum_sq": 0.0, "count": 0,
            "min": float("inf"), "max": float("-inf"),
            "medians": []
        })

        def safe(val):
            """Return True if val is a valid number (not NaN/None)."""
            return val is not None and not (isinstance(val, float) and math.isnan(val))

        for d in dicts:
            for col, stats in d.items():
                count = stats.get("count", 0)
                mean = stats.get("mean")
                std = stats.get("std")
                vmin = stats.get("min")
                vmax = stats.get("max")
                median = stats.get("median")

                # only merge if values are valid
                if count and safe(mean):
                    agg[col]["count"] += count
                    agg[col]["sum"] += mean * count
                    if safe(std):
                        agg[col]["sum_sq"] += (std**2 + mean**2) * count
                    else:
                        agg[col]["sum_sq"] += mean**2 * count

                if safe(vmin):
                    agg[col]["min"] = min(agg[col]["min"], vmin)
                if safe(vmax):
                    agg[col]["max"] = max(agg[col]["max"], vmax)
                if safe(median):
                    agg[col]["medians"].append(median)

        # finalize stats
        final = {}
        for col, v in agg.items():
            count = v["count"]
            if count == 0:
                # no valid samples, return NaN values
                final[col] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "count": 0,
                    "median": float("nan")
                }
                continue

            mean = v["sum"] / count
            mean_sq = v["sum_sq"] / count
            variance = mean_sq - mean**2
            std = math.sqrt(max(variance, 0.0))

            # "soft median" = mean of medians if available
            median = float("nan")
            if v["medians"]:
                median = sum(v["medians"]) / len(v["medians"])

            final[col] = {
                "mean": mean,
                "std": std,
                "min": v["min"] if v["min"] != float("inf") else float("nan"),
                "max": v["max"] if v["max"] != float("-inf") else float("nan"),
                "count": count,
                "median": median
            }

        return final
    
    @staticmethod
    def _merge_categorical_dicts(dicts):
        agg = defaultdict(set)
        for d in dicts:
            for col, values in d.items():
                agg[col].update(values.keys())

        # reassign new indices consistently
        final = {col: {val: idx for idx, val in enumerate(sorted(values))} for col, values in agg.items()}
        return final