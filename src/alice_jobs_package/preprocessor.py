import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.utils import logging, tools
from alice_jobs_package.data_loader import AliceDataLoader

logger = logging.get_logger(__name__)

class AliceDataPreprocessor:
    # POSSIBLE TODO (DELETED FORM LEGACY PREPROCESSING DUE TO INACTIVITY)
    # - ASSIGN WILE PROCESSING PREVIOUS WALLTIME FROM SAME MACHINE
    # - SELECT ONLY N (N from 0.0, to 1.0) MOST IMPORTANT FEATURES
    # - SPLIT DATA OPTIONS (FOR NOW MOVED TO TOOLS)
    # - SHORT LONG JOBS

    def __init__(self, data_path: Path, processing_target: ProcessingTarget, numerical_columns_config: dict, categorical_columns_config: dict, filter_data_config: dict = {}, prediction_target_column = 'walltime_hours'):
        self.processing_target = processing_target
        self.data_path = tools.parse_to_pathlib(data_path)

        self.config_hash = tools.get_hash_from_config_file(numerical_columns_config | categorical_columns_config | filter_data_config)
        self.file_hash = tools.get_hash_from_file_conntent(file_path=__file__)
        self.cache_hash = tools.combine_hashes(self.config_hash, self.file_hash)
        self.cache_folder = self.data_path / CacheDataFolders.CACHE_DATA_PROCESSED_FOLDER.value / f'{self.cache_hash}'
        self.cache_file_path = (self.cache_folder / (PROCESSING_TARGET_MAP[self.processing_target.name])).with_suffix('.npz')

        self.numerical_columns_config = numerical_columns_config
        self.categorical_columns_config = categorical_columns_config
        self.filter_data_config = filter_data_config
        self.prediction_target_column = prediction_target_column
        self.columns = []
        self.num_col_numbers = []
        self.cat_col_numbers = []
        
        self.ignore_cache = False
        self.output_dtype = np.float32
        return

    def preprocess(self, pandas: bool = False, ignore_cache: bool = False, verbose: bool = False) -> tuple:
        if verbose is True: logging.set_verbosity_info()
        elif verbose is False: logging.set_verbosity_warning()

        if not ignore_cache and self.cache_file_path.is_file():
            logger.info(f'Founded cached file {self.cache_file_path}')

            processed_data = tools.load_dataframe_from_numpy(self.cache_file_path)

        else:
            logger.info(f'Cache not found {self.cache_file_path}')
            raw_csv_subfolder = self.data_path / RawDataSubFolders.RAW_CSV.value
            mon_jdls_files_count = tools.count_matching_files(raw_csv_subfolder, RawDataFiles.MON_JDLS.value)

            processed_data_paths = []

            for file_number in range(mon_jdls_files_count):
                part_file_path = self.preprocess_for_part(file_number, ignore_cache, verbose)
                processed_data_paths.append(part_file_path)

            dfs = []
            for path in processed_data_paths:
                df = tools.load_dataframe_from_numpy(path)
                dfs.append(df)

            processed_data = pd.concat(dfs, axis=0, ignore_index=True)
            processed_data = self._sort_data_by_values(processed_data)
            tools.save_dataframe_to_numpy(self.cache_file_path, processed_data)

        X = processed_data.drop(self.prediction_target_column, axis=1)
        y = processed_data[[self.prediction_target_column]]

        self.parse_column_numbers(X)

        if pandas:
            return (X, y)
        else:
            return (X.to_numpy(), y.to_numpy())
        
    def preprocess_for_part(self, part: int = 0, ignore_cache: bool = False, verbose: bool = False) -> Path:
        if verbose is True: logging.set_verbosity_info()
        elif verbose is False: logging.set_verbosity_warning()

        logger.info(f'Preprocessing data for {self.processing_target}, hash {self.cache_hash}')

        self.ignore_cache = ignore_cache

        part_file_name = Path(self.cache_file_path.name)
        preprocess_part_file = self.cache_folder / part_file_name.with_stem(f"{part_file_name.stem}_{part}")

        if not ignore_cache and preprocess_part_file.is_file():
            logger.info(f'Founded cached file for part {part} {preprocess_part_file}')
                    
        else:
            logger.info(f'Cache not found for part {part} {preprocess_part_file}')

            raw_data = AliceDataLoader.load_part_of_data(part = part,
                                                        data_path = self.data_path, 
                                                        ignore_cache = False, 
                                                        verbose = verbose)

            if self.filter_data_config:
                for data_type, dictionary in self.filter_data_config.items():
                    if data_type == "numerical":
                        for key, value in dictionary.items():
                            if key in raw_data:
                                raw_data = raw_data[raw_data[key] >= value["min"]]
                                raw_data = raw_data[raw_data[key] <= value["max"]]
                    elif data_type == "categorical":
                        for key, value in dictionary.items():
                            if key in raw_data:
                                if isinstance(value, str):
                                    raw_data = raw_data[raw_data[key] == value]
                                elif isinstance(value, list):
                                    raw_data = raw_data[raw_data[key].isin(value)]

            if PROCESSING_TARGET_MAP[self.processing_target.name] == ProcessingType.NO_PROCESSING.name:
                logger.info(f"Skipping encoding")
                processed_data = self._preprocess_noprocessing(raw_data)
                tools.save_dataframe_to_numpy(preprocess_part_file, processed_data)

            elif PROCESSING_TARGET_MAP[self.processing_target.name] == ProcessingType.ONE_HOT_ENCODED.name:
                logger.info(f"Processing for ONE_HOT_CONFIGS")
                processed_data = self._preprocess_one_hot_encode(raw_data)
                tools.save_dataframe_to_numpy(preprocess_part_file, processed_data)
                processed_data.astype(self.output_dtype)
            
            elif PROCESSING_TARGET_MAP[self.processing_target.name] == ProcessingType.EMBEDINGS_ENCODED.name:
                logger.info(f"Processing for EMBEDING_CONFIGS")
                processed_data = self._preprocess_embedding(raw_data)
                tools.save_dataframe_to_numpy(preprocess_part_file, processed_data)
                processed_data.astype(self.output_dtype)

            logger.info(f"Saved sucesfully for part {part}")

        return preprocess_part_file
        

    def _preprocess_one_hot_encode(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        raw_data = self._sort_data_by_values(raw_data)
        raw_data = self._filter_columns_by_config(raw_data, additional_columns = [self.prediction_target_column])
        raw_data = self._trasnforme_one_hot_encoding(raw_data)
        return raw_data
    
    def _trasnforme_one_hot_encoding(self, raw_data: pd.DataFrame):

        with tqdm(self.categorical_columns_config.keys()) as pbar:
            for column_name in pbar:
                pbar.set_description(f"Processing {column_name}")
                
                raw_data[column_name] = raw_data[column_name].fillna(UNKNOWN_VALUE_NAME)
                config_unique_values = set(self.categorical_columns_config[column_name].keys())

                if isinstance(raw_data[column_name][0],list):
                    raw_data = self._trasnforme_one_hot_encoding_list_colums(raw_data, column_name, config_unique_values, pbar = pbar)	
                else:
                    raw_data = self._trasnforme_one_hot_encoding_object_colums(raw_data, column_name, config_unique_values, pbar = pbar)

            raw_data = self._trasnforming_numerical(raw_data)

        return raw_data
    
    def _trasnforme_one_hot_encoding_list_colums(self, raw_data: pd.DataFrame, column_name: str, config_unique_values: set, pbar) -> pd.DataFrame:
        #Handling new values that are absent in config
        def map_values(value_list):
            if isinstance(value_list, list):
                return [
                    value if value in config_unique_values else UNKNOWN_VALUE_NAME
                    for value in value_list
                ]
            else:
                return ['unknown']

        pbar.set_description(f"Processing, maping unhandeled values")
        raw_data[column_name] = raw_data[column_name].apply(map_values)

        one_hot_columns = []
        one_hot_columns_seq = []

        for i, unique_value in enumerate(config_unique_values):
            pbar.set_description(f"Processing {column_name}, one hoting, column {i}")
            column_new_name = f"{column_name}_{unique_value}"
            one_hot_columns_seq.append(column_new_name)

            # Vectorized one-hot encoding using np.isin for faster performance
            one_hot = np.array([1 if unique_value in row else 0 for row in raw_data[column_name]], dtype=np.uint8)
            one_hot_columns.append(one_hot)

        one_hot_columns = np.column_stack(one_hot_columns)
        one_hot_columns = pd.DataFrame(one_hot_columns, columns=one_hot_columns_seq)
        one_hot_columns = one_hot_columns[sorted(one_hot_columns_seq)]

        raw_data = pd.concat([raw_data] + [one_hot_columns], axis=1)
        raw_data.drop(columns=[column_name], inplace=True)

        return raw_data
           
    def _trasnforme_one_hot_encoding_object_colums(self, raw_data: pd.DataFrame, column_name: str, config_unique_values: set, pbar):
        #Handling new values that are absent in config
        raw_data[column_name] = raw_data[column_name].astype(str)
        raw_data[column_name] = raw_data[column_name].apply(lambda x: UNKNOWN_VALUE_NAME if x not in config_unique_values else x)

        # One hot encode only known values
        pbar.set_description(f"Processing {column_name}, one hoting")
        one_hot_columns = pd.get_dummies(raw_data[column_name], prefix=column_name).astype(np.uint8)
        one_hot_columns_seq = []

        #Fill up empty columns for values absent in data
        for i, unique_value in enumerate(config_unique_values):
            column_new_name = f"{column_name}_{unique_value}"
            one_hot_columns_seq.append(column_new_name)
            if column_new_name not in one_hot_columns.columns:
                one_hot_columns[column_new_name] = False
                one_hot_columns[column_new_name] = one_hot_columns[column_new_name].astype(np.uint8)

        #always organize columnd in sorted order
        one_hot_columns = one_hot_columns[sorted(one_hot_columns_seq)]

        pbar.set_description(f"Processing {column_name}, concatenating")
        raw_data = pd.concat([raw_data] + [one_hot_columns], axis=1)
        raw_data.drop(columns=[column_name], inplace=True)

        return raw_data

    def _preprocess_embedding(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        raw_data = self._sort_data_by_values(raw_data)
        raw_data = self._filter_columns_by_config(raw_data, additional_columns = [self.prediction_target_column])
        raw_data = self._transforming_for_embedings(raw_data)
        return raw_data
  
    def _transforming_for_embedings(self, raw_data: pd.DataFrame):
        for column_name in tqdm(self.categorical_columns_config.keys()):
            mapping = self.categorical_columns_config[column_name]
            raw_data[column_name] = raw_data[column_name].map(mapping) #New values which are not in map will be mapped to Nan
            raw_data[column_name] = raw_data[column_name].fillna(mapping[UNKNOWN_VALUE_NAME]) #For new data with unknown values for column

        raw_data = self._trasnforming_numerical(raw_data)

        return raw_data
    
    def _trasnforming_numerical(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        for column_name in tqdm(self.numerical_columns_config.keys()):
            stats = self.numerical_columns_config[column_name]

            raw_data[column_name] = raw_data[column_name].fillna(stats['median'])
            raw_data[column_name] = raw_data[column_name].astype(self.output_dtype)
            if stats['std'] != 0:
                raw_data[column_name] = (raw_data[column_name] - stats['mean']) / stats['std'] 
            else:
                raw_data[column_name] = raw_data[column_name] - stats['mean'] 
        return raw_data

    def _preprocess_noprocessing(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        raw_data = self._sort_data_by_values(raw_data)
        raw_data = self._filter_columns_by_config(raw_data, additional_columns = [self.prediction_target_column])
        return raw_data
    
    def _filter_columns_by_config(self, raw_data: pd.DataFrame, additional_columns = []) -> pd.DataFrame:
        all_columns = list(self.categorical_columns_config.keys()) + list(self.numerical_columns_config.keys()) + additional_columns
        
        missing_columns = [col for col in all_columns if col not in raw_data.columns]
        if len(missing_columns) > 0:
            print(f"Missing columns: {', '.join(missing_columns)}")

            for col in missing_columns:
                raw_data[col] = UNKNOWN_VALUE_NAME

        return raw_data[all_columns]
        
    def _sort_data_by_values(self, raw_data: pd.DataFrame, columns: list = ['startedtimestamp'], asscending: list = [True]) -> pd.DataFrame:
        return raw_data.sort_values(by=columns, ascending=asscending).reset_index(drop=True)

    def parse_column_numbers(self, X: pd.DataFrame):
        # Get numerical column indices
        self.num_col_numbers = [X.columns.get_loc(col) for col in self.numerical_columns_config.keys()]

        # Get categorical column indices as the complement of numerical indices
        self.cat_col_numbers = [i for i in range(len(X.columns)) if i not in self.num_col_numbers]

        self.columns = X.columns