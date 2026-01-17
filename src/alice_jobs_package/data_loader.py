import gc
import os
import re
import sys
import json
import time
import psutil
import numpy as np
import pandas as pd
import pkg_resources
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.utils import logging, tools 
from alice_jobs_package.utils.statistical_model.weighted_ttl.online_statistical_model import OnlineStatisticalModel

logger = logging.get_logger(__name__)

class AliceDataLoader():
    @staticmethod
    def load_part_of_data(data_path: Path, part: int = 0, ignore_cache: bool = False, verbose: bool = False) -> pd.DataFrame:
        if verbose: logging.set_verbosity_info()
        else: logging.set_verbosity_warning()
        #Be shure all data is properly loaded and joined
        AliceDataLoader.load_data(data_path = data_path, 
                                    ignore_cache = False, 
                                    verbose = verbose,
                                    only_join=True)

        data_path = tools.parse_to_pathlib(data_path)
        hash_value = AliceDataLoader.generate_hash()
        file_part =  PreprocessedDataFiles.OUTPUT_DATA.value.with_stem(PreprocessedDataFiles.OUTPUT_DATA.value.stem + f'_{part}')
        output_file_part = data_path / CacheDataFolders.CACHE_JOINED_DATA_FOLDER.value / Path(hash_value) / file_part

        logger.info(f"Function: Loading data part {part} from {output_file_part}")

        df = tools.load_dataframe_from_numpy(output_file_part)
        return df

    @staticmethod
    def load_data(data_path: Path, ignore_cache: bool = False, verbose: bool = False, only_join: bool = False) -> pd.DataFrame | None:
        if verbose: logging.set_verbosity_info()
        else: logging.set_verbosity_warning()
        
        data_path = tools.parse_to_pathlib(data_path)
        site_sonar_subfolder = data_path / RawDataSubFolders.SITE_SONARS.value
        raw_csv_subfolder = data_path / RawDataSubFolders.RAW_CSV.value
        cache = data_path / CacheDataFolders.CACHE_JOINED_DATA_FOLDER.value
        hash_value = AliceDataLoader.generate_hash()

        joined_data_path = cache / Path(hash_value) / PreprocessedDataFiles.JOINED_DATA.value
        output_data_path = cache / Path(hash_value) / PreprocessedDataFiles.OUTPUT_DATA.value
        mon_jdls_files_paths = tools.get_files_by_stem_regex(raw_csv_subfolder, RawDataFiles.MON_JDLS.value)

        logger.info(f"Function: Loading data from {cache / Path(hash_value)}")

        if tools.count_matching_files(output_data_path.parent, output_data_path.name) != len(mon_jdls_files_paths) or ignore_cache:
            if not (joined_data_path).exists() or ignore_cache:
                logger.info(f"Processed data file not found: {joined_data_path}")
                AliceDataLoader._validate_file_paths(raw_csv_subfolder, site_sonar_subfolder)
                
                jobinfo_and_trace = AliceDataLoader._load_jobinfo_and_trace(raw_csv_subfolder)

                partial_paths = []

                for i, mon_jdls_path in enumerate(sorted(mon_jdls_files_paths)):
                    logger.info(f"--- Processing mon jdls {i}/{len(mon_jdls_files_paths)} ---")

                    part_npz_path = AliceDataLoader._join_data_for_one_part(
                        mon_jdls_path=mon_jdls_path,
                        jobinfo_and_trace=jobinfo_and_trace,
                        site_sonar_subfolder=site_sonar_subfolder,
                        cache=cache,
                        hash_value=hash_value,
                        part=i,
                        ignore_cache = False,
                        verbose = verbose
                    )

                    partial_paths.append(part_npz_path)

                    gc.collect()
                    
                all_dfs = []
                logger.info("Combining parts of data into single DataFrame")
                for p in partial_paths:
                    df_part = tools.load_dataframe_from_numpy(p)
                    all_dfs.append(df_part)

                df = pd.concat(all_dfs, ignore_index=True)
                df = AliceDataLoader._additional_processing(df)

                tools.save_dataframe_to_numpy(joined_data_path, df)

            else:
                logger.info(f"Processed data found, preparing to divide into output files")
                df = tools.load_dataframe_from_numpy(joined_data_path)
            
            tools.save_dataframe_to_numpy_in_parts(output_data_path, df, len(mon_jdls_files_paths))
            
            if not only_join:
                return df
        
        elif len(mon_jdls_files_paths) == 0:
            raise f'There is a problem with detecting mon_jdls_files_paths, probably name of files is wrong'

        elif not only_join:
            logger.info(f"Processed data found")
            df = tools.load_dataframe_from_numpy(joined_data_path)
            return df
        
    @staticmethod
    def _join_data_for_one_part(mon_jdls_path: Path,
                                jobinfo_and_trace: dict[str, pd.DataFrame], 
                                site_sonar_subfolder: Path, 
                                cache: Path, 
                                hash_value: str, 
                                part: int, 
                                ignore_cache: bool = False, 
                                verbose: bool = False) -> Path:
        if verbose: logging.set_verbosity_info()
        else: logging.set_verbosity_warning()

        joined_data_path = cache / Path(hash_value) / PreprocessedDataFiles.JOINED_DATA.value.with_stem(f"{PreprocessedDataFiles.JOINED_DATA.value.stem}_{part}")

        if not (joined_data_path).exists() or ignore_cache:
            logger.info(f"Processed data file not found: {joined_data_path}")

            mon_jdls = AliceDataLoader._load_single_mon_jdls(mon_jdls_path)
            
            df = AliceDataLoader._join_data(mon_jdls_in = mon_jdls, 
                                            job_info_in = jobinfo_and_trace[RawDataFiles.JOB_INFO.name], 
                                            trace_in = jobinfo_and_trace[RawDataFiles.TRACE.name], 
                                            site_sonar_subfolder = site_sonar_subfolder,
                                            cache=cache,
                                            target_part=part)
            
            tools.save_dataframe_to_numpy(joined_data_path, df)

        return joined_data_path

    @staticmethod
    def _validate_file_paths(data_path: Path, site_sonar_subfolder: Path):
        logger.info(f"Function: Validating data paths")
        if not data_path.exists() or not (site_sonar_subfolder).exists():
            raise IOError(f"Given path/s not found. {data_path} {site_sonar_subfolder}")

        expected_files = list(RawDataFiles)
        site_sonar_regex = r'site-sonar-\d+.out.xz'

        missing_files = []
        for file in expected_files:
            matches = [
                f for f in data_path.iterdir()
                if f.is_file() and file.regex.match(f.name)
            ]
            if not matches:
                missing_files.append(file)

        if missing_files:
            raise FileNotFoundError(f"The following files are missing: {', '.join(filename.value.name for filename in missing_files)}")
        else:
            logger.info(f"Found all raw files: {[filename.value.name for filename in expected_files]} in {data_path}")

        matching_files = [f for f in (site_sonar_subfolder).iterdir() if re.match(site_sonar_regex, f.name)]
        num_matching_files = len(matching_files)

        if num_matching_files <= 0:
            raise FileNotFoundError(f"There are no proper named site-sonar data in {site_sonar_subfolder}")
        else:
            logger.info(f"Found {num_matching_files} site-sonar files in {site_sonar_subfolder}")
    
    @staticmethod
    def _find_raw_csv_files(data_path: Path, file: RawDataFiles):
        files = [
                f for f in data_path.iterdir()
                if f.is_file() and file.regex.match(f.name)
            ]
        
        if not files:
            logger.warning(f"No files found for {file.name}")
        else:
            logger.info(f"{file.name}: Found {len(files)} files")
        
        return files

    @staticmethod
    def _load_jobinfo_and_trace(data_path: Path) -> dict[str, pd.DataFrame]:
        logger.info("Function: Loading JOB_INFO and TRACE split files")
        loaded_data = {}

        for file in [RawDataFiles.JOB_INFO, RawDataFiles.TRACE]:
            
            files = AliceDataLoader._find_raw_csv_files(data_path, file)
            
            # load dtypes
            dtype_file = pkg_resources.resource_filename(
                __name__, f"resources/dtypes/{file.value.stem}.json"
            )
            with open(dtype_file, "r") as f:
                dtypes = json.load(f)

            # read and concat all matching files
            dfs = []
            for f in tqdm(sorted(files), desc=f"Loading {file.name}"):
                df = pd.read_csv(f, dtype=dtypes)
                df.columns = map(str.lower, df.columns)
                dfs.append(df)

            df = pd.concat(dfs, ignore_index=True)

            # cleaning rules
            if file == RawDataFiles.JOB_INFO:
                df = df[df.status == "DONE"].drop_duplicates(subset='job_id', keep='last')
            elif file == RawDataFiles.TRACE:
                df = df.drop_duplicates(subset='job_id', keep='last')

            loaded_data[file.name] = df

        return loaded_data

    @staticmethod
    def _load_single_mon_jdls(mon_jdls_path: Path):

        logger.info(f"Function: Loading mon jdls from {mon_jdls_path}")

        dtype_file = pkg_resources.resource_filename(
            __name__, f"resources/dtypes/{RawDataFiles.MON_JDLS.value.stem}.json"
        )
        with open(dtype_file, "r") as f:
            dtypes = json.load(f)

        mon_jdls = pd.read_csv(mon_jdls_path, dtype=dtypes)
        mon_jdls.columns = map(str.lower, mon_jdls.columns)
        mon_jdls = mon_jdls.drop_duplicates(subset='job_id', keep='last')

        return mon_jdls

    @staticmethod
    def _get_earliest_job(joined_data: pd.DataFrame) -> int:
        earliest_job_timestamp = joined_data.where(joined_data.status == 'DONE')['job_submit_timestamp'].min()
        return int(earliest_job_timestamp)

    @staticmethod
    def _get_latest_job(joined_data: pd.DataFrame) -> int:
        earliest_job_timestamp = joined_data.where(joined_data.status == 'DONE')['job_submit_timestamp'].max()
        return int(earliest_job_timestamp)
    
    @staticmethod
    def _get_timestamps_boundaries(joined_data: pd.DataFrame) -> tuple[float, float]:
        earliest_job_timestamp = AliceDataLoader._get_earliest_job(
            joined_data = joined_data
        )
        earliest_job_timestamp = (datetime.fromtimestamp(earliest_job_timestamp) - timedelta(days=1)).timestamp()

        latest_job_timestamp = AliceDataLoader._get_latest_job(
            joined_data = joined_data
        )
        latest_job_timestamp = datetime.fromtimestamp(latest_job_timestamp).timestamp()

        logger.info(f"Earliest job {earliest_job_timestamp}, latest job {latest_job_timestamp}")
    
        return earliest_job_timestamp, latest_job_timestamp

    @staticmethod
    def _load_site_sonar(site_sonar_subfolder: Path, cache: Path, target_part: int, earliest_job_timestamp: float, latest_job_timestamp: float, ignore_cache: bool = False) -> pd.DataFrame:
        logger.info(f"Function: Loading site sonars")

        file_hash = tools.get_hash_from_file_conntent(file_path=__file__)
        config_hash = tools.get_hash_from_config_file(JobTagsConfig.FILE_TYPE | JobTagsConfig.WORKING_GROUP | JobTagsConfig.COLLISION_TYPE)
        hash_value = tools.combine_hashes(config_hash, file_hash)

        joined_site_sonar_data_path = cache / Path(hash_value)
        joined_site_sonar_file = joined_site_sonar_data_path / PreprocessedDataFiles.JOINED_SITE_SONAR.value.with_stem(f"{PreprocessedDataFiles.JOINED_SITE_SONAR.value.stem}_{target_part}")

        if not joined_site_sonar_file.exists() or ignore_cache:
            logger.info(f"Saved site-sonar for part {target_part} not found or ignoring cache")
            files_list = [f for f in site_sonar_subfolder.iterdir() if f.is_file()]
            logger.info(f"Found {len(files_list)} files")

            target_files = []
            for f in files_list:
                match = re.match(r'site-sonar-(\d+)\.out\.xz', f.name)
                if match:
                    timestamp = int(match.group(1))
                    if earliest_job_timestamp <= timestamp <= latest_job_timestamp:
                        target_files.append(f)

            logger.info(f"Processing part {target_part} with {len(target_files)} files")
            if not target_files:
                logger.warning(f"No files found for part {target_part}")
                return pd.DataFrame()

            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)
            logger.debug(f"Memory used before processing data for part {target_part}: {mem_before:.2f} MB")

            max_workers = min(32, len(target_files))
            loaded_site_sonar: dict[int, pd.DataFrame] = {}

            with open(pkg_resources.resource_filename(__name__, 'resources/site_sonar_config/considered_columns.json'), 'r') as f:
                columns = json.load(f)

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(AliceDataLoader._process_file_threaded, f, columns): f
                    for f in target_files
                }

                for future in tqdm(as_completed(futures), total=len(futures),
                                desc=f"Processing part {target_part}", file=sys.stdout,
                                dynamic_ncols=True, mininterval=0.1):
                    result = future.result()
                    if result:
                        timestamp, df = result
                        loaded_site_sonar[timestamp] = df
                
                executor.shutdown(wait=True)

            if loaded_site_sonar:
                joined_site_sonar_part = AliceDataLoader._join_stite_sonars_into_one(loaded_site_sonar)
                joined_site_sonar_part = AliceDataLoader._clear_site_sonar_from_duplicate_column(joined_site_sonar_part)

                tools.save_dataframe_to_numpy(joined_site_sonar_file, joined_site_sonar_part)
                logger.info(f"Saved site-sonar for part {target_part}, shape={joined_site_sonar_part.shape}")
            else:
                logger.warning(f"No valid site-sonar data loaded for part {target_part}")
                joined_site_sonar_part = pd.DataFrame()

            # Czyszczenie pamięci
            del loaded_site_sonar
            gc.collect()

            mem_after = process.memory_info().rss / (1024 ** 2)
            logger.debug(f"Memory after processing part {target_part}: {mem_after:.2f} MB")

        else:
            logger.info(f"Saved site-sonar found for part {target_part}")
            joined_site_sonar_part = tools.load_dataframe_from_numpy(joined_site_sonar_file)
            logger.info(f"Site sonar loaded: {joined_site_sonar_part.shape}")
        
        return joined_site_sonar_part

    @staticmethod
    def _process_file_threaded(file_path: Path, columns: list) -> tuple[int, pd.DataFrame] | None:
        try:
            match = re.match(r'site-sonar-(\d+)\.out\.xz', file_path.name)
            if not match:
                return None

            df = pd.read_json(file_path, compression='xz', lines=True)
            if 'test_results_json' not in df.columns:
                return None

            normalized = pd.json_normalize(df['test_results_json'])
            df = df.drop(columns=['test_results_json']).join(normalized)
            timestamp = int(match.group(1))

            # --- Enforce column schema ---
            # Drop extra columns not in the desired schema
            df = df[[col for col in columns if col in df.columns]]

            # Reorder columns exactly
            df = df.reindex(columns=columns)

            return (timestamp, df)

        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            return None

    @staticmethod
    def _join_stite_sonars_into_one(site_sonar_dict: dict) -> pd.DataFrame:
        logger.info(f"Function: Joining site sonars")
        merged_df = pd.concat(site_sonar_dict.values(), keys=site_sonar_dict.keys(), names=["timestamp"])
        merged_df = merged_df.sort_values(['last_updated'], ascending=False)
        merged_df = merged_df.reset_index(drop=True)
        return merged_df
    
    @staticmethod #potentialy in future delete all duplicated columns inside one table, for now only one was found
    def _clear_site_sonar_from_duplicate_column(site_sonar: pd.DataFrame) -> pd.DataFrame: 
        if 'cpu_info.CPU_BogoMIPS' in site_sonar.columns:
            site_sonar['cpu_info.CPU_bogomips'] = site_sonar['cpu_info.CPU_bogomips'].fillna(site_sonar['cpu_info.CPU_BogoMIPS'])
            site_sonar = site_sonar.drop('cpu_info.CPU_BogoMIPS', axis=1)
        return site_sonar

    @staticmethod
    def _log_shape(mon_jdls, job_info, trace):
        logger.info(f"Shape of mon_jdsl: {mon_jdls.shape}, job_info: {job_info.shape}, trace: {trace.shape}")

    @staticmethod
    def _join_data(mon_jdls_in: pd.DataFrame, job_info_in: pd.DataFrame, trace_in: pd.DataFrame, site_sonar_subfolder: Path, cache: Path, target_part: int) -> pd.DataFrame:
        # 1.Droping duplicates for every table by job_id, if there are additional cryteria use it first
        # Mon_jdls have some corupted data, like different records for same job_id, for all such records are droped
        # Due to change in framework, mon_jdls, job_info and trace are droped earlier
        logger.info(f"Function: Joining All Data")
        AliceDataLoader._log_shape(mon_jdls_in, job_info_in, trace_in)

        mon_jdls = mon_jdls_in.copy()

        job_info = job_info_in.copy()

        trace = trace_in.copy()

        # 2.Joining data
        joined_data = pd.merge(job_info, mon_jdls, on='job_id', suffixes = ("_job_info", "_mon_jdls"))
        logger.info(f"After joining job_info and mon_jdls {joined_data.shape}")

        joined_data = pd.merge(joined_data, trace, on='job_id', suffixes = ("", "_trace"))
        logger.info(f"After joining trace {joined_data.shape}")

        # 2.5 Load only necesary site sonars
        earliest_job_timestamp, latest_job_timestamp = AliceDataLoader._get_timestamps_boundaries(joined_data)
            
        site_sonar = AliceDataLoader._load_site_sonar(site_sonar_subfolder = site_sonar_subfolder, 
                                                        cache = cache, 
                                                        target_part = target_part,
                                                        earliest_job_timestamp = earliest_job_timestamp, 
                                                        latest_job_timestamp = latest_job_timestamp,
                                                        ignore_cache = False)
        
        site_sonar = site_sonar.drop_duplicates(subset=['hostname', 'last_updated'])
        site_sonar.columns = map(str.lower, site_sonar.columns) 

        # 3. Join Site Sonar
        start = time.time()
        ### This could potentialy cause problems if entering joined_data and joined_site_sonar will be big tables
        joined_data = joined_data.rename(columns={'host': 'hostname'})
        joined_data = joined_data.dropna(subset=['hostname'])
        logger.info(f"Before site sonar joined and after droping hostname Nan {joined_data.shape}")

        unique_hostnames = site_sonar.hostname.unique()
        joined_data = joined_data[joined_data.hostname.isin(unique_hostnames)]
        logger.info(f"After droping hostnames absent in site_sonar {joined_data.shape}")

        joined_data_small = joined_data[['job_id', 'hostname', 'startedtimestamp']]
        site_sonar_small = site_sonar[['hostname', 'last_updated']] 
        logger.info(f"Before site sonar joined (reduced data) {joined_data_small.shape}, {site_sonar_small.shape}")

        joined_data_small = pd.merge(joined_data_small, site_sonar_small, on='hostname')
        logger.info(f"Site sonar merge table created (reduced data) {joined_data_small.shape}")

        joined_data_small = joined_data_small[joined_data_small['last_updated'] < joined_data_small['startedtimestamp']]
        logger.info(f"Site sonar merge table filtered (reduced data) {joined_data_small.shape}")

        joined_data_small = joined_data_small.sort_values(['job_id', 'last_updated'], ascending=[True, False])
        logger.info(f"Site sonar merge table filtered second time (reduced data) {joined_data_small.shape}")

        joined_data_small = joined_data_small.drop_duplicates(subset='job_id', keep='first')
        logger.info(f"Site sonar merge table at the end {joined_data_small.shape}")

        joined_data = pd.merge(joined_data, joined_data_small, on=['job_id', 'hostname', 'startedtimestamp'], how='inner')
        joined_data = pd.merge(joined_data, site_sonar, on=['last_updated', 'hostname'], how='inner')
        logger.info(f"After merging all joined data {joined_data.shape}")
        ### End of potentialy problematic code 
        stop = time.time()
        logger.info(f'Time for site-sonar: {stop - start} sec')
        return joined_data

    @staticmethod
    def _additional_processing(joined_data : pd.DataFrame):
        joined_data = AliceDataLoader._parsing_timestamps(joined_data)
        joined_data = AliceDataLoader._parsing_col_job_tags(joined_data)
        joined_data = AliceDataLoader._parsing_arguments(joined_data)
        joined_data = AliceDataLoader._parsing_energy_levels(joined_data)
        joined_data = AliceDataLoader._drop_na(joined_data)
        joined_data = AliceDataLoader._add_statistical_predictions(joined_data)
        return joined_data

    @staticmethod
    def _parsing_timestamps(joined_data : pd.DataFrame): 
        logger.info(f"After joining, joined_data shape:{joined_data.shape}")

        joined_data['walltime'] = joined_data['walltime'] / joined_data['requestedcpus']
        joined_data['walltime_minutes'] = joined_data['walltime'] / 60
        joined_data['walltime_hours'] = joined_data['walltime'] / 3600

        joined_data = joined_data.query("0 < walltime_hours < 24 and finaltimestamp > 0 and startedtimestamp > 0").copy()

        joined_data['delay_submission_to_final'] = joined_data['finaltimestamp'] - (joined_data['job_submit_timestamp'] * 1000)
        joined_data['delay_submission_to_final_minutes'] = joined_data['delay_submission_to_final'] / 1000 / 60
        joined_data['delay_submission_to_final_hours'] = joined_data['delay_submission_to_final'] / 1000 / 3600

        joined_data['delay_submission_to_started'] = joined_data['startedtimestamp'] - (joined_data['job_submit_timestamp'] * 1000)
        joined_data['delay_submission_to_started_minutes'] = joined_data['delay_submission_to_started'] / 1000 / 60
        joined_data['delay_submission_to_started_hours'] = joined_data['delay_submission_to_started'] / 1000 / 3600
        
        return joined_data
    
    @staticmethod
    def _parsing_col_job_tags(joined_data: pd.DataFrame):
        
        def categorize_data(patterns_map: dict, jobtag_list):
            # Extract first comment if list, else assume it's a single string
            comment = jobtag_list[0] if isinstance(jobtag_list, list) and jobtag_list else jobtag_list
            
            if not isinstance(comment, str):
                return np.nan  # Skip invalid data

            matched_types = {
                file_type for file_type, patterns in patterns_map.items()
                if any(re.search(pattern, comment, re.IGNORECASE) for pattern in patterns)
            }

            return str(sorted(matched_types)) if matched_types else np.nan  # Ensuring order and uniqueness

        #We are processing those to string beacuse important is all table and not idependet value.
        joined_data["file_type_category"] = joined_data["jobtag"].map(lambda jobtag_list: categorize_data(JobTagsConfig.FILE_TYPE, jobtag_list)).astype("string")
        joined_data["working_group"] = joined_data["jobtag"].map(lambda jobtag_list: categorize_data(JobTagsConfig.WORKING_GROUP, jobtag_list)).astype("string")
        joined_data["collision_type"] = joined_data["jobtag"].map(lambda jobtag_list: categorize_data(JobTagsConfig.COLLISION_TYPE, jobtag_list)).astype("string")
        
        return joined_data
    
    @staticmethod
    def _parsing_arguments(joined_data: pd.DataFrame) -> pd.DataFrame:
        joined_data['arguments'] = joined_data['arguments'].fillna("").astype(str)

        for column, pattern in ArgumentsConfig.ARGUMENT_PATTERNS.items():
            joined_data[column] = joined_data['arguments'].str.extract(pattern)[0].astype('float64')

        return joined_data

    @staticmethod
    def _parsing_energy_levels(joined_data: pd.DataFrame) -> pd.DataFrame:
    
        joined_data["energy_level"] = (
            joined_data["jobtag"]
            .astype('string')
            .str.extract(f'({JobTagsConfig.ENERGY_LEVEL})', expand=False)
        )

        joined_data["energy_level_numeric"] = (
            joined_data["energy_level"]
            .str.replace("TeV", "", regex=False)
            .str.strip()
            .astype('float64')
        )

        return joined_data

    @staticmethod
    def _drop_na(joined_data : pd.DataFrame): 
        logger.info(f"After assigning, joined_data shape:{joined_data.shape}")
        joined_data.dropna(axis=1, how='all').drop_duplicates(subset='job_id', keep='last').reset_index()

        logger.info(f"Output joined_data shape: {joined_data.shape}")
        return joined_data

    @staticmethod
    def _add_statistical_predictions(joined_data: pd.DataFrame, sort_column : str = 'startedtimestamp') -> pd.DataFrame:
        joined_data = joined_data.sort_values(by=sort_column).reset_index(drop=True)
        #TODO should this be changed to end job timestamp?

        model = OnlineStatisticalModel()
        predicted_means = []
        predicted_stds = []

        for _, row in tqdm(joined_data.iterrows()):
            pred, std = model.process_row(row, train=True)
            predicted_means.append(pred)
            predicted_stds.append(std)

        joined_data['predicted_weighted_ttl'] = predicted_means
        joined_data['predicted_weighted_ttl_std'] = predicted_stds

        return joined_data
    
    @staticmethod
    def generate_hash():
        file_hash = tools.get_hash_from_file_conntent(file_path=__file__)
        config_hash = tools.get_hash_from_config_file(JobTagsConfig.FILE_TYPE | JobTagsConfig.WORKING_GROUP | JobTagsConfig.COLLISION_TYPE)
        hash_value = tools.combine_hashes(config_hash, file_hash)
        return hash_value