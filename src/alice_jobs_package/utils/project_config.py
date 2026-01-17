import re
from enum import Enum
from pathlib import Path

class RawDataFiles(Enum):
    MON_JDLS = Path('mon_jdls_parsed.csv')
    JOB_INFO = Path('job_info.csv')
    TRACE = Path('trace.csv')

    @property
    def regex(self) -> re.Pattern:
        stem = self.value.stem
        suffix = self.value.suffix
        return re.compile(rf"^{stem}(_\d+)?\{suffix}$")

class RawDataSubFolders(Enum):
    SITE_SONARS = Path('site-sonars-all')
    RAW_CSV = Path('raw_csv')

class JobTagsConfig():
    ENERGY_LEVEL = r'\b\d+(?:\.\d+)?\s*TeV\b'

    FILE_TYPE = {
        "AOD": [r"AOD"], 
        "ESD": [r"ESD"],
    }
    
    WORKING_GROUP = {
        "PWGCF": [r"PWGCF"], 
        "PWGDQ": [r"PWGDQ"], 
        "PWGGA": [r"PWGGA"], 
        "PWGHF": [r"PWGHF"], 
        "PWGHM": [r"PWGHM"], 
        "PWGJE": [r"PWGJE"], 
        "PWGLF": [r"PWGLF"], 
        "PWGMM": [r"PWGMM"], 
        "PWGPP": [r"PWGPP"], 
        "PWGUD": [r"PWGUD"], 
        "PWGZZ": [r"PWGZZ"],
        "MC": [r"MC", r"Monte Carlo"],
        "General Purpose": [r"General Purpose"],
        "User Jobs": [r"Automatically generated analysis JDL"],
        "Hyperloop": [r"Hyperloop analysis"]
    }

    COLLISION_TYPE = {
        "Pb-Pb": [r"Pb-Pb", r"PbPb"], 
        "p-Pb": [r"p-Pb", r"pPb"],
        "p-p": [r"p-p", r"pp"]
    }

class ArgumentsConfig():
    ARGUMENT_PATTERNS = {
        'nevents': r'--nevents (\d+)',
        'nsigevents': r'NSIGEVENTS=(\d+)',
        'ntimeframes': r'NTIMEFRAMES=(\d+)',
    }

class CacheDataFolders(Enum):
    CACHE = Path('.cache')
    CACHE_JOINED_DATA_FOLDER = Path('.cache/joined_data')
    CACHE_DATA_CONFIG_FOLDER = Path('.cache/preprocessor_config')
    CACHE_DATA_PROCESSED_FOLDER = Path('.cache/processed_data')

class PreprocessedDataFiles(Enum):
    JOINED_DATA = Path('joined_data.npz')
    JOINED_SITE_SONAR = Path('joined_site_sonar.npz')
    OUTPUT_DATA = Path('output_data.npz')

class ProcessingTarget(Enum):
    MLP = 'MLP'
    MLP_EMBEDINGS = 'MLP_EMBEDINGS'
    TRANSFORMER = 'TRANSFORMER'
    TRANSFORMER_EMBEDINGS = 'TRANSFORMER_EMBEDINGS'
    NO_PROCESSING = 'NO_PROCESSING'

class ProcessingType(Enum):
    ONE_HOT_ENCODED = 'ONE_HOT_ENCODED'
    EMBEDINGS_ENCODED = 'EMBEDINGS_ENCODED'
    NO_PROCESSING = 'NO_PROCESSING'

class ModelCacheSubFolders(Enum):
    SAVED_MODEL = Path('saved_model')
    SAVED_PLOTS = Path('saved_plots')
    SAVED_HISTORY = Path('saved_history')
 
PROCESSING_TARGET_MAP = {
   ProcessingTarget.MLP.name : ProcessingType.ONE_HOT_ENCODED.name,
   ProcessingTarget.TRANSFORMER.name : ProcessingType.ONE_HOT_ENCODED.name,
   ProcessingTarget.MLP_EMBEDINGS.name : ProcessingType.EMBEDINGS_ENCODED.name,
   ProcessingTarget.TRANSFORMER_EMBEDINGS.name : ProcessingType.EMBEDINGS_ENCODED.name,
   ProcessingTarget.NO_PROCESSING.name : ProcessingType.NO_PROCESSING.name
}

UNKNOWN_COLUMN_NAME = 'unknown_col'
UNKNOWN_VALUE_NAME = 'unknown'

class ArgsMode(Enum):
    FILE = 'FILE'
    CMD_LINE = 'CMD_LINE'