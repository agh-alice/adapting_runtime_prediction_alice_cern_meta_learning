# CAN BE RUN ON ACCESS NODE, ONLY STORAGE INTENSIVE

import pyarrow as pa
from time import time
from pathlib import Path
from enum import Enum
from tqdm import tqdm
from datetime import datetime, timezone
from pyarrow import flight

import pyarrow.parquet as pq
import pandas as pd
import argparse
import logging

import os
import re
import time
import requests

# CONFIG
logging.basicConfig(level=logging.INFO)

import builtins
original_print = print
def flush_print(*args, **kwargs):
    return original_print(*args, flush=True, **kwargs)
builtins.print = flush_print

URL_FOR_SS = ""
BASE_URL = ""
BATCHES = 10

DREMIO_CONFIG = {
    'hostname': '',
    'port': '',
    'username': '',
    'password': ''
}

tables_on_dremio_to_download = [
    "mon_jdls_parsed",
    "job_info",
	"trace"
]

# CLASSES
class DataType(Enum):
  BOTH = 'BOTH'
  SITE_SONAR = 'SITE_SONAR'
  DREMIO = 'DREMIO'

# METHODS
def get_year_from_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).year

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--destinstion-path", dest="destination_path", type=str, default=Path('./download'), 
                        required=False, help="Path to folder where downloaded files will be saved. Defaults to 'downloaded_data' folder in script folder.")
    
    parser.add_argument("--site_sonar_year", dest="site_sonar_year", type=int, default=2024, required=False,
        help="Year from which all site-sonar data shuld be downloaded")
    
    parser.add_argument("--data_type", dest="data_type", type=DataType, default=DataType.BOTH, required=False,
                        help="Choose data which should be downloaded, BOOTH, SITE-SONAR, DREMIO")
    
    parser.add_argument("--dremio_data_limit", dest="dremio_data_limit", type=int, default=None, required=False,
                        help="Set limit for amount of records to download from dremio")
    
    parser.add_argument("--chunked_csv", dest="chunks", type=int, default=None,
                        help="If set then all files will be in chunks")
    
    parser.add_argument("--to_csv", help="If set dremio output file will be csv via pandas", action='store_true')

    parser.add_argument("--limit_job_id", dest="limit_job_id", type=int, default=0, help="If set then this argument says what the smallest job id in dataset could be")

    return parser.parse_args()

def download_tables_from_dremio(args):
    client = flight.FlightClient(f"grpc+tcp://{DREMIO_CONFIG['hostname']}:{DREMIO_CONFIG['port']}")
    auth_token = client.authenticate_basic_token(DREMIO_CONFIG['username'], DREMIO_CONFIG['password'])
    options = flight.FlightCallOptions(headers=[auth_token])

    if args.chunks:
        BATCHES = args.chunks

    def get_percentile_bounds(tab, batches, limit_job_id):
        percentile_points = [i / batches for i in range(batches + 1)]
        percentile_selects = ",\n    ".join(
            [f"APPROX_PERCENTILE(job_id, {round(p, 4)}) AS p{int(p * 100)}" for p in percentile_points]
        )
        query = f'''
            SELECT
                {percentile_selects}
            FROM Alice."{tab}"
            WHERE job_id >= {limit_job_id}
        '''

        flight_info = client.get_flight_info(flight.FlightDescriptor.for_command(query), options)
        reader = client.do_get(flight_info.endpoints[0].ticket, options)
        record_batch = next(reader).data
        table = pa.Table.from_batches([record_batch])
        df = table.to_pandas()
        return [int(df[f'p{int(p * 100)}'][0]) for p in percentile_points]

    def download_range(tab, start_id, end_id, part_name, depth=0):
        if end_id <= start_id:
            print(f"    ⚠️ Skipping empty range {start_id} - {end_id}")
            return []

        query = f'''
            SELECT * FROM Alice."{tab}"
            WHERE job_id >= {start_id} AND job_id < {end_id}
        '''
        if args.dremio_data_limit:
            query += f' LIMIT {args.dremio_data_limit}'

        indent = '  ' * (depth+1)
        print(f"{indent}🔹 Downloading job_id {start_id} - {end_id - 1}")
        t0 = time.time()

        try:
            flight_info = client.get_flight_info(flight.FlightDescriptor.for_command(query), options)
            reader = client.do_get(flight_info.endpoints[0].ticket, options)

            batch_list = [chunk.data for chunk in reader if chunk.data.num_rows > 0]

            if not batch_list:
                print(f"{indent}⚠️ No data in range {start_id} - {end_id}")
                return []

            table_chunk = pa.Table.from_batches(batch_list)
            df = table_chunk.to_pandas()

            if df.empty:
                print(f"{indent}⚠️ Empty dataframe in range {start_id} - {end_id}")
                return []

            chunk_path = f"{args.destination_path}/raw_csv/{tab}_{part_name}.csv"
            df.to_csv(chunk_path, index=False)

            t1 = time.time()
            print(f"{indent}✅ Saved {part_name} ({len(df)} rows) in {t1 - t0:.2f}s")
            return [chunk_path]  # ✅ MOD: always return a list

        except Exception as e:
            t1 = time.time()
            if end_id - start_id <= 1:
                print(f"{indent}❌ Final split failed for range {start_id}-{end_id} after {t1 - t0:.2f}s: {e}")
                return []
            else:
                print(f"{indent}⚠️ Error. Splitting range {start_id}-{end_id} further...")
                mid_id = (start_id + end_id) // 2
                path1 = download_range(tab, start_id, mid_id, f"{part_name}_a", depth + 1)
                path2 = download_range(tab, mid_id, end_id, f"{part_name}_b", depth + 1)

                result = []
                for p in (path1, path2):
                    if isinstance(p, list):
                        result.extend(p)
                    elif p:
                        result.append(p)
                return result

    for tab in tables_on_dremio_to_download:
        print(f"\n📥 Starting download: {tab}")
        chunk_paths = []

        try:
            pct_bounds = get_percentile_bounds(tab, BATCHES, args.limit_job_id)
        except Exception as e:
            print(f"❌ Failed to compute percentiles for table {tab}: {e}")
            continue

        chunk_paths = []
        total_rows = 0 

        for i in range(BATCHES):            
            start_id, end_id = pct_bounds[i], pct_bounds[i + 1]
            result = download_range(tab, start_id, end_id, f"{i}")

            # Merge all chunks for this batch
            if len(result) > 1:
                print(f"🔹 Saving batch {i+1}/{BATCHES}")
                dfs = []
                for p in result:
                    df = pd.read_csv(p, low_memory=False)
                    total_rows += len(df)
                    dfs.append(df)
                    os.remove(p)  # cleanup small chunks

                batch_df = pd.concat(dfs, ignore_index=True)
                batch_path = f"{args.destination_path}/raw_csv/{tab}_{i}.csv"
                batch_df.to_csv(batch_path, index=False)
                chunk_paths.extend(batch_path)

                print(f"📦 Saved batch {i+1}/{BATCHES} → {batch_path} ({len(batch_df)} rows)")

        # Final save after all batches
        if chunk_paths and not args.chunks:
            print(f"🔗 Combining {len(chunk_paths)} parts into one CSV")
            save_start = time.time()

            dfs = []
            for p in chunk_paths:
                df = pd.read_csv(p, low_memory=False)
                total_rows += len(df)
                dfs.append(df)

            print(f"🔢 Total records (rows): {total_rows}")

            full_df = pd.concat(dfs, ignore_index=True)
            final_path = f"{args.destination_path}/{tab}.{'csv' if args.to_csv else 'parquet'}"

            if args.to_csv:
                full_df.to_csv(final_path, index=False)
            else:
                pq.write_table(pa.Table.from_pandas(full_df), final_path)

            save_end = time.time()
            print(f"✅ File saved to: {final_path} in {save_end - save_start:.2f}s")

            for p in chunk_paths:
                os.remove(p)
        elif not args.chunks:
            print(f"⚠️ No data downloaded for {tab}, skipping final file.")

        print(f"🏁 Finished downloading table: {tab}")

def download_site_sonar_from_alice(args):
    DOWNLOAD_DIR = Path(args.destination_path + '/site-sonars-all')
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

    response = requests.get(URL_FOR_SS)
    content = response.text

    files = re.findall(r'href="([^"]+\.out\.xz)"', content)

    for file in tqdm(files):
        timestamp_match = re.search(r'(\d+)', file)

        if timestamp_match:
            timestamp = int(timestamp_match.group(1))
            file_year = get_year_from_timestamp(timestamp)

            if file_year >= args.site_sonar_year:
                file_url = f"{BASE_URL}{file}"
                file_path = DOWNLOAD_DIR / Path(file).name

                with requests.get(file_url, stream=True) as r:
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

    print(f"Download of all site-sonar for year {args.site_sonar_year} completed.")


#SCRIPT

if __name__ == "__main__":
    args = parse_args()
    Path(args.destination_path).mkdir(parents=True, exist_ok=True)
    Path(args.destination_path + '/raw_csv').mkdir(parents=True, exist_ok=True)

    if args.data_type != DataType.SITE_SONAR:
        print('Downloading from Dremio')
        download_tables_from_dremio(args)
    if args.data_type != DataType.DREMIO:
        print('Downloading site-sonar')
        download_site_sonar_from_alice(args)
