# Shree KRISHNAya Namaha
# Merges Consolidated Configs and Results
# Author: Nagabhushan S N
# Last Modified: 23/09/2022

import datetime
import time
import traceback
from pathlib import Path

import pandas

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_filenum = int(this_filename[-2:])


def merge_configs_and_results(configs_path: Path, results_path: Path, output_path: Path):
    configs = pandas.read_csv(configs_path)
    results = pandas.read_csv(results_path)
    results_keys = list(results.columns)
    results_keys[0] = 'test_num'
    results.columns = results_keys

    merged_data = configs.merge(results, on='test_num')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_data.to_csv(output_path, index=False)
    return


def demo1():
    configs_path = Path('../../Runs/ConsolidatedStats/TestConfigs01.csv')
    results_path = Path('../../Runs/ConsolidatedStats/TestResults01.csv')
    output_path = Path(f'../../Runs/ConsolidatedStats/TestConfigsAndResults{this_filenum:02}.csv')
    merge_configs_and_results(configs_path, results_path, output_path)
    return


def main():
    demo1()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
