# Shree KRISHNAya Namaha
# Consolidates train/test config files into a single csv file
# Author: Nagabhushan S N
# Last Modified: 23/09/2022

import datetime
import json
import time
import traceback
from pathlib import Path

import pandas

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_filenum = int(this_filename[-2:])


def consolidate_configs(runs_dirpath: Path) -> pandas.DataFrame:
    all_configs = []
    for run_dirpath in sorted(runs_dirpath.iterdir()):
        if not run_dirpath.is_dir():
            continue
        configs_path = run_dirpath / 'Configs.json'
        with open(configs_path.as_posix(), 'r') as configs_file:
            configs = json.load(configs_file)
        configs = flatten_nested_dicts(configs)
        configs = format_dicts_within_list(configs)
        all_configs.append(configs)
    configs_data = pandas.DataFrame(all_configs)
    return configs_data


def flatten_nested_dicts(nested_configs: dict):
    flattened_configs = {}
    for key in nested_configs.keys():
        if isinstance(nested_configs[key], dict):
            nested_dict = flatten_nested_dicts(nested_configs[key])
            for nested_key in nested_dict.keys():
                full_key = f'{key}_{nested_key}'
                flattened_configs[full_key] = nested_dict[nested_key]
        else:
            flattened_configs[key] = nested_configs[key]
    return flattened_configs


def format_dicts_within_list(raw_configs: dict):
    formatted_configs = {}
    for key in raw_configs.keys():
        if isinstance(raw_configs[key], list) and isinstance(raw_configs[key][0], dict):
            str_value = str(raw_configs[key][0])
            for list_elem in raw_configs[key][1:]:
                str_value += ('\n' + str(list_elem))
            formatted_configs[key] = str_value
        else:
            formatted_configs[key] = raw_configs[key]
    return formatted_configs


def demo1():
    train_dirpath = Path('../../Runs/Training')
    output_path = Path(f'../../Runs/ConsolidatedStats/TrainConfigs{this_filenum:02}.csv')
    consolidated_configs = consolidate_configs(train_dirpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_configs.to_csv(output_path, index=False)
    return


def demo2():
    test_dirpath = Path('../../Runs/Testing')
    output_path = Path(f'../../Runs/ConsolidatedStats/TestConfigs{this_filenum:02}.csv')
    consolidated_configs = consolidate_configs(test_dirpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_configs.to_csv(output_path, index=False)
    return


def main():
    demo1()
    demo2()
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
