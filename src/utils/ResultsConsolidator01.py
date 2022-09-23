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


def consolidate_results(runs_dirpath: Path) -> pandas.DataFrame:
    all_results = []
    for run_dirpath in sorted(runs_dirpath.iterdir()):
        test_name = run_dirpath.stem
        test_num = int(test_name[4:])
        if not run_dirpath.is_dir():
            continue
        results_path = run_dirpath / 'QA_Scores.json'
        if not results_path.exists():
            continue
        with open(results_path.as_posix(), 'r') as results_file:
            qa_scores = json.load(results_file)
        results_dict = {
            'Test Num': test_num
        }
        for output_type in qa_scores.keys():
            results_dict['Output Folder Name'] = output_type
            results_dict.update(qa_scores[output_type])
        all_results.append(results_dict)
    results_data = pandas.DataFrame(all_results)
    return results_data


def demo1():
    test_dirpath = Path('../../Runs/Testing')
    output_path = Path(f'../../Runs/ConsolidatedStats/TestResults{this_filenum:02}.csv')
    consolidated_results = consolidate_results(test_dirpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_results.to_csv(output_path, index=False)
    return


def demo2():
    """
    Should be put in a new file if used
    :return:
    """
    test_dirpath = Path('../../Runs/Testing')
    output_path = Path(f'../../Runs/ConsolidatedStats/TestResults{this_filenum:02}.csv')
    metrics = ['CroppedMSE01', 'CroppedMaskedMSE01', 'CroppedMaskedMSE02', 'CroppedSSIM01', 'CroppedMaskedSSIM01',
               'CroppedMaskedSSIM02']

    consolidated_results = consolidate_results(test_dirpath)
    metrics = [metric_name for metric_name in metrics if metric_name in consolidated_results.keys()]
    columns = list(consolidated_results.keys())[:2] + metrics
    consolidated_results = consolidated_results[columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_results.to_csv(output_path, index=False)
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
