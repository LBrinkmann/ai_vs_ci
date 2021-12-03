"""Usage: train.py RUN_FOLDER

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

from docopt import docopt
import os
import pandas as pd
from aci.utils.io import ensure_dir
from itertools import groupby


def find_files_with_extension(_dir, ext):
    for root, dirs, files in os.walk(_dir):
        for file in files:
            if file.endswith(ext):
                yield os.path.join(root, file)


def merge_files(files):
    return pd.concat(
        pd.read_parquet(f)
        for f in files
    )


def merge(run_folder):
    print(f'Start with {run_folder}')

    exp_dir = os.path.join(run_folder)
    out_folder = os.path.join(run_folder, 'merge')

    ensure_dir(out_folder)

    preprocess_folder = os.path.join(run_folder, 'post')
    files_g = find_files_with_extension(preprocess_folder, ".pgzip")

    files_g = list(files_g)
    # print(files_g[1])
    files_g = sorted(files_g, key=lambda f: f.split('.')[-2])
    files_grouped = groupby(files_g, lambda f: f.split('.')[-2])

    for name, files in files_grouped:
        files = list(files)
        # print(files)
        df = merge_files(files)
        filename = os.path.join(out_folder, f'{name}.pgzip')
        df.to_parquet(filename)


def main():
    arguments = docopt(__doc__)
    run_folder = arguments['RUN_FOLDER']

    merge(run_folder)


if __name__ == "__main__":
    main()
