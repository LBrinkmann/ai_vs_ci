"""Usage: train.py RUN_FOLDER

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

from docopt import docopt
import pandas as pd
import os
import shutil
import numpy as np
from aci.utils.df_utils import (
    selector, expand_selectors, summarize_df, preprocess)
from aci.utils.io import load_yaml, ensure_dir


def load_and_preprocess(input_files, filename, **preprocess_args):
    df = pd.read_parquet(input_files[filename])
    return preprocess(df, **preprocess_args)


def _main(*, input_files, clean, preprocess_args=None, output_path):
    if clean:
        shutil.rmtree(output_path, ignore_errors=True)
    ensure_dir(output_path)

    input_files = {
        os.path.split(f)[-1].split('.')[-2]: f
        for f in input_files
    }

    df = pd.concat(
        load_and_preprocess(**pa, input_files=input_files)
        for name, pa in preprocess_args.items()
    )

    summarize_df(df, 'merged')



def main():
    arguments = docopt(__doc__)
    run_folder = arguments['RUN_FOLDER']
    parameter_file = os.path.join(run_folder, 'causality.yml')

    out_folder = os.path.join(run_folder, 'causality')
    ensure_dir(out_folder)
    parameter = load_yaml(parameter_file)

    input_files = os.listdir(os.path.join(run_folder, 'merge'))
    input_files = [os.path.join(run_folder, 'merge', f) for f in input_files]

    _main(output_path=out_folder, input_files=input_files, **parameter)


if __name__ == "__main__":
    main()
