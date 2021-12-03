"""Usage: test.py RUN_FOLDER

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""
import os
import shutil
from docopt import docopt
from aci.post import _main
from aci.utils.compare import compare_folder
from aci.utils.io import load_yaml


def test_post(run_folder='runs/tests/seed'):
    parameter_file = os.path.join(run_folder, 'test_post.yml')

    test_dir = os.path.join(run_folder, 'test_post')

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    parameter = load_yaml(parameter_file)

    _main(job_folder=run_folder, out_folder=test_dir, cores=1,
          labels=parameter['labels'], **parameter['params'])

    ref_dir = os.path.join(run_folder, 'post')

    compare_folder(test_dir, ref_dir)
    print(f'Successfully tested.')


if __name__ == "__main__":
    arguments = docopt(__doc__)
    run_folder = arguments['RUN_FOLDER']
    test_post(run_folder)
