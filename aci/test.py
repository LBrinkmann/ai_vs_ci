"""Usage: test.py RUN_FOLDER

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

from aci.envs.tests.adversial_graph_coloring_historized import test_setting
from aci.utils.io import load_yaml
import os
from docopt import docopt


def main():
    print('start')

    arguments = docopt(__doc__)
    run_dir = arguments['RUN_FOLDER']

    parameter_file = os.path.join(run_dir, 'runtest.yml')
    out_dir = os.path.join(run_dir, 'runtest')
    parameter = load_yaml(parameter_file)
    # ensure_dir(out_dir)
    test_setting(parameter['params']['env_args'])

    print(f'Success: {parameter["labels"]}')


if __name__ == "__main__":
    main()