import os
import shutil
from docopt import docopt
from aci.train import _main
from aci.utils.io import load_yaml
from aci.utils.compare import compare_folder


def test_train(run_folder='runs/dev/full'):
    parameter_file = os.path.join(run_folder, 'train.yml')
    test_dir = os.path.join(run_folder, 'train')

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    parameter = load_yaml(parameter_file)

    meta = {f'label.{k}': v for k, v in parameter.get('labels', {}).items()}
    _main(meta=meta, output_path=test_dir, **parameter['params'])

    ref_dir = os.path.join(run_folder, 'train')

    compare_folder(test_dir, ref_dir)
    print(f'Successfully tested.')


if __name__ == "__main__":
    test_train()
