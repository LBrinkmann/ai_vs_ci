import os
from aci.utils.io import load_yaml, relative_file


def load_configs(names, scope):
    return [
        load_yaml(relative_file(scope, os.path.join('configs', f'{n}.yml')))
        for n in names
    ]
