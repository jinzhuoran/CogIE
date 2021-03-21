"""
@Author: jinzhuan
@File: model.py
@Desc: 
"""
import os
from cogie.utils import load_yaml

ROOT_PATH = os.path.join(os.getenv('HOME'), '.cogie')
ROOT_URL = 'http://159.226.21.226/cog_ie_api/'


def absolute_path(file_path, file_name=None):
    if file_path is None:
        return None
    if file_name is None:
        return os.path.join(ROOT_PATH, file_path)
    else:
        return os.path.join(ROOT_PATH, file_path, file_name)


def load_configuration(file_path='configuration', file_name='basis.yaml'):
    if not os.path.exists(absolute_path(file_path, file_name)):
        os.system('wget -P ' + absolute_path(file_path) + ' ' + os.path.join(ROOT_URL, file_path, file_name))
    config = load_yaml(absolute_path(file_path, file_name))
    return config


def download_model(config):
    check_root()
    if 'path' in config and config['path'] is not None:
        path = config['path']
        if not os.path.exists(absolute_path(path)):
            if 'data' in config and config['data'] is not None:
                data = config['data']
                for file in data.values():
                    if file is not None:
                        os.system(
                            'wget -P ' + absolute_path(path) + ' ' + os.path.join(ROOT_URL, path, file))


def check_root(root_path=ROOT_PATH):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(os.path.join(root_path, 'configuration'))
        os.mkdir(os.path.join(root_path, 'model'))
        load_configuration()


def check_file(file_path):
    if os.path.exists(file_path):
        return True
    else:
        return False
