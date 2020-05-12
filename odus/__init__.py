import os

root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, 'data')
data_path_of = lambda path: os.path.join(data_dir, path)
