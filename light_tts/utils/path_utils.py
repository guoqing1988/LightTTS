import os


def trans_relative_to_abs_path(base_dir, dest_path):
    if type(dest_path) == list:
        return [trans_relative_to_abs_path(base_dir, dp) for dp in dest_path]
    return os.path.normpath(os.path.join(base_dir, dest_path))
