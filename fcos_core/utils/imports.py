# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

if torch._six.PY3:  #True
    import importlib
    import importlib.util
    import sys

    # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    #通过知道模型的名称和路径，来导入模型
    def import_file(module_name, file_path, make_importable=False):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if make_importable:
            sys.modules[module_name] = module
        return module
else:
    import imp

    def import_file(module_name, file_path, make_importable=None):
        module = imp.load_source(module_name, file_path)
        return module

paths_catalog = import_file(
    "fcos_core.config.paths_catalog", "/home/sifan/slam-package/FCOS/fcos_core/config/paths_catalog.py", True)
print(dir(paths_catalog))