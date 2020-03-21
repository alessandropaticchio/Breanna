import os
def to_ospath(path_as_str):
    return os.path.join(path_as_str)

import shutil, errno
def copyfromto(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: 
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise