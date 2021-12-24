import os
import datetime
import random

def make_unique_dir(dir_base, mode=0o777):
    """
    Creates and returns a unique sub directory inside dir_base.
    The name of the directory will have the format
    yyyy.mm.dd.HH.MM.xxxx, where the first part is a date/time
    string of the current system clock and xxxx is a random
    hexadecimal hash. Inherits behavior from os.makedirs.
    """
    unique_dir = None
    while unique_dir is None:
        now = datetime.datetime.now()
        hash = random.getrandbits(16)
        test_dir = os.path.join(dir_base,"%s.%04x" % (now.strftime("%Y.%m.%d.%H.%M"),hash))
        try:
            os.makedirs(test_dir, mode, exist_ok=False)
            unique_dir = test_dir
        except FileExistsError:
            pass
    return unique_dir

