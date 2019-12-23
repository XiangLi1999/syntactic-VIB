#!/usr/bin/python
# --------------------------------------- 
# File Name : mlie.py
# Creation Date : 20-11-2016
# Last Modified : Thu Apr 19 23:52:09 2018
# Created By : wdd 
# ---------------------------------------
import inspect
import logging
import os
import sys
from functools import wraps


def get_logger(level=logging.INFO):
    logger = logging.getLogger(os.path.basename(inspect.getouterframes(inspect.currentframe())[1][1]))
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s-%(name)s[%(levelname)s]$ %(message)s', '%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger()


# A function wrapper
def experiment(before=[], after=[]):
    def decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            for bfr in before: bfr()
            func(*args, **kwargs)
            for afr in after: afr()

        return decorated

    return decorator


class VarDict(object):
    @staticmethod
    def _setattr_(obj, key, val):
        if key.endswith('__'):
            key = key[:-2]
        elif key in obj.my_dict:
            logger.info('re-assign glb.%s' % key)
        obj.my_dict[key] = val

    @staticmethod
    def _getattr_(obj, key):
        if key.endswith('__'):
            key = key[:-2]
        return obj.my_dict[key]

    def __init__(self, dict=None):
        self.__dict__['my_dict'] = {}
        if dict:
            for key, val in dict.items():
                self.__setattr__(key, val)

    def __setattr__(self, key, value):
        VarDict._setattr_(self, key, value)

    def __getattr__(self, key):
        return VarDict._getattr_(self, key)

    def __str__(self):
        return '\n'.join('{0}:{1}'.format(key, val) for (key, val) in sorted(self.my_dict.items()))

    def to_dict(self):
        return self.my_dict

    def add(self, dict):
        for key, val in dict.items():
            self.__setattr__(key, val)


Global = VarDict()
Option = VarDict()
