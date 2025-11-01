# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse

def get_model_len(model):
    return os.path.getsize(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="the model file path")
    args = parser.parse_args()
    if not args.model:
        parser.print_help()
        sys.exit()

    len = get_model_len(args.model)
    print(str(len))