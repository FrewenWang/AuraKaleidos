# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import mmap
import os
import sys

# embed model data in the lib file (.so)
def embed_models(lib_file, model_file):
    with open(model_file, "rb+") as f:
        blob = f.read()

    with open(lib_file, "rb+") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        pos = m.find(b'_IOVVASDK_MODEL_RESOURCE')
        if pos == -1:
            print("model data region was NOT FOUND inside the library file!")
            return False
        else:
            print("FOUND model data region in the library file, at position: " + str(pos))

        m.seek(pos + 32)
        m.write(blob)
        m.close()
        print("embed model data into the library file DONE!")

def main(model_file, lib_file):
    if not model_file or not os.path.isfile(model_file):
        print("model_file is NOT ACCESSIBLE! model_file: " + model_file)
        return
    if not lib_file or not os.path.isfile(lib_file):
        print("lib_file is NOT ACCESSIBLE! lib_file: " + lib_file)
        return

    embed_models(lib_file, model_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="the merged model")
    parser.add_argument("--lib", help="the dynamic library file (*.so)")
    args = parser.parse_args()
    if not args.model or not args.lib:
        parser.print_help()
        sys.exit()

    main(args.model, args.lib)