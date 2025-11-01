import os
import re
import sys
import shutil

def nnHeadersGen(src_path, bin_path):
    backends = [i for i in os.listdir(src_path) if i != 'nn.cmake']
    for b in backends:
        path = os.path.join(src_path, b)
        if not os.path.isdir(path):
            continue
        vers = os.listdir(path)
        for ver in vers:
            files = [os.path.join(dp, f).replace(path + '/', '') for dp, dn, fn in os.walk(os.path.join(path, ver)) for f in fn]
            for f in files:
                if f.split('.')[-1] != 'h' and f.split('.')[-1] != 'hpp':
                    continue
                lines = open(os.path.join(path, f), 'r').readlines()
                for i, l in enumerate(lines):
                    if '#include' in l:
                        if '<' in l or '.' not in l or l.strip()[0:2] == '//':
                            continue
                        query = re.search(r'#include "(.*)"', l).group(1)
                        for inc in files:
                            if query in inc and os.path.split(query)[1] == os.path.split(inc)[1]:
                                lines[i] = '#include "' + inc + '"' + '\n'
                                break
                dst_file = os.path.join(os.path.join(bin_path, b), f)
                dir, _ = os.path.split(dst_file)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                with open(dst_file, 'w') as fout:
                    fout.writelines(lines)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise RuntimeError("bad input args")
    src_path, bin_path = sys.argv[1:]

    if not os.path.exists(src_path):
        raise RuntimeError(src_path + " does not exist")
    if os.path.exists(bin_path):
        shutil.rmtree(bin_path)

    nnHeadersGen(src_path, bin_path)