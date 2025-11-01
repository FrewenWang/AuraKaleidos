import os
import sys
import shutil

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("input params: ", sys.argv)
        raise RuntimeError("bad input args")

    #compile
    sdk_path, qaic_path, cur_bin_dir, cur_src_dir, idl_func = sys.argv[1:]
    idl_inc_dirs = "-I{}/incs/stddef -I{}/incs".format(sdk_path, sdk_path)
    qaic_cmd = "-mdll -o {} {}".format(cur_bin_dir, idl_inc_dirs)
    cmd = "{} {} {}/idl".format(qaic_path, qaic_cmd, cur_src_dir) + "/{}"
    status = os.system(cmd.format(idl_func))
    if status != 0:
        raise RuntimeError("qaic failed to compile")

    #define __QAIC_SKEL_EXPORT
    for f in os.listdir(cur_bin_dir):
        ext = f.split('.')[-1]
        if ext != 'c' and ext != 'h':
            continue
        f = os.path.join(cur_bin_dir, f)
        lines = open(f, "r").read()
        if '#define __QAIC_SKEL_EXPORT' in lines:
            with open(f, "w") as fout:
                fout.write(lines.replace("#define __QAIC_SKEL_EXPORT", '#define __QAIC_SKEL_EXPORT __attribute__ ((visibility("default")))'))
