import sys
import os
import re


class CodeGenerator(object):
    def __init__(self, fname, inc_dirs, src_dirs, dst_dir):
        self._params          = {}
        self._params['fname'] = fname
        self._parse_args(inc_dirs, src_dirs, dst_dir)

    def valid(self):
        pattern = r'^[a-zA-Z0-9\.\-\_]+$'
        if (not self._params['fname']) or (not re.match(pattern, self._params['fname'])):
            print('error : fname', self._params['fname'], 'is invalid')
            return False

        for path in self._params['inc_dirs']:
            if not os.path.exists(path):
                print('error : inc dir', path, 'is invalid')
                return False

        for path in self._params['src_dirs']:
            if not os.path.exists(path):
                print('error : src dir', path, 'is invalid')
                return False

        return True

    def _parse_args(self, inc_dirs, src_dirs, dst_dir):
        inc_dir_set = set()
        src_dir_set = set()

        for path in inc_dirs:
            inc_dir_set.add(os.path.abspath(path))

        for path in src_dirs:
            src_dir_set.add(os.path.abspath(path))

        self._params['inc_dirs'] = list(inc_dir_set)
        self._params['src_dirs'] = list(src_dir_set)
        self._params['dst_dir']  = os.path.abspath(dst_dir)

    def _format(self, code_str, indent=0):
        lines       = code_str.split('\n')
        align_idx   = 9999
        space_acc   = 0
        lines_strip = []
        for i, l in enumerate(lines):
            if l.isspace() or '' == l:
                if space_acc > 0 or (not lines_strip):
                    continue
                else:
                    lines_strip.append(l)
                    space_acc += 1
            else:
                align_idx = min(len(l) - len(l.lstrip()), align_idx)
                lines_strip.append(l)
                space_acc = 0
        return '\n'.join(['    ' * indent + l[align_idx:] for l in lines_strip])

    def _wirte(self, code_str, path, fname):
        file_dir = os.path.join(self._params['dst_dir'], path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(os.path.join(file_dir, fname), 'w') as f:
            f.write(self._format(code_str).strip())

    def _deduplicate(self, heads, target):
        pattern = r'#include\s*["<]\s*(.*?)\s*[">]'
        result = []
        for head in heads:
            match = re.search(pattern, head)
            fname = ''
            if match:
                fname = match.group(1)
            for path in self._params['inc_dirs']:
                fpath = os.path.join(path, fname)
                if os.path.exists(fpath):
                    if target == fpath:
                        result.append(head)

        if len(result) > 1:
            print('The cpp file include same head file')

        if not result:
            print('Can not find head file, exiting function')
            return result

        return result[0]

    def _get_inc_string(self, context):
        inc_pattern = r'^\s*#\s*include\s*(<|"|\')([^"\'>]+)(>|"|\').*$'
        result      = {}

        matches = re.findall(inc_pattern, context, re.MULTILINE)
        for match in matches:
            fname = os.path.basename(match[1].strip())
            inc_string = '#include {}{}{}'.format(match[0], match[1].strip(), match[2])
            if fname in result:
                result[fname].append(inc_string)
            else:
                inc_list = []
                inc_list.append(inc_string)
                result[fname] = inc_list

        return result

    def _get_file_paths(self, pattern, paths):
        fpaths = set()
        for path in paths:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith(pattern):
                            fpath = os.path.join(root, file)
                            fpaths.add(fpath)
            elif os.path.isfile(path):
                if path.endswith(pattern):
                    fpaths.add(path)
            else:
                print('Path', path, 'is neither a file nor a directory.')

        return list(fpaths)

    def _search_cpp(self, hpp_paths):
        pattern     = r'AURA_XTENSA_RPC_FUNC_REGISTER\s*\(\s*["\']?([^"\']+)["\']?\s*,\s*([^)]+)\s*\)'
        rpc_map     = []
        inc_strings = set()

        if not hpp_paths:
            print('error : not search any hpp')
            return [], []

        fpaths = self._get_file_paths('vdsp.cpp', self._params['src_dirs'])

        for fpath in fpaths:
            with open(fpath, 'r') as f:
                context  = f.read()
                inc_dict = self._get_inc_string(context)

                matches = re.findall(pattern, context)
                for match in matches:
                    if len(match) != 2:
                        print('AURA_XTENSA_RPC_FUNC_REGISTER params error, exiting function')
                        return [], []

                    rpc = match[1]
                    hpp_path = hpp_paths.get(rpc)
                    if hpp_path is None:
                        print('no hpp file declare this rpc, exiting function')
                        return [], []

                    rpc_map.append((match[0], match[1]))
                    inc_list = inc_dict[os.path.basename(hpp_path.strip())]
                    if inc_list is None:
                        print('cpp code not include the hpp file, exiting function')
                        return [], []

                    if len(inc_list) > 1:
                        inc_string = self._deduplicate(inc_list, hpp_path)
                        inc_list.clear()
                        inc_list.append(inc_string)

                    inc_strings.add(inc_list[0])

        return rpc_map, list(inc_strings)

    def _search_hpp(self):
        suffix_pattern = r'Rpc\s*\(\s*TileManager\s*\w*\s*,\s*XtensaRpcParam\s*\&\s*\w*\s*\)'
        pattern        = rf'\b([a-zA-Z_]\w*){suffix_pattern}'
        result         = {}

        fpaths = self._get_file_paths('.hpp', self._params['inc_dirs'])

        for fpath in fpaths:
            with open(fpath, 'r') as f:
                content = f.read()
                matches = re.findall(pattern, content)
                if matches:
                    rpcs = set('{}Rpc'.format(match) for match in matches)
                    for rpc in rpcs:
                        result[rpc] = fpath

        return result

    def _gen_code_core(self, rpc_map):
        result = []
        result.append('static XtensaRpcFuncRegister g_rpc_func_map[] = {\n')
        for key, value in rpc_map:
            result.append(f'    {{\"{key}\", {value}}},\n')
        result.append("};\n\n")

        return result

    def _gen_inc_code(self, inc_strings):
        result = []
        for inc in inc_strings:
            result.append(''.join(inc))
            result.append('\n')

        return result

    def generate(self):
        fname     = self._params['fname']
        hpp_paths = self._search_hpp()

        rpc_map, inc_strings = self._search_cpp(hpp_paths)
        if (not rpc_map) or (not inc_strings):
            print('error : no op register or declare the rpc func')

        inc_string = self._gen_inc_code(inc_strings)
        inc_string = self._format(''.join(inc_string), 2).strip()

        rpc_map    = self._gen_code_core(rpc_map)
        code_core  = self._format(''.join(rpc_map), 2).strip()

        code_str = f'''
        #ifndef AURA_XTENSA_{fname.upper()}_HPP__
        #define AURA_XTENSA_{fname.upper()}_HPP__

        #include <stdio.h>

        #include "aura/runtime/xtensa/device/xtensa_runtime.hpp"
        {inc_string}

        namespace aura
        {{
        namespace xtensa
        {{

        {code_core}

        }} // namespace xtensa
        }} // namespace aura

        #endif // AURA_XTENSA__{fname.upper()}_HPP__
        '''

        self._wirte(code_str, '', '{}.hpp'.format(fname.lower()))


if __name__ == '__main__':
    fname     = sys.argv[1]
    inc_dirs  = (sys.argv[2]).replace(';', ' ').split()
    src_dirs  = (sys.argv[3]).replace(';', ' ').split()
    dst_dir   = sys.argv[4]

    gen = CodeGenerator(fname, inc_dirs, src_dirs, dst_dir)
    if gen.valid():
        gen.generate()
        print('\033[1;32mRpc code generate done.\033[0m')
    else:
        print('\033[1;31mRpc code generate failed.\033[0m')