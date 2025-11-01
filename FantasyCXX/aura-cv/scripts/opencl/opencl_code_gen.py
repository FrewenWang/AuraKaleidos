import os
import re
import sys
import shutil
from typing import List, Tuple

class CodeGenerator:
    def __init__(self, dirs: List[str]):
        self._inc_dir, self._cl_dir, self._cpp_dir, self._rt_dir, self._tmp_file = dirs

        if not os.path.exists(self._inc_dir):
            self._exit()
            raise RuntimeError(f'{self._inc_dir} does not exist')
        if not os.path.exists(self._cl_dir):
            self._exit()
            raise RuntimeError(f'{self._cl_dir} does not exist')

        if os.path.exists(self._cpp_dir):
            shutil.rmtree(self._cpp_dir)
        os.makedirs(self._cpp_dir)

        if os.path.exists(self._rt_dir):
            shutil.rmtree(self._rt_dir)
        os.makedirs(self._rt_dir)

    def _remove_comments(self, code: str) -> str:
        code = re.sub('/\*(.*?)\*/', '', code, flags=re.DOTALL)
        code = re.sub('//.*', '', code)
        return code

    def _resolve_incs(self, line: str, files: List[str], inc_list: List[str]) -> None:
        inc_file_name = line.split('#include')[-1].strip()[1:-1]
        find_flag = False
        for file in files:
            file_tmp = file
            if inc_file_name in file and file_tmp.replace(inc_file_name, '')[-1] == '/':
                find_flag = True
                inc_file_path = file
                break

        if not find_flag:
            self._exit()
            raise RuntimeError(f'{inc_file_name} not found')
        else:
            with open(inc_file_path, 'r') as f:
                code_lines = f.readlines()

            for line in code_lines:
                if '#' in line and 'include' in line:
                    self._resolve_incs(line, files, inc_list)

            name = inc_file_name.split('.')[0] + '_inc'
            if name not in inc_list:
                inc_list.append(name)

    def _get_code_str(self, fpath: str, files: List[str]) -> Tuple[str, List[str]]:
        with open(fpath, 'r') as f:
            code_str = f.read()

        inc_list = []
        code_str = self._remove_comments(code_str)
        code_lines = code_str.split('\n')

        code_new = ''
        for line in code_lines:
            if '#' in line and 'include' in line:
                self._resolve_incs(line, files, inc_list)
            elif not line.isspace() and line != '':
                line = self._remove_space(line)
                code_new += line

        return code_new, inc_list

    def _remove_space(self, line: str) -> str:
        chars = ('+', '-', '*', '/', '%', '=', '<', '>', '~', '^', '|', '#',
                 '&', '!', '?', ':', '[', ']', '(', ')', ',', ';', '{', '}')

        ch_set = set()
        line = line.strip()
        for c in line:
            if c in chars:
                ch_set.add(c)

        for c in ch_set:
            if c == '(' and '#' in line and 'define' in line:
                continue
            pattern = r'\s*{}\s*'.format(re.escape(c))
            line = re.sub(pattern, c, line)

        line = re.sub(r'\s+', ' ', line)
        return line + '\n'

    def generate(self) -> None:
        inc_files = [os.path.join(dp, f) for dp, _, fn in os.walk(self._inc_dir) for f in fn]
        cl_files = [os.path.join(dp, f) for dp, _, fn in os.walk(self._cl_dir) for f in fn]
        all_files = inc_files + cl_files

        for fpath in all_files:
            fname = os.path.basename(fpath)

            code_str, incs = self._get_code_str(fpath, all_files)
            name, cpp_string = self._get_cpp_string(fname, code_str, incs)

            dir = self._cpp_dir if fpath in cl_files else self._rt_dir
            cpp_name = fname.split('.')[0] + '_cl.cpp' if '.cl' in fname else fname.split('.')[0] + '_inc.cpp'
            with open(os.path.join(dir, cpp_name), 'w') as f:
                f.write(cpp_string)

            inc_string = f'    extern aura::CLProgramString g_ops_{name};\n'
            inc_string += f'    g_ops_{name}.Register();\n\n'
            with open(self._tmp_file, 'a+') as f:
                f.write(inc_string)

    def _exit(self) -> None:
        if os.path.exists(self._tmp_file):
            os.remove(self._tmp_file)

    def _get_cpp_string(self, fname: str, code: str, incs: List[str]) -> Tuple[str, str]:
        name = fname.split('.')[0] if '.cl' in fname else fname.split('.')[0] + '_inc'
        inc_str = ', '.join([f'"{inc}"' for inc in incs])

        code_str = ''
        for c in code:
            hex_ = hex(ord(c))
            code_str += hex_ + ', '
        code_str += '0x0'

        source = f"""
            #include \"aura/runtime/opencl/cl_runtime.hpp\"

            static const MI_CHAR g_cl_program_string[]
            {{
                {code_str}
            }};

            aura::CLProgramString g_ops_{name}("{name}", g_cl_program_string, {{{inc_str}}});
        """.strip()

        source = [line[12:] if line != '' and line[0] == ' ' else line for line in source.split('\n')]
        return name, '\n'.join(source)

    @classmethod
    def gen_inc_file(cls, src: str, dst: str) -> None:
        src_lines = ''

        if os.path.exists(src):
            with open(src, 'r') as f:
                src_lines = f.read()

        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        with open(dst, 'w') as f:
            code_str = '#include \"aura/runtime/opencl/cl_runtime.hpp\"\n\n'
            code_str += 'AURA_INLINE aura::Status CLProgramStringRegister()\n{\n'
            code_str += src_lines
            code_str += '    return aura::Status::OK;\n};'
            f.write(code_str)

        if os.path.exists(src):
            os.remove(src)


if __name__ == '__main__':
    if len(sys.argv) != 6 and len(sys.argv) != 3:
        raise RuntimeError('bad input args')

    if len(sys.argv) == 6:
        gen = CodeGenerator(sys.argv[1:])
        gen.generate()
    else:
        CodeGenerator.gen_inc_file(sys.argv[1], sys.argv[2])