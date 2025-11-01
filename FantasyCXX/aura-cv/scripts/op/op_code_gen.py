from abc import ABC, abstractmethod
import argparse
import os
import datetime
import calendar
import shutil

class OpCodeBase(ABC):
    def __init__(self, params):
        self._params = params

    @abstractmethod
    def getDeclStr(self):
        pass

    @abstractmethod
    def getImplStr(self):
        pass

class OpCodeImpl(OpCodeBase):
    def __init__(self, params):
        super().__init__(params)
        self.utils = {}
        self.utils['case'] = ['', 0]
        self.utils['check_width'] = ''
        self.utils['comm'] = ''
        self.utils['includes'] = {}
        self.utils['includes']['private'] = ''
        self.utils['includes']['comm'] = ''
        self.utils['includes']['impl'] = {}

        self._decl_func_map = {}
        self._decl_func_map["construct"] = "{}(Context *ctx, const OpTarget &target);"
        self._decl_func_map["SetArgs"] = "Status SetArgs(/* add parameters here */) override;";
        self._decl_func_map["Initialize"] = "Status Initialize() override;";
        self._decl_func_map["DeInitialize"] = "Status DeInitialize() override;";
        self._decl_func_map["Run"] = "Status Run() override;";
        self._decl_func_map["ToString"] = "std::string ToString() const override;";

        self.__getImplFuncMap()

    def __getImplFuncMap(self):
        self._impl_func_map = {}

        self._impl_func_map["construct"] = """
        {}::{}(Context *ctx, const OpTarget &target) : {}Impl(ctx, target)
        {{}}
        """.strip()

        self._impl_func_map["SetArgs"] = """
        Status {}::SetArgs(/* add parameters here */)
        {{
            if ({}Impl::SetArgs(/* add parameters here */) != Status::OK)
            {{
                AURA_ADD_ERROR_STRING(m_ctx, "{}Impl::SetArgs failed");
                return Status::ERROR;
            }}

            /* add code here */

            return Status::OK;
        }}
        """.strip()

        self._impl_func_map["Initialize"] = """
        Status {}::Initialize()
        {{
            /* add code here */
        }}
        """.strip()

        self._impl_func_map["DeInitialize"] = """
        Status {}::DeInitialize()
        {{
            /* add code here */
        }}
        """.strip()

        self._impl_func_map["Run"] = """
        Status {}::Run()
        {{
            /* add code here */
        }}
        """.strip()

        self._impl_func_map["ToString"] = """
        std::string {}::ToString() const
        {{
            return {}Impl::ToString() + m_profiling_string;
        }}
        """.strip()

    def getDeclStr(self):
        class_name = self._params['name'] + "Impl"

        code_str = f"""
        class {class_name} : public OpImpl
        {{
        public:
            {self._decl_func_map["construct"].format(class_name)}

            virtual Status SetArgs(/* add parameters here */);

            {self._decl_func_map["Initialize"]}

            {self._decl_func_map["DeInitialize"]}

            {self._decl_func_map["ToString"]}

            AURA_VOID Dump(const std::string &prefix) const override;

        private:
            /* add code here */

        protected:
            /* add code here */
        }};
        """

        return code_str

    def getImplStr(self):
        class_name = self._params['name'] + "Impl"

        code_str = f"""
        {class_name}::{class_name}(Context *ctx, const OpTarget &target) : OpImpl(ctx, "{class_name[:-4]}", target), /* add code here */
        {{}}

        Status {class_name}::SetArgs(/* add parameters here */)
        {{
            if (MI_NULL == m_ctx)
            {{
                return Status::ERROR;
            }}

            /* add code here */
        }}

        std::string {class_name}::ToString() const
        {{
            std::string str;

            /* add code here */

            return str;
        }}

        AURA_VOID {class_name}::Dump(const std::string &prefix) const
        {{
            /* add code here */
        }}

        {self._impl_func_map["Initialize"].format(class_name)}

        {self._impl_func_map["DeInitialize"].format(class_name)}
        """

        return code_str

class OpCodeImplNone(OpCodeImpl):
    def __init__(self, params):
        super().__init__(params)

        name = self._params['name']
        self.utils['case'] = [f"""
        case TargetType::NONE:
        {{
            impl.reset(new {name}None(ctx, target));
            break;
        }}
        """, 2]

    def getDeclStr(self):
        class_name = self._params['name'] + "None"

        code_str = f"""
        class {class_name} : public {class_name[:-4]}Impl
        {{
        public:
            {self._decl_func_map["construct"].format(class_name)}

            {self._decl_func_map["SetArgs"]}

            {self._decl_func_map["Initialize"]}

            {self._decl_func_map["DeInitialize"]}

            {self._decl_func_map["Run"]}

        private:
            /* add code here */
        }};
        """

        return code_str

    def getImplStr(self):
        class_name = self._params['name'] + "None"

        code_str_dict = {}
        code_str_dict['none'] = f"""
        {self._impl_func_map["construct"].format(class_name, class_name, class_name[:-4])}

        {self._impl_func_map["SetArgs"].format(class_name, class_name[:-4], class_name[:-4])}

        Status {class_name}::Initialize()
        {{
            if ({class_name[:-4]}Impl::Initialize() != Status::OK)
            {{
                AURA_ADD_ERROR_STRING(m_ctx, "{class_name[:-4]}Impl::Initialize() failed");
                return Status::ERROR;
            }}

            /* add code here */
        }}

        {self._impl_func_map["Run"].format(class_name)}

        {self._impl_func_map["DeInitialize"].format(class_name)}
        """

        return code_str_dict

class OpCodeImplNeon(OpCodeImpl):
    def __init__(self, params):
        super().__init__(params)

        name = self._params['name']
        self.utils['case'][0] = f"""
                case TargetType::NEON:
                {{
        #if defined(AURA_ENABLE_NEON)
                    impl.reset(new {name}Neon(ctx, target));
        #endif // AURA_ENABLE_NEON
                    break;
                }}
        """
        self.utils['check_width'] = f"""
                case TargetType::NEON:
                {{
        #if defined(AURA_ENABLE_NEON)
                    if (CheckNeonWidth(/* add parameters here */) != Status::OK)
                    {{
                        impl_target = OpTarget::None();
                    }}
        #endif // AURA_ENABLE_NEON
                    break;
                }}
        """

    def getDeclStr(self):
        class_name = self._params['name'] + "Neon"

        code_str = f"""
        #if defined(AURA_ENABLE_NEON)
        class {class_name} : public {class_name[:-4]}Impl
        {{
        public:
            {self._decl_func_map["construct"].format(class_name)}

            {self._decl_func_map["SetArgs"].format(class_name)}

            {self._decl_func_map["Run"].format(class_name)}
        }};

        /* add code here */

        #endif // AURA_ENABLE_NEON
        """

        return code_str

    def getImplStr(self):
        class_name = self._params['name'] + "Neon"

        code_str_dict = {}
        code_str_dict['host/neon'] = f"""
        {self._impl_func_map["construct"].format(class_name, class_name, class_name[:-4])}

        {self._impl_func_map["SetArgs"].format(class_name, class_name[:-4], class_name[:-4])}

        {self._impl_func_map["Run"].format(class_name)}
        """

        return code_str_dict

class OpCodeImplCL(OpCodeImpl):
    def __init__(self, params):
        super().__init__(params)

        name = self._params['name']
        self.utils['case'][0] = f"""
                case TargetType::OPENCL:
                {{
        #if defined(AURA_ENABLE_OPENCL)
                    impl.reset(new {name}CL(ctx, target));
        #endif // AURA_ENABLE_OPENCL
                    break;
                }}
        """
        self.utils['includes']['private'] = f"""
        #if defined(AURA_ENABLE_OPENCL)
        #  include "aura/runtime/opencl.h"
        #endif // AURA_ENABLE_OPENCL
        """

    def getDeclStr(self):
        class_name = self._params['name'] + "CL"

        code_str = f"""
        #if defined(AURA_ENABLE_OPENCL)
        class {class_name} : public {class_name[:-2]}Impl
        {{
        public:
            {self._decl_func_map["construct"].format(class_name)}

            {self._decl_func_map["SetArgs"]}

            {self._decl_func_map["Initialize"]}

            {self._decl_func_map["DeInitialize"]}

            {self._decl_func_map["Run"]}

            {self._decl_func_map["ToString"]}

        private:
            /* add code here */
            std::string m_profiling_string;
        }};
        #endif // AURA_ENABLE_OPENCL
        """

        return code_str

    def getImplStr(self):
        class_name = self._params['name'] + "CL"

        code_str_dict = {}
        code_str_dict['host/opencl'] = f"""
        {self._impl_func_map["construct"].format(class_name, class_name, class_name[:-2])}

        {self._impl_func_map["SetArgs"].format(class_name, class_name[:-2], class_name[:-2])}

        Status {class_name}::Initialize()
        {{
            if ({class_name[:-2]}Impl::Initialize() != Status::OK)
            {{
                AURA_ADD_ERROR_STRING(m_ctx, "{class_name[:-2]}Impl::Initialize() failed");
                return Status::ERROR;
            }}

            /* add code here */
        }}

        {self._impl_func_map["Run"].format(class_name)}

        {self._impl_func_map["DeInitialize"].format(class_name)}

        {self._impl_func_map["ToString"].format(class_name, class_name[:-2])}
        """

        return code_str_dict

class OpCodeImplHvx(OpCodeImpl):
    def __init__(self, params):
        super().__init__(params)

        name = self._params['name']
        module = self._params['module']
        self.utils['case'][0] = f"""
                case TargetType::HVX:
                {{
        #if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
                    impl.reset(new {name}Hvx(ctx, target));
        #endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
                    break;
                }}
        """
        self.utils['check_width'] = f"""
                case TargetType::HVX:
                {{
        #if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
                    if (CheckHvxWidth(/* add parameters here */) != Status::OK)
                    {{
                        impl_target = OpTarget::None();
                    }}
        #endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
                    break;
                }}
        """
        self.utils['includes']['private'] = f"""
        #if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
        #  include "aura/runtime/hexagon.h"
        #endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
        """
        self.utils['includes']['comm'] = '#include "aura/config.h"'
        self.utils['comm'] = f"""
        #if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
        #  define AURA_OPS_{module.upper()}_PACKAGE_NAME              "aura.ops.{module.lower()}"
        #endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
        """
        self.utils['includes']['impl']['host/hexagon'] = f'''#include "{module.lower()}_comm.hpp"'''
        self.utils['includes']['impl']['hexagon'] = f"""
        #include "aura/ops/{module.lower()}/{name.lower()}.hpp"
        #include "{module.lower()}_comm.hpp"
        """

    def getDeclStr(self):
        module_name = self._params['module']
        class_name = self._params['name'] + "Hvx"

        code_str = f"""
        #if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
        class {class_name} : public {class_name[:-3]}Impl
        {{
        public:
            {self._decl_func_map["construct"].format(class_name)}

            {self._decl_func_map["SetArgs"]}

            {self._decl_func_map["Run"]}

            {self._decl_func_map["ToString"]}

        private:
            std::string m_profiling_string;
        }};

        #  if defined(AURA_BUILD_HEXAGON)
        /* add code here */
        #  endif // AURA_BUILD_HEXAGON

        using {class_name[:-3]}InParam = HexagonRpcParamType</* add parameters here */>;
        #  define AURA_OPS_{module_name.upper()}_{class_name[:-3].upper()}_OP_NAME          "{class_name[:-3]}"

        #endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
        """

        return code_str

    def getImplStr(self):
        name = self._params['name']
        module = self._params['module']
        class_name = self._params['name'] + "Hvx"

        code_str_dict = {}
        code_str_dict['host/hexagon'] = f"""
        {self._impl_func_map["construct"].format(class_name, class_name, class_name[:-3])}

        {self._impl_func_map["SetArgs"].format(class_name, class_name[:-3], class_name[:-3])}

        Status {class_name}::Run()
        {{
            /* add code here */

            Status ret = Status::ERROR;

            HexagonRpcParam rpc_param(m_ctx);
            {name}InParam in_param(m_ctx, rpc_param);
            ret = in_param.Set(/* add parameters here */);
            if (ret != Status::OK)
            {{
                AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
                return Status::ERROR;
            }}

            HexagonProfiling profiling;
            HexagonEngine *engine = m_ctx->GetHexagonEngine();
            ret = engine->Run(AURA_OPS_{module.upper()}_PACKAGE_NAME, AURA_OPS_{module.upper()}_{name.upper()}_OP_NAME, rpc_param, &profiling);

            if (Status::OK == ret && MI_TRUE == m_target.m_data.hvx.profiling)
            {{
                m_profiling_string = " " + HexagonProfilingToString(profiling);
            }}

            AURA_RETURN(m_ctx, ret);
        }}

        {self._impl_func_map["ToString"].format(class_name, class_name[:-3])}
        """

        code_str_dict['hexagon'] = f"""
        {self._impl_func_map["construct"].format(class_name, class_name, class_name[:-3])}

        {self._impl_func_map["SetArgs"].format(class_name, class_name[:-3], class_name[:-3])}

        {self._impl_func_map["Run"].format(class_name)}

        std::string {class_name}::ToString() const
        {{
            return {class_name[:-3]}Impl::ToString();
        }}

        Status {name}Rpc(Context *ctx, HexagonRpcParam &rpc_param)
        {{
            /* add code here */

            {name}InParam in_param(ctx, rpc_param);
            Status ret = in_param.Get(/* add parameters here */);
            if (ret != Status::OK)
            {{
                AURA_ADD_ERROR_STRING(ctx, "Get failed");
                return Status::ERROR;
            }}

            {name} {name.lower()}(ctx, OpTarget::Hvx());

            return OpCall(ctx, {name.lower()}, /* add parameters here */);
        }}

        AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_{module.upper()}_PACKAGE_NAME, AURA_OPS_{module.upper()}_{name.upper()}_OP_NAME, {name}Rpc);
        """

        return code_str_dict

class OpCodeInterface(OpCodeBase):
    def __init__(self, params, format, impls):
        super().__init__(params)
        self.__format = format
        self.__impls = impls

    def getDeclStr(self):
        class_name = self._params['name']

        code_str = f"""
        class AURA_EXPORTS {class_name} : public Op
        {{
        public:
            {class_name}(Context *ctx, const OpTarget &target = OpTarget::Default());

            Status SetArgs(/* add parameters here */);
        }};

        AURA_EXPORTS Status I{class_name}(/* add parameters here */);
        """

        return code_str

    def getImplStr(self):
        class_name = self._params['name']

        min_indent = 9999
        need_check_width = False
        for impl in self.__impls[1:]:
            if impl.utils['check_width'] != '':
                need_check_width = True
            min_indent = min(impl.utils['case'][1], min_indent)
        cases = self.__format('\n'.join([self.__format(impl.utils['case'][0], impl.utils['case'][1]) for impl in self.__impls]), min_indent + 1).strip()
        checks = self.__format('\n'.join([self.__format(impl.utils['check_width']) for impl in self.__impls]), 1).strip()
        switch_code = f"""
        switch (target.m_type)
        {{
            {cases}

            default :
            {{
                break;
            }}
        }}
        """
        check_code = f"""
        switch (m_target.m_type)
        {{
            {checks}

            default:
            {{
                break;
            }}
        }}
        """

        if len(self._params['targets']) == 0:
            switch_code = f"""impl.reset(new {class_name}Impl(ctx, target));"""
        else:
            if min_indent == 0:
                align = 2
            else:
                align = 3
            switch_code = self.__format(switch_code, align).strip()

        if not need_check_width:
            check_code = ''
        else:
            check_code = self.__format(check_code, 2).strip()

        code_str = f"""
        static std::shared_ptr<{class_name}Impl> Create{class_name}Impl(Context *ctx, const OpTarget &target)
        {{
            std::shared_ptr<{class_name}Impl> impl;

            {switch_code}

            return impl;
        }}

        {class_name}::{class_name}(Context *ctx, const OpTarget &target) : Op(ctx, target)
        {{}}

        Status {class_name}::SetArgs(/* add parameters here */)
        {{
            if (MI_NULL == m_ctx)
            {{
                return Status::ERROR;
            }}

            /* add code here */

            OpTarget impl_target = m_target;

            {check_code}

            // set m_impl
            if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
            {{
                m_impl = Create{class_name}Impl(m_ctx, impl_target);
            }}

            // run SetArgs
            {class_name}Impl *{class_name.lower()}_impl = dynamic_cast<{class_name}Impl *>(m_impl.get());
            if (MI_NULL == {class_name.lower()}_impl)
            {{
                AURA_ADD_ERROR_STRING(m_ctx, "{class_name.lower()}_impl is null ptr");
                return Status::ERROR;
            }}

            Status ret = {class_name.lower()}_impl->SetArgs(/* add parameters here */);

            AURA_RETURN(m_ctx, ret);
        }}

        AURA_EXPORTS Status I{class_name}(/* add parameters here */)
        {{
            {class_name} {class_name.lower()}(ctx, target);

            return OpCall(ctx, {class_name.lower()}, /* add parameters here */);
        }}
        """

        return code_str

class OpCodeGenerator(object):
    def __init__(self, args):
        self.__parseArgs(args)
        if self.valid():
            self.__code_impls = [OpCodeImpl(self._params)] + [eval('OpCodeImpl' + t)(self._params) for t in self._params['targets']]

    def valid(self):
        return len(self._params) > 0

    def __parseArgs(self, args):
        self._params = {}
        self._params['module'] = args.module
        self._params['name'] = args.name
        self._params['author'] = args.author
        self._params['path'] = os.path.abspath(args.path)
        if not os.path.exists(args.path):
            os.makedirs(args.path)

        targets = []
        if args.targets != '':
            for i in args.targets:
                if not i.isalpha() and i != ',' and i != ' ':
                    self._params = {}
                    print('error : targets', args.targets, 'is invalid')
                    return
            targets = [t.strip() for t in args.targets.split(',') if t.strip() != '']
            for t in targets:
                try:
                    isinstance(eval('OpCodeImpl' + t), type)
                except (NameError, AttributeError) as e:
                    self._params = {}
                    print('error : target', t, 'is invalid')
                    return

        self._params['targets'] = targets

        time = datetime.date.today()
        self._params['time'] = {}
        self._params['time']['year'] = time.year
        self._params['time']['day'] = time.day
        self._params['time']['month'] = calendar.month_abbr[time.month]

    def __getFileHeader(self, file_name):
        author = self._params['author'].lower()
        day = self._params['time']['day']
        month = self._params['time']['month']
        year = self._params['time']['year']

        code_str = f"""
        /** @brief     : {file_name} header for aura
        *  @file       : {file_name}.hpp
        *  @author     : {author}@xiaomi.com
        *  @version    : 1.0.0
        *  @date       : {month}. {day}, {year}
        *  @Copyright  : Copyright {year} Xiaomi Mobile Software Co., Ltd. All Rights reserved.
        */"""

        return code_str

    def __format(self, code_str, indent = 0):
        lines = code_str.split('\n')
        align_idx = 9999
        space_acc = 0
        lines_strip = []
        for i, l in enumerate(lines):
            if l.isspace() or l == '':
                if space_acc > 0 or len(lines_strip) == 0:
                    continue
                else:
                    lines_strip.append(l)
                    space_acc += 1
            else:
                align_idx = min(len(l) - len(l.lstrip()), align_idx)
                lines_strip.append(l)
                space_acc = 0
        return '\n'.join(['    ' * indent + l[align_idx:] for l in lines_strip])

    def __wirte(self, code_str, path, filename):
        file_dir = os.path.join(self._params['path'], path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(os.path.join(file_dir, filename), 'w') as f:
            f.write(self.__format(code_str).strip())

    def __genExportHeader(self):
        module = self._params['module'].upper()
        name = self._params['name']

        code_core = self.__format(OpCodeInterface(self._params, self.__format, self.__code_impls).getDeclStr(), 2).strip()

        code_str = f"""
        {self.__getFileHeader(name.lower())}

        #ifndef AURA_OPS_{module}_{name.upper()}_HPP__
        #define AURA_OPS_{module}_{name.upper()}_HPP__

        #include "aura/ops/core.h"
        #include "aura/runtime/mat.h"

        namespace aura
        {{

        {code_core}

        }} // namespace aura

        #endif // AURA_OPS_{module}_{name.upper()}_HPP__
        """

        self.__wirte(code_str, '{}/include/aura/ops/{}'.format(module.lower(), module.lower()), '{}.hpp'.format(name.lower()))

    def __genPrivateHeader(self):
        module = self._params['module'].upper()
        name = self._params['name']

        code_core = self.__format('\n'.join([self.__format(impl.getDeclStr()) + "\n" for impl in self.__code_impls]), 2).strip()
        code_incs = self.__format(''.join([self.__format(impl.utils['includes']['private']) for impl in self.__code_impls]), 2).strip()

        code_str = f"""
        {self.__getFileHeader(name.lower() + '_impl')}

        #ifndef AURA_OPS_{module}_{name.upper()}_IMPL_HPP__
        #define AURA_OPS_{module}_{name.upper()}_IMPL_HPP__

        #include "aura/ops/core.h"
        #include "aura/runtime/mat.h"
        {code_incs}

        namespace aura
        {{

        {code_core}

        }} // namespace aura

        #endif // AURA_OPS_{module}_{name.upper()}_IMPL_HPP__
        """

        self.__wirte(code_str, '{}/private'.format(module.lower()), '{}_impl.hpp'.format(name.lower()))

    def __genPrivateCommHeader(self):
        module = self._params['module'].upper()

        code_core = self.__format('\n'.join([self.__format(impl.utils['comm']) + "\n" for impl in self.__code_impls]), 2).strip()
        code_incs = self.__format(''.join([self.__format(impl.utils['includes']['comm']) for impl in self.__code_impls if impl.utils['includes']['comm'] != '']), 2).strip()

        code_str = f"""
        {self.__getFileHeader(module.lower() + '_comm')}

        #ifndef AURA_OPS_{module}_COMM_HPP__
        #define AURA_OPS_{module}_COMM_HPP__

        {code_incs}

        {code_core}

        #endif // AURA_OPS_{module}_COMM_HPP__
        """

        self.__wirte(code_str, '{}/private'.format(module.lower()), '{}_comm.hpp'.format(module.lower()))

    def __genImplCpp(self):
        module = self._params['module']
        name = self._params['name']

        code_core = self.__format('\n'.join([self.__format(OpCodeInterface(self._params, self.__format, self.__code_impls).getImplStr()), self.__format(self.__code_impls[0].getImplStr())]), 2).strip()

        code_str = f"""
        #include "aura/ops/{module.lower()}/{name.lower()}.hpp"
        #include "{name.lower()}_impl.hpp"
        #include "aura/runtime/logger.h"

        namespace aura
        {{

        {code_core}

        }} // namespace aura
        """

        self.__wirte(code_str, '{}/src'.format(module.lower()), '{}.cpp'.format(name.lower()))

    def __genTargetImplCpps(self):
        module = self._params['module']
        name = self._params['name']

        for i in range(1, len(self.__code_impls)):
            impl = self.__code_impls[i]
            code_dict = impl.getImplStr()
            for k in code_dict:
                code_incs = ''
                if impl.utils['includes']['impl'] != {}:
                    code_incs = impl.utils['includes']['impl'][k]
                code_core = self.__format(code_dict[k], 4).strip()

                code_str = f"""
                {self.__format(code_incs, 4).strip()}
                #include "{name.lower()}_impl.hpp"
                #include "aura/runtime/logger.h"

                namespace aura
                {{

                {code_core}

                }} // namespace aura
                """

                self.__wirte(code_str, '{}/src/{}'.format(module.lower(), k), '{}.cpp'.format(name.lower() + '_' + self._params['targets'][i - 1].lower()))

    def gen(self):
        module_dir = os.path.join(self._params['path'], self._params['module'])
        if os.path.exists(module_dir):
            shutil.rmtree(module_dir)
        self.__genExportHeader()
        self.__genPrivateHeader()
        self.__genPrivateCommHeader()
        self.__genImplCpp()
        self.__genTargetImplCpps()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Op Code Generation.')

    parser.add_argument('-m', '--module', type=str, required=True, help='module name, e.g.filter')
    parser.add_argument('-n', '--name', type=str, required=True, help='op name, e.g.Gaussian')
    parser.add_argument('-a', '--author', type=str, required=True, help='author name, e.g.fankai1')
    parser.add_argument('-p', '--path', type=str, required=True, help='code dst path, e.g./home/fankai/Desktop')
    parser.add_argument('-t', '--targets', type=str, default='', help='targets, e.g.None,Neon,CL,Hvx')

    args = parser.parse_args()

    code_gen = OpCodeGenerator(args)
    if code_gen.valid():
        code_gen.gen()
        print("\033[1;32mCode generate done.\033[0m")
    else:
        print("\033[1;31mCode generate failed.\033[0m")