import ctypes

mylib = ctypes.cdll.LoadLibrary(
    "/Users/frewen/03.ProgramStudy/15.CLang/01.WorkSpace/NyxCLang/VisionAbilityLearn/auralib-tools/vast/cpptest.so")

mylib.test()
