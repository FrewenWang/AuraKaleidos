#include "sample_matrix.hpp"

static std::string help_info = R"(
Usage:
    Usage: sample_matrix [Operators] [TargetType]

Example usage:
    Usage: ./sample_matrix rotate none 

Operators:
    This module contains commone used matrix operators, for example:
    arithm_scalar              Arithm with the scalar value by each element of the source matrix and stores the result.
    arithmetic                 Support iaura add, subtract, multiply and divide operations.
    convert                    Convert iaura datatype to another datatype.
    dct                        Including the DCT and IDCT operation.
    dft                        Performs a forward or inverse Discrete Fourier transform of a 1D or 2D array.
    flip                       Flips a 2D array around vertical, horizontal, or both axes.
    gemm                       Performs generalized matrix multiplication.
    grid_dft                   Performs a grid-based Discrete Fourier Transform (DFT) on the source matrix.
    integral                   Computes the integral and squared integral iauras of the source matrix.
    make_border                Adds borders to the src matrix and stores the result in the dst matrix.
    mean_std_dev               Computes the mean and standard deviation of the src matrix.
    merge                      Merges several arrays to make a single multi-channel array.
    min_max_loc                Finds the minimum and maximum values and their positions in the source matrix.
    minmax                     Performs element-wise maximum or minimum operation between two matrices.
    mul_spectrums              Multiplies two source matrices (spectrums).
    norm                       Computes a specific type of norm on the source matrix.
    normalize                  Normalizes the src matrix using the provided parameters (alpha, beta, type).
    psnr                       Compute iaura quality metric.
    rotate                     Rotates a 2D array in multiples of 90 degrees.
    split                      Splits a multi-channel array into separate single-channel arrays.
    sum_mean                   Compute the mean and sum of all elements in each channel of the src matrix.
    transpose                  Use this function to transposes a matrix.

TargetType:
    These matrix operators run on different hardware(CPU, GPU and DSP),
    you can choose the target type to run the operator, for example:

    none                 Run on CPU. (Supported by all devices, Android/Linux/Windows)
    neon                 Run on Android  CPU with NEON   support.
    opencl               Run on Android  GPU with OpenCL support.
    hvx                  Run on Qualcomm DSP with HVX    support.
)";

const static std::map<std::string, SampleOpsFunc> g_func_map = {
    {"arithm_scalar", ArithmScalarSampleTest},
    {"arithmetic",    ArithmeticSampleTest  },
    {"convert",       ConvertToSampleTest   },
    {"dct",           DctIDctSampleTest     },
    {"dft",           DftIDftSampleTest     },
    {"flip",          FlipSampleTest        },
    {"gemm",          GemmSampleTest        },
    {"grid_dft",      GridDftIDftSampleTest },
    {"integral",      IntegralSampleTest    },
    {"make_border",   MakeBorderSampleTest  },
    {"mean_std_dev",  MeanStdDevSampleTest  },
    {"merge",         MergeSampleTest       },
    {"min_max_loc",   MinMaxLocSampleTest   },
    {"minmax",        MinMaxSampleTest      },
    {"mul_spectrums", MulSpectrumsSampleTest},
    {"norm",          NormSampleTest        },
    {"normalize",     NormalizeSampleTest   },
    {"psnr",          PsnrSampleTest        },
    {"rotate",        RotateSampleTest      },
    {"split",         SplitSampleTest       },
    {"sum_mean",      SumMeanSampleTest     },
    {"transpose",     TransposeSampleTest   }
};

MI_S32 main(MI_S32 argc, MI_CHAR *argv[])
{
    SampleOpsFunc sample_func;
    aura::TargetType type;

    // create context for sample
    std::shared_ptr<aura::Context> ctx = CreateContext();
    if (nullptr == ctx)
    {
        return -1;
    }

    // parse inputs
    if (InputParser(argc, argv, help_info, g_func_map, sample_func, type) != aura::Status::OK)
    {
        AURA_LOGE(ctx.get(), SAMPLE_TAG, "InputParser failed\n");
        return -1;
    }

    // run matrix sample
    aura::Status ret = sample_func(ctx.get(), type);
    if (aura::Status::OK == ret)
    {
        AURA_LOGD(ctx, SAMPLE_TAG, "=================== sample_matrix execute succeeded ===================");
        return 0;
    }
    else
    {
        AURA_LOGE(ctx, SAMPLE_TAG, "=================== sample_matrix execute failed ===================");
        return -1;
    }
}