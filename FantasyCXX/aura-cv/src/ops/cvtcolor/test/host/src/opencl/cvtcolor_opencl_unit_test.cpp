#include "host/include/cvtcolor_unit_test.hpp"

static CvtColorParam::TupleTable g_cvtcolor_bgr2gray_table_cl
{
    // elem_type
    {
        ElemType::U8,
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 3), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 1), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 3), Sizes(480, 2560 * 3)}
            },
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)}
            }
        },
        {
            {
                {Sizes3(238, 318, 3), Sizes()}
            },
            {
                {Sizes3(238, 318, 1), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::BGR2GRAY,
        CvtColorType::RGB2GRAY,
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_bgra2gray_table_cl
{
    // elem_type
    {
        ElemType::U8,
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 4), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 1), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 4), Sizes(480, 2560 * 4)}
            },
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)}
            }
        },
        {
            {
                {Sizes3(238, 318, 4), Sizes()}
            },
            {
                {Sizes3(238, 318, 1), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::BGRA2GRAY,
        CvtColorType::RGBA2GRAY,
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_yuv2rgb_nv_table_cl
{
    // elem_type
    {
        ElemType::U8
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 1), Sizes()},
                {Sizes3(512, 1024, 2), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 3), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
                {Sizes3(239, 319, 2), Sizes(240, 1280 * 2)}
            },
            {
                {Sizes3(478, 638, 3), Sizes(480, 2560 * 3)}
            }
        },
        {
            {
                {Sizes3(238, 318, 1), Sizes()},
                {Sizes3(119, 159, 2), Sizes()}
            },
            {
                {Sizes3(238, 318, 3), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::YUV2RGB_NV12,
        CvtColorType::YUV2RGB_NV21,
        CvtColorType::YUV2RGB_NV12_601,
        CvtColorType::YUV2RGB_NV21_601
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_yuv2rgb_y420_table_cl
{
    // elem_type
    {
        ElemType::U8
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 1), Sizes()},
                {Sizes3(512, 1024, 1), Sizes()},
                {Sizes3(512, 1024, 1), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 3), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
                {Sizes3(239, 319, 1), Sizes(240, 1280 * 1)},
                {Sizes3(239, 319, 1), Sizes(240, 1280 * 1)}
            },
            {
                {Sizes3(478, 638, 3), Sizes(480, 2560 * 3)}
            }
        },
        {
            {
                {Sizes3(238, 318, 1), Sizes()},
                {Sizes3(119, 159, 1), Sizes()},
                {Sizes3(119, 159, 1), Sizes()}
            },
            {
                {Sizes3(238, 318, 3), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::YUV2RGB_YU12,
        CvtColorType::YUV2RGB_YV12,
        CvtColorType::YUV2RGB_YU12_601,
        CvtColorType::YUV2RGB_YV12_601
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_yuv2rgb_y422_table_cl
{
    // elem_type
    {
        ElemType::U8
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 2), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 3), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 2), Sizes(480, 2560 * 2)}
            },
            {
                {Sizes3(478, 638, 3), Sizes(480, 2560 * 3)}
            }
        },
        {
            {
                {Sizes3(238, 318, 2), Sizes()}
            },
            {
                {Sizes3(238, 318, 3), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::YUV2RGB_Y422,
        CvtColorType::YUV2RGB_YUYV,
        CvtColorType::YUV2RGB_YVYU,
        CvtColorType::YUV2RGB_Y422_601,
        CvtColorType::YUV2RGB_YUYV_601,
        CvtColorType::YUV2RGB_YVYU_601
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_yuv2rgb_y444_table_cl
{
    // elem_type
    {
        ElemType::U8
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 1), Sizes()},
                {Sizes3(1024, 2048, 1), Sizes()},
                {Sizes3(1024, 2048, 1), Sizes()},
            },
            {
                {Sizes3(1024, 2048, 3), Sizes()},
            }
        },
        {
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
            },
            {
                {Sizes3(478, 638, 3), Sizes(480, 2560 * 3)},
            }
        },
        {
            {
                {Sizes3(238, 318, 1), Sizes()},
                {Sizes3(238, 318, 1), Sizes()},
                {Sizes3(238, 318, 1), Sizes()}
            },
            {
                {Sizes3(238, 318, 3), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::YUV2RGB_Y444,
        CvtColorType::YUV2RGB_Y444_601
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_rgb2yuv_y420_table_cl
{
    // elem_type
    {
        ElemType::U8
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 3), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 1), Sizes()},
                {Sizes3(512, 1024, 1), Sizes()},
                {Sizes3(512, 1024, 1), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 3), Sizes(960, 2560 * 3)}
            },
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
                {Sizes3(239, 319, 1), Sizes(240, 1280 * 1)},
                {Sizes3(239, 319, 1), Sizes(240, 1280 * 1)}
            }
        },
        {
            {
                {Sizes3(238, 318, 3), Sizes()}
            },
            {
                {Sizes3(238, 318, 1), Sizes()},
                {Sizes3(119, 159, 1), Sizes()},
                {Sizes3(119, 159, 1), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::RGB2YUV_YU12,
        CvtColorType::RGB2YUV_YV12,
        CvtColorType::RGB2YUV_YU12_601,
        CvtColorType::RGB2YUV_YV12_601
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_rgb2yuv_nv_table_cl
{
    // elem_type
    {
        ElemType::U8
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 3), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 1), Sizes()},
                {Sizes3(512, 1024, 2), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 3), Sizes(480, 2560 * 3)}
            },
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
                {Sizes3(239, 319, 2), Sizes(240, 1280 * 2)}
            }
        },
        {
            {
                {Sizes3(238, 318, 3), Sizes()}
            },
            {
                {Sizes3(238, 318, 1), Sizes()},
                {Sizes3(119, 159, 2), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::RGB2YUV_NV12,
        CvtColorType::RGB2YUV_NV21,
        CvtColorType::RGB2YUV_NV12_601,
        CvtColorType::RGB2YUV_NV21_601
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_rgb2yuv_y444_table_cl
{
    // elem_type
    {
        ElemType::U8
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 3), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 1), Sizes()},
                {Sizes3(1024, 2048, 1), Sizes()},
                {Sizes3(1024, 2048, 1), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 3), Sizes(480, 2560 * 3)}
            },
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)}
            }
        },
        {
            {
                {Sizes3(238, 318, 3), Sizes()}
            },
            {
                {Sizes3(238, 318, 1), Sizes()},
                {Sizes3(238, 318, 1), Sizes()},
                {Sizes3(238, 318, 1), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::RGB2YUV_Y444,
        CvtColorType::RGB2YUV_Y444_601
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_rgb2yuv_nv_p010_table_cl
{
    // elem_type
    {
        ElemType::U16
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 3), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 1), Sizes()},
                {Sizes3(512, 1024, 2), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 3), Sizes(480, 2560 * 3)}
            },
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)},
                {Sizes3(239, 319, 2), Sizes(240, 1280 * 2)}
            }
        },
        {
            {
                {Sizes3(238, 318, 3), Sizes()}
            },
            {
                {Sizes3(238, 318, 1), Sizes()},
                {Sizes3(119, 159, 2), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::RGB2YUV_NV12_P010,
        CvtColorType::RGB2YUV_NV21_P010,
    },

    // target
    {
        OpTarget::Opencl()
    },
};

static CvtColorParam::TupleTable g_cvtcolor_bayer2bgr_table_cl
{
    // elem_type
    {
        ElemType::U8,
        ElemType::U16,
    },

    // mat size pair
    {
        {
            {
                {Sizes3(1024, 2048, 1), Sizes()}
            },
            {
                {Sizes3(1024, 2048, 3), Sizes()}
            }
        },
        {
            {
                {Sizes3(478, 638, 1), Sizes(480, 2560 * 1)}
            },
            {
                {Sizes3(478, 638, 3), Sizes(480, 2560 * 3)}
            }
        },
        {
            {
                {Sizes3(238, 318, 1), Sizes()}
            },
            {
                {Sizes3(238, 318, 3), Sizes()}
            }
        }
    },

    // cvtcolor type
    {
        CvtColorType::BAYERBG2BGR,
        CvtColorType::BAYERGB2BGR,
        CvtColorType::BAYERRG2BGR,
        CvtColorType::BAYERGR2BGR
    },

    // target
    {
        OpTarget::Opencl()
    },
};

NEW_TESTCASE(cvtcolor, cvtcolor, opencl)
{
    // BGR <-> GRAY
    {
        CvtColorTest test_bgr2gray(UnitTest::GetInstance()->GetContext(), g_cvtcolor_bgr2gray_table_cl);
        test_bgr2gray.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    {
        CvtColorTest test_bgra2gray(UnitTest::GetInstance()->GetContext(), g_cvtcolor_bgra2gray_table_cl);
        test_bgra2gray.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    // YUV -> RGB
    {
        CvtColorTest test_yuv2rgb_nv(UnitTest::GetInstance()->GetContext(), g_cvtcolor_yuv2rgb_nv_table_cl);
        test_yuv2rgb_nv.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    {
        CvtColorTest test_yuv2rgb_y420(UnitTest::GetInstance()->GetContext(), g_cvtcolor_yuv2rgb_y420_table_cl);
        test_yuv2rgb_y420.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    {
        CvtColorTest test_yuv2rgb_y422(UnitTest::GetInstance()->GetContext(), g_cvtcolor_yuv2rgb_y422_table_cl);
        test_yuv2rgb_y422.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    {
        CvtColorTest test_yuv2rgb_y444(UnitTest::GetInstance()->GetContext(), g_cvtcolor_yuv2rgb_y444_table_cl);
        test_yuv2rgb_y444.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    // RGB -> YUV
    {
        CvtColorTest test_rgb2yuv_y420(UnitTest::GetInstance()->GetContext(), g_cvtcolor_rgb2yuv_y420_table_cl);
        test_rgb2yuv_y420.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    {
        CvtColorTest test_rgb2yuv_nv(UnitTest::GetInstance()->GetContext(), g_cvtcolor_rgb2yuv_nv_table_cl);
        test_rgb2yuv_nv.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    {
        CvtColorTest test_rgb2yuv_y444(UnitTest::GetInstance()->GetContext(), g_cvtcolor_rgb2yuv_y444_table_cl);
        test_rgb2yuv_y444.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    {
        CvtColorTest test_rgb2yuv_nv_p010(UnitTest::GetInstance()->GetContext(), g_cvtcolor_rgb2yuv_nv_p010_table_cl);
        test_rgb2yuv_nv_p010.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }

    // BAYER -> BGR
    {
        CvtColorTest test_bayer2bgr(UnitTest::GetInstance()->GetContext(), g_cvtcolor_bayer2bgr_table_cl);
        test_bayer2bgr.RunTest(this, UnitTest::GetInstance()->GetStressCount());
    }
}