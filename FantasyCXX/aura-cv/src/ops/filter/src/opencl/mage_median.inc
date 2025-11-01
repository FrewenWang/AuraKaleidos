#include "cl_helper.inc"

#define V2Tp                                       VTYPE(Tp, 2)
#define V3Tp                                       VTYPE(Tp, 3)
#define V4Tp                                       VTYPE(Tp, 4)
#define V8Tp                                       VTYPE(Tp, 8)
#define V16Tp                                      VTYPE(Tp, 16)

#define SORT2ElemX1(v0, v1)                        {Tp    temp = v0; v0 = min(v0, v1); v1 = max(temp, v1);}
#define SORT2ElemX2(v0, v1)                        {V2Tp  temp = v0; v0 = min(v0, v1); v1 = max(temp, v1);}
#define SORT2ElemX3(v0, v1)                        {V3Tp  temp = v0; v0 = min(v0, v1); v1 = max(temp, v1);}
#define SORT2ElemX4(v0, v1)                        {V4Tp  temp = v0; v0 = min(v0, v1); v1 = max(temp, v1);}
#define SORT2ElemX8(v0, v1)                        {V8Tp  temp = v0; v0 = min(v0, v1); v1 = max(temp, v1);}
#define SORT2ElemX16(v0, v1)                       {V16Tp temp = v0; v0 = min(v0, v1); v1 = max(temp, v1);}

#define SORT3ElemX1(v0, v1, v2)                    {SORT2ElemX1(v0, v1);  SORT2ElemX1(v1, v2);     \
                                                    SORT2ElemX1(v0, v1);}

#define SORT3ElemX2(v0, v1, v2)                    {SORT2ElemX2(v0, v1);  SORT2ElemX2(v1, v2);     \
                                                    SORT2ElemX2(v0, v1);}

#define SORT3ElemX3(v0, v1, v2)                    {SORT2ElemX3(v0, v1);  SORT2ElemX3(v1, v2);     \
                                                    SORT2ElemX3(v0, v1);}

#define SORT3ElemX4(v0, v1, v2)                    {SORT2ElemX4(v0, v1);  SORT2ElemX4(v1, v2);     \
                                                    SORT2ElemX4(v0, v1);}

#define SORT4ElemX1(v0, v1, v2, v3)                {SORT2ElemX1(v0, v1);  SORT2ElemX1(v2, v3);     \
                                                    SORT2ElemX1(v0, v2);  SORT2ElemX1(v1, v3);     \
                                                    SORT2ElemX1(v1, v2);}

#define SORT4ElemX2(v0, v1, v2, v3)                {SORT2ElemX2(v0, v1);  SORT2ElemX2(v2, v3);     \
                                                    SORT2ElemX2(v0, v2);  SORT2ElemX2(v1, v3);     \
                                                    SORT2ElemX2(v1, v2);}

#define SORT4ElemX3(v0, v1, v2, v3)                {SORT2ElemX3(v0, v1);  SORT2ElemX3(v2, v3);     \
                                                    SORT2ElemX3(v0, v2);  SORT2ElemX3(v1, v3);     \
                                                    SORT2ElemX3(v1, v2);}

#define SORT4ElemX4(v0, v1, v2, v3)                {SORT2ElemX4(v0, v1);  SORT2ElemX4(v2, v3);     \
                                                    SORT2ElemX4(v0, v2);  SORT2ElemX4(v1, v3);     \
                                                    SORT2ElemX4(v1, v2);}

#define SORT4ElemX8(v0, v1, v2, v3)                {SORT2ElemX8(v0, v1);  SORT2ElemX8(v2, v3);     \
                                                    SORT2ElemX8(v0, v2);  SORT2ElemX8(v1, v3);     \
                                                    SORT2ElemX8(v1, v2);}

#define SORT4ElemX16(v0, v1, v2, v3)               {SORT2ElemX16(v0, v1); SORT2ElemX16(v2, v3);    \
                                                    SORT2ElemX16(v0, v2); SORT2ElemX16(v1, v3);    \
                                                    SORT2ElemX16(v1, v2);}

#define SORT5ElemX1(v0, v1, v2, v3, v4)            {SORT2ElemX1(v0, v1);  SORT2ElemX1(v2, v3);     \
                                                    SORT2ElemX1(v0, v2);  SORT2ElemX1(v1, v3);     \
                                                    SORT2ElemX1(v0, v4);  SORT2ElemX1(v3, v4);     \
                                                    SORT2ElemX1(v1, v2);  SORT2ElemX1(v1, v3);     \
                                                    SORT2ElemX1(v2, v3);}

#define SORT5ElemX2(v0, v1, v2, v3, v4)            {SORT2ElemX2(v0, v1);  SORT2ElemX2(v2, v3);     \
                                                    SORT2ElemX2(v0, v2);  SORT2ElemX2(v1, v3);     \
                                                    SORT2ElemX2(v0, v4);  SORT2ElemX2(v3, v4);     \
                                                    SORT2ElemX2(v1, v2);  SORT2ElemX2(v1, v3);     \
                                                    SORT2ElemX2(v2, v3);}

#define SORT5ElemX3(v0, v1, v2, v3, v4)            {SORT2ElemX3(v0, v1);  SORT2ElemX3(v2, v3);     \
                                                    SORT2ElemX3(v0, v2);  SORT2ElemX3(v1, v3);     \
                                                    SORT2ElemX3(v0, v4);  SORT2ElemX3(v3, v4);     \
                                                    SORT2ElemX3(v1, v2);  SORT2ElemX3(v1, v3);     \
                                                    SORT2ElemX3(v2, v3);}

#define SORT5ElemX4(v0, v1, v2, v3, v4)            {SORT2ElemX4(v0, v1);  SORT2ElemX4(v2, v3);     \
                                                    SORT2ElemX4(v0, v2);  SORT2ElemX4(v1, v3);     \
                                                    SORT2ElemX4(v0, v4);  SORT2ElemX4(v3, v4);     \
                                                    SORT2ElemX4(v1, v2);  SORT2ElemX4(v1, v3);     \
                                                    SORT2ElemX4(v2, v3); }

#define SORT6ElemX1(v0, v1, v2, v3, v4, v5)        {SORT2ElemX1(v0, v2);  SORT2ElemX1(v1, v4);     \
                                                    SORT2ElemX1(v3, v5);  SORT2ElemX1(v0, v1);     \
                                                    SORT2ElemX1(v0, v3);  SORT2ElemX1(v1, v3);     \
                                                    SORT2ElemX1(v2, v4);  SORT2ElemX1(v2, v5);     \
                                                    SORT2ElemX1(v4, v5);  SORT2ElemX1(v1, v2);     \
                                                    SORT2ElemX1(v3, v4);  SORT2ElemX1(v2, v3); }

#define SORT6ElemX2(v0, v1, v2, v3, v4, v5)        {SORT2ElemX2(v0, v2);  SORT2ElemX2(v1, v4);     \
                                                    SORT2ElemX2(v3, v5);  SORT2ElemX2(v0, v1);     \
                                                    SORT2ElemX2(v0, v3);  SORT2ElemX2(v1, v3);     \
                                                    SORT2ElemX2(v2, v4);  SORT2ElemX2(v2, v5);     \
                                                    SORT2ElemX2(v4, v5);  SORT2ElemX2(v1, v2);     \
                                                    SORT2ElemX2(v3, v4);  SORT2ElemX2(v2, v3);} 

#define SORT6ElemX3(v0, v1, v2, v3, v4, v5)        {SORT2ElemX3(v0, v2);  SORT2ElemX3(v1, v4);     \
                                                    SORT2ElemX3(v3, v5);  SORT2ElemX3(v0, v1);     \
                                                    SORT2ElemX3(v0, v3);  SORT2ElemX3(v1, v3);     \
                                                    SORT2ElemX3(v2, v4);  SORT2ElemX3(v2, v5);     \
                                                    SORT2ElemX3(v4, v5);  SORT2ElemX3(v1, v2);     \
                                                    SORT2ElemX3(v3, v4);  SORT2ElemX3(v2, v3);}
 
#define SORT6ElemX4(v0, v1, v2, v3, v4, v5)        {SORT2ElemX4(v0, v2);  SORT2ElemX4(v1, v4);     \
                                                    SORT2ElemX4(v3, v5);  SORT2ElemX4(v0, v1);     \
                                                    SORT2ElemX4(v0, v3);  SORT2ElemX4(v1, v3);     \
                                                    SORT2ElemX4(v2, v4);  SORT2ElemX4(v2, v5);     \
                                                    SORT2ElemX4(v4, v5);  SORT2ElemX4(v1, v2);     \
                                                    SORT2ElemX4(v3, v4);  SORT2ElemX4(v2, v3);}

#define SORT6ElemX8(v0, v1, v2, v3, v4, v5)        {SORT2ElemX8(v0, v2);  SORT2ElemX8(v1, v4);     \
                                                    SORT2ElemX8(v3, v5);  SORT2ElemX8(v0, v1);     \
                                                    SORT2ElemX8(v0, v3);  SORT2ElemX8(v1, v3);     \
                                                    SORT2ElemX8(v2, v4);  SORT2ElemX8(v2, v5);     \
                                                    SORT2ElemX8(v4, v5);  SORT2ElemX8(v1, v2);     \
                                                    SORT2ElemX8(v3, v4);  SORT2ElemX8(v2, v3);}

#define SORT6ElemX16(v0, v1, v2, v3, v4, v5)       {SORT2ElemX16(v0, v2); SORT2ElemX16(v1, v4);    \
                                                    SORT2ElemX16(v3, v5); SORT2ElemX16(v0, v1);    \
                                                    SORT2ElemX16(v0, v3); SORT2ElemX16(v1, v3);    \
                                                    SORT2ElemX16(v2, v4); SORT2ElemX16(v2, v5);    \
                                                    SORT2ElemX16(v4, v5); SORT2ElemX16(v1, v2);    \
                                                    SORT2ElemX16(v3, v4); SORT2ElemX16(v2, v3);}

#define MERGE_SORT_2_2_X1(v0, v1, v2, v3)          {SORT2ElemX1(v0, v2);  SORT2ElemX1(v1, v3);     \
                                                    SORT2ElemX1(v1, v2)}

#define MERGE_SORT_2_2_X3(v0, v1, v2, v3)          {SORT2ElemX3(v0, v2);  SORT2ElemX3(v1, v3);     \
                                                    SORT2ElemX3(v1, v2)}

#define MERGE_SORT_2_2_X4(v0, v1, v2, v3)          {SORT2ElemX4(v0, v2);  SORT2ElemX4(v1, v3);     \
                                                    SORT2ElemX4(v1, v2)}

#define MERGE_SORT_3_3_X1(v0, v1, v2, v3, v4, v5)  {SORT2ElemX1(v0, v3);  SORT2ElemX1(v2, v5);     \
                                                    SORT4ElemX1(v1, v2, v3, v4)}

#define MERGE_SORT_3_3_X2(v0, v1, v2, v3, v4, v5)  {SORT2ElemX2(v0, v3);  SORT2ElemX2(v2, v5);     \
                                                    SORT4ElemX2(v1, v2, v3, v4)}

#define MERGE_SORT_3_3_X3(v0, v1, v2, v3, v4, v5)  {SORT2ElemX3(v0, v3);  SORT2ElemX3(v2, v5);     \
                                                    SORT4ElemX3(v1, v2, v3, v4)}

#define MERGE_SORT_3_3_X4(v0, v1, v2, v3, v4, v5)  {SORT2ElemX4(v0, v3);  SORT2ElemX4(v2, v5);     \
                                                    SORT4ElemX4(v1, v2, v3, v4)}

// ===================
//     Median 5x5
// ===================
// mid10 will be stored in 2/3/5/6/7/8/9/10/12/13 
#define  GET_MID10_OF_16(N, elem0, elem1, elem2, elem3, elem4, elem5, elem6, elem7, elem8, elem9, elem10, elem11, elem12, elem13, elem14, elem15) \
{                                                                                     \
    /* sort hori */                                                                   \
    SORT4ElemX##N(elem0,  elem1,  elem2,  elem3);                                     \
    SORT4ElemX##N(elem4,  elem5,  elem6,  elem7);                                     \
    SORT4ElemX##N(elem8,  elem9,  elem10, elem11);                                    \
    SORT4ElemX##N(elem12, elem13, elem14, elem15);                                    \
    /* sort diagonal */                                                               \
    SORT2ElemX##N(elem4,  elem1);                                                     \
    SORT3ElemX##N(elem8,  elem5,  elem2);                                             \
    SORT4ElemX##N(elem12, elem9,  elem6, elem3);                                      \
    SORT3ElemX##N(elem13, elem10, elem7);                                             \
    SORT2ElemX##N(elem14, elem11);                                                    \
    /* find a min and a max num */                                                    \
    SORT2ElemX##N(elem1, elem8);                                                      \
    SORT2ElemX##N(elem7, elem14);                                                     \
    SORT2ElemX##N(elem8, elem12);                                                     \
    SORT2ElemX##N(elem3, elem7);                                                      \
    /* sort 10 nums */                                                                \
    SORT2ElemX##N(elem3,  elem10);                                                    \
    SORT2ElemX##N(elem3,  elem13);                                                    \
    SORT2ElemX##N(elem6,  elem3);                                                     \
    SORT2ElemX##N(elem5,  elem12);                                                    \
    SORT2ElemX##N(elem2,  elem12);                                                    \
    SORT2ElemX##N(elem12, elem9);                                                     \
    SORT2ElemX##N(elem9,  elem6);                                                     \
}

// mid6 will be stored in 2/3/6/9/12/13
#define GET_MID6_OF_14(N, elem2, elem3, elem5, elem6, elem7, elem8, elem9, elem10, elem12, elem13, border0, border1, border2, border3) \
{                                                                                     \
    MERGE_SORT_2_2_X##N(border0, border1, elem8,   elem5);                            \
    MERGE_SORT_2_2_X##N(elem10,  elem7,   border2, border3);                          \
    SORT4ElemX##N(elem8, elem5, elem10, elem7);                                       \
                                                                                      \
    MERGE_SORT_2_2_X##N(elem8, elem5,  elem2,  elem12);                               \
    MERGE_SORT_2_2_X##N(elem3, elem13, elem10, elem7);                                \
                                                                                      \
    SORT2ElemX##N(elem2,  elem9);                                                     \
    SORT2ElemX##N(elem2,  elem3);                                                     \
    SORT2ElemX##N(elem9,  elem3);                                                     \
    SORT2ElemX##N(elem12, elem6);                                                     \
    SORT2ElemX##N(elem12, elem13);                                                    \
    SORT2ElemX##N(elem6,  elem13);                                                    \
    SORT2ElemX##N(elem12, elem3);                                                     \
}

// mid2 will be stored in 3/12
#define GET_MID2_OF_10(N, elem2, elem3, elem6, elem9, elem12, elem13, border0, border1, border2, border3) \
{                                                                                     \
    MERGE_SORT_2_2_X##N(border0, border1, elem2,   elem9);                            \
    MERGE_SORT_2_2_X##N(elem6,   elem13,  border2, border3);                          \
    MERGE_SORT_2_2_X##N(elem2,   elem9,   elem12,  elem3);                            \
    MERGE_SORT_2_2_X##N(elem12,  elem3,   elem6,   elem13);                           \
    MERGE_SORT_2_2_X##N(elem2,   elem9,   elem12,  elem3);                            \
}

#define GET_MEDIAN5x5(N, mid10, left4, right4, edge4, left_corner, right_corner, dst_left, dst_right)  \
{                                                                                     \
    /* get mid 6 of 14 */                                                             \
    GET_MID6_OF_14(N, mid10[0], mid10[1], mid10[2], mid10[3], mid10[4],               \
                      mid10[5], mid10[6], mid10[7], mid10[8], mid10[9],               \
                      edge4[0], edge4[1], edge4[2], edge4[3]);                        \
    VTYPE(Tp, N) mid6_left[6] = { mid10[0], mid10[1], mid10[3],                       \
                                  mid10[6], mid10[8], mid10[9] };                     \
    /* left part */                                                                   \
    /* get mid 2 of 10  */                                                            \
    GET_MID2_OF_10(N, mid6_left[0], mid6_left[1], mid6_left[2],                       \
                      mid6_left[3], mid6_left[4], mid6_left[5],                       \
                      left4[0], left4[1], left4[2], left4[3]);                        \
    /* get median out of 3 */                                                         \
    SORT2ElemX##N(left_corner,  mid6_left[4]);                                        \
    SORT2ElemX##N(mid6_left[4], mid6_left[1]);                                        \
    dst_left = mid6_left[4];                                                          \
    /* right part */                                                                  \
    /* get mid 2 of 10  */                                                            \
    GET_MID2_OF_10(N, mid10[0], mid10[1], mid10[3],                                   \
                      mid10[6], mid10[8], mid10[9],                                   \
                      right4[0], right4[1], right4[2], right4[3]);                    \
    /* get median out of 3 */                                                         \
    SORT2ElemX##N(right_corner,  mid10[8]);                                           \
    SORT2ElemX##N(mid10[8], mid10[1]);                                                \
    dst_right  = mid10[8];                                                            \
}

// ===================
//     Median 7x7
// ===================
// mid14 will be stored in 4/9/14/15/19/20/21/25/26/27/30/31/32/33 
#define GET_MID14_OF_36(N,  elem0,  elem1,  elem2,  elem3,  elem4,  elem5,  elem6,  elem7,  elem8,  elem9,  elem10, elem11, \
                            elem12, elem13, elem14, elem15, elem16, elem17, elem18, elem19, elem20, elem21, elem22, elem23, \
                            elem24, elem25, elem26, elem27, elem28, elem29, elem30, elem31, elem32, elem33, elem34, elem35) \
{                                                                                     \
    /* sort hori */                                                                   \
    SORT6ElemX##N(elem0,  elem1,  elem2,  elem3,  elem4,  elem5);                     \
    SORT6ElemX##N(elem6,  elem7,  elem8,  elem9,  elem10, elem11);                    \
    SORT6ElemX##N(elem12, elem13, elem14, elem15, elem16, elem17);                    \
    SORT6ElemX##N(elem18, elem19, elem20, elem21, elem22, elem23);                    \
    SORT6ElemX##N(elem24, elem25, elem26, elem27, elem28, elem29);                    \
    SORT6ElemX##N(elem30, elem31, elem32, elem33, elem34, elem35);                    \
    /* sort diagonal */                                                               \
    SORT3ElemX##N(elem12, elem7,  elem2);                                             \
    SORT4ElemX##N(elem18, elem13, elem8 , elem3);                                     \
    SORT5ElemX##N(elem24, elem19, elem14, elem9,  elem4);                             \
    SORT6ElemX##N(elem30, elem25, elem20, elem15, elem10, elem5);                     \
    SORT5ElemX##N(elem31, elem26, elem21, elem16, elem11);                            \
    SORT4ElemX##N(elem32, elem27, elem22, elem17);                                    \
    SORT3ElemX##N(elem33, elem28, elem23);                                            \
    /* find a min and a max num */                                                    \
    SORT2ElemX##N(elem3, elem14);                                                     \
    SORT2ElemX##N(elem4, elem15);                                                     \
    SORT2ElemX##N(elem5, elem16);                                                     \
    SORT2ElemX##N(elem19, elem30);                                                    \
    SORT2ElemX##N(elem20, elem31);                                                    \
    SORT2ElemX##N(elem21, elem32);                                                    \
    /* merge sort, find the three largest and smallest points and remove*/            \
    MERGE_SORT_3_3_X##N(elem2,  elem8,  elem3,  elem19, elem30, elem25);              \
    MERGE_SORT_3_3_X##N(elem32, elem27, elem33, elem10, elem5,  elem16);              \
    /* creat 4x4 box */                                                               \
    SORT2ElemX##N(elem14, elem20);                                                    \
    SORT4ElemX##N(elem19, elem32, elem9,  elem31);                                    \
    SORT4ElemX##N(elem30, elem27, elem4,  elem26);                                    \
    SORT4ElemX##N(elem25, elem33, elem15, elem21);                                    \
    /* sort diagonal */                                                               \
    SORT3ElemX##N(elem30, elem32, elem14);                                            \
    SORT4ElemX##N(elem25, elem27, elem9,  elem20);                                    \
    SORT3ElemX##N(elem33, elem4,  elem31);                                            \
    SORT2ElemX##N(elem15, elem26);                                                    \
    /* sequencing 14 numbers */                                                       \
    SORT2ElemX##N(elem31, elem15);                                                    \
    SORT2ElemX##N(elem20, elem31);                                                    \
    SORT2ElemX##N(elem20, elem4);                                                     \
    SORT2ElemX##N(elem9,  elem33);                                                    \
    SORT2ElemX##N(elem20, elem33);                                                    \
    SORT2ElemX##N(elem9,  elem20);                                                    \
                                                                                      \
    SORT2ElemX##N(elem19, elem30);                                                    \
    SORT2ElemX##N(elem30, elem25);                                                    \
    SORT2ElemX##N(elem32, elem25);                                                    \
    SORT2ElemX##N(elem14, elem27);                                                    \
    SORT2ElemX##N(elem14, elem25);                                                    \
    SORT2ElemX##N(elem25, elem27);                                                    \
                                                                                      \
    SORT2ElemX##N(elem27, elem9);                                                     \
}

// 14: 4/9/14/15/19/20/21/25/26/27/30/31/32/33 
// mid8 will be stored in 4/9/14/20/25/27/31/33
#define GET_MID8_OF_20(N, elem4, elem9, elem14, elem15, elem19, elem20, elem21, elem25, elem26, elem27, elem30, elem31, elem32, elem33, \
                          border0, border1, border2, border3, border4, border5)       \
{                                                                                     \
    MERGE_SORT_3_3_X##N(border0, border1, border2, elem19,  elem30,  elem32);         \
    MERGE_SORT_3_3_X##N(elem15,  elem26,  elem21,  border3, border4, border5);        \
    MERGE_SORT_3_3_X##N(elem19,  elem30,  elem32,  elem15,  elem26,  elem21);         \
    MERGE_SORT_3_3_X##N(elem19,  elem30,  elem32,  elem14,  elem25,  elem27);         \
    MERGE_SORT_3_3_X##N(elem33,  elem4,   elem31,  elem15,  elem26,  elem21);         \
                                                                                      \
    SORT2ElemX##N(elem14, elem33);                                                    \
    SORT3ElemX##N(elem9,  elem25, elem4);                                             \
    SORT3ElemX##N(elem20, elem27, elem31);                                            \
                                                                                      \
    SORT2ElemX##N(elem9,  elem14);                                                    \
    SORT3ElemX##N(elem20, elem25, elem33);                                            \
    SORT2ElemX##N(elem27, elem4);                                                     \
    SORT2ElemX##N(elem14, elem20);                                                    \
    SORT2ElemX##N(elem33, elem27);                                                    \
}

#define GET_MID2_OF_14(N, elem4, elem9, elem14, elem20, elem25, elem27, elem31, elem33, border0, border1, border2, border3, border4, border5) \
{                                                                                     \
    MERGE_SORT_3_3_X##N(border0, border1, border2, elem9,  elem14,  elem20);          \
    MERGE_SORT_3_3_X##N(elem27,  elem4,   elem31,  border3, border4, border5);        \
                                                                                      \
    SORT2ElemX##N(elem9,  elem27);                                                    \
    SORT3ElemX##N(elem25, elem14, elem4);                                             \
    SORT3ElemX##N(elem33, elem20, elem31);                                            \
                                                                                      \
    SORT2ElemX##N(elem25, elem9);                                                     \
    SORT3ElemX##N(elem33, elem14, elem27);                                            \
    SORT2ElemX##N(elem20, elem4);                                                     \
    SORT2ElemX##N(elem9,  elem33);                                                    \
    SORT2ElemX##N(elem27, elem20);                                                    \
}

#define GET_MEDIAN7x7(N, mid14, left6, right6, edge6, left_corner, right_corner, dst_left, dst_right)  \
{                                                                                     \
    /* get mid 8 of 20 */                                                             \
    GET_MID8_OF_20(N, mid14[0],  mid14[1],  mid14[2],  mid14[3],  mid14[4],           \
                      mid14[5],  mid14[6],  mid14[7],  mid14[8],  mid14[9],           \
                      mid14[10], mid14[11], mid14[12], mid14[13], edge6[0],           \
                      edge6[1],  edge6[2],  edge6[3],  edge6[4],  edge6[5]);          \
    VTYPE(Tp, N) mid8_left[8] = { mid14[0], mid14[1], mid14[2],  mid14[5],            \
                                  mid14[7], mid14[9], mid14[11], mid14[13] };         \
    /* left part */                                                                   \
    /* get mid 2 of 14  */                                                            \
    GET_MID2_OF_14(N, mid8_left[0], mid8_left[1], mid8_left[2], mid8_left[3],         \
                      mid8_left[4], mid8_left[5], mid8_left[6], mid8_left[7],         \
                      left6[0], left6[1], left6[2], left6[3], left6[4],               \
                      left6[5]);                                                      \
    /* get median out of 3 */                                                         \
    SORT2ElemX##N(left_corner,  mid8_left[2]);                                        \
    SORT2ElemX##N(mid8_left[2], mid8_left[5]);                                        \
    dst_left = mid8_left[2];                                                          \
    /* right part */                                                                  \
    /* get mid 2 of 14  */                                                            \
    GET_MID2_OF_14(N, mid14[0], mid14[1], mid14[2],  mid14[5],                        \
                      mid14[7], mid14[9], mid14[11], mid14[13],                       \
                      right6[0], right6[1], right6[2], right6[3],                     \
                      right6[4], right6[5]);                                          \
    /* get median out of 3 */                                                         \
    SORT2ElemX##N(right_corner,  mid14[2]);                                           \
    SORT2ElemX##N(mid14[2], mid14[9]);                                                \
    dst_right  = mid14[2];                                                            \
}