/************************************************************************************

Filename  : hist_u8_asm.s
Content   :
Created   : Mar. 15, 2021
Author    : Rong Hu (hurong3@xiaomi.com)
Copyright : Copyright 2021 Xiaomi Mobile Software Co., Ltd. All Rights reserved.

*************************************************************************************/

#define VLEN 128
#define LOG2VLEN 7
/*[*****************************************************************************]*/
/*[  FUNCTION   : void HistogramPernRow()                                       ]*/
/*[*****************************************************************************]*/
/*[  DESCRIPTION: calcluate histogram of a iaura block                          ]*/
/*[=============================================================================]*/
/*[  INPUTS     : R0 : unsigned char *inp  -- pointer to input iaura            ]*/
/*[               R1 : int stride_i        -- stride of iaura input             ]*/
/*[               R2 : int width           -- width of iaura block              ]*/
/*[               R3 : int height          -- height of iaura block             ]*/
/*[               R4 : int *hist           -- pointer to histogram array        ]*/
/*[=============================================================================]*/
/*[                                                                             ]*/
/*[*****************************************************************************]*/
/* ============================================================ */
    .text
    .p2align 2
    .p2align 4,,15
    .globl  HistogramPernRow
    .type   HistogramPernRow, @function
HistogramPernRow:
    {
        R2 = LSR(R2,#LOG2VLEN)                        // size/VLEN
        R5 = AND(R2,#VLEN-1)                          // size%VLEN
        V1 = #0                                       //
        V0 = #0                                       //
    }
    //[----------------------------------------------]
    //[ Step 1: Clean the whole vector Register file ]
    //[ to get ready for vhist                       ]
    //[----------------------------------------------]
    {
        V3:2 = V1:0                                   //
        V5:4 = V1:0                                   //
        P0 = CMP.GT(R2, #0)                           // P0 = (width/VLEN > 0)
        P1 = CMP.EQ(R5, #0)                           // P1 = (width%VLEN==0)
    }
    {
        Q0 = VSETQ(R5)                                //
        V7:6 = V1:0                                   //
    }
    {
        V9:8   = V1:0                                 //
        V11:10 = V1:0                                 //
    }
    {
        V13:12 = V1:0                                 //
        V15:14 = V1:0                                 //
    }
    {
        V17:16 = V1:0                                 //
        V19:18 = V1:0                                 //
    }
    {
        V21:20 = V1:0                                 //
        V23:22 = V1:0                                 //
    }
    {
        V25:24 = V1:0                                 //
        V27:26 = V1:0                                 //
    }
    {
        V29:28 = V1:0                                 //
        V31:30 = V1:0                                 //
        R10 = ADD(R0,R1)                              // R10 = &src[2*stride]
        LOOP1(.histogram_verticalLoop,R3)             // set loop1 with lc1=height
    }
    //[----------------------------------------------]
    //[ Step 2: vhist                                ]
    //[----------------------------------------------]

    .falign
.histogram_verticalLoop:
    {
        IF !P0 JUMP .histogramLPEND                   //
        LOOP0(.histogramLoop, R2)                     //
        NOP; NOP                                      //
    }
    .falign
.histogramLoop:
    {
        V0.tmp = VMEM(R0++#1)                         //
        VHIST                                         //
        NOP; NOP                                      //
    }:endloop0
.histogramLPEND:
    {
        IF (P1) JUMP .skipRemainder                   // if (width%VLEN==0) done with current row
    }
    {
        V0.tmp = VMEM(R0+#0)                          // otherwise process the rest pixels
        VHIST(Q0)                                     //
    }
    .falign
.skipRemainder:
    {
        R0 = R10                                      // set R0  = &src[(i+1)*stride]
        R10 = ADD(R10,R1)                             // update R10 = &src[(i+2)*stride]
    }:endloop1

    //[----------------------------------------------]
    //[ Step 3:                                      ]
    //[  - Sum up the data in the register file      ]
    //[  - accumulate to hist[]                      ]
    //[----------------------------------------------]
    {
        V0.h = VSHUFF(V0.h)                           //
        R10 = ##0x00010001                            //
    }
    {
        V1.h = VSHUFF(V1.h)                           //
    }
    {
        V2.h = VSHUFF(V2.h)                           //
        V0.w = VDMPY(V0.h, R10.h):sat                 //
    }
    {
        V3.h = VSHUFF(V3.h)                           //
        V1.w = VDMPY(V1.h, R10.h):sat                 //
    }
    {
        V4.h = VSHUFF(V4.h)                           //
        V2.w = VDMPY(V2.h, R10.h):sat                 //
    }
    {
        V5.h = VSHUFF(V5.h)                           //
        V3.w = VDMPY(V3.h, R10.h):sat                 //
    }
    {
        V6.h = VSHUFF(V6.h)                           //
        V4.w = VDMPY(V4.h, R10.h):sat                 //
    }
    {
        V7.h = VSHUFF(V7.h)                           //
        V5.w = VDMPY(V5.h, R10.h):sat                 //
    }
    {
        V8.h = VSHUFF(V8.h)                           //
        V6.w = VDMPY(V6.h, R10.h):sat                 //
    }
    {
        V9.h = VSHUFF(V9.h)                           //
        V7.w = VDMPY(V7.h, R10.h):sat                 //
    }
    {
        V10.h = VSHUFF(V10.h)                         //
        V8.w = VDMPY(V8.h, R10.h):sat                 //
    }
    {
        V11.h = VSHUFF(V11.h)                         //
        V9.w = VDMPY(V9.h, R10.h):sat                 //
    }
    {
        V12.h = VSHUFF(V12.h)                         //
        V10.w = VDMPY(V10.h, R10.h):sat               //
    }
    {
        V13.h = VSHUFF(V13.h)                         //
        V11.w = VDMPY(V11.h, R10.h):sat               //
    }
    {
        V14.h = VSHUFF(V14.h)                         //
        V12.w = VDMPY(V12.h, R10.h):sat               //
    }
    {
        V15.h = VSHUFF(V15.h)                         //
        V13.w = VDMPY(V13.h, R10.h):sat               //
    }
    {
        V16.h = VSHUFF(V16.h)                         //
        V14.w = VDMPY(V14.h, R10.h):sat               //
    }
    {
        V17.h = VSHUFF(V17.h)                         //
        V15.w = VDMPY(V15.h, R10.h):sat               //
    }
    {
        V18.h = VSHUFF(V18.h)                         //
        V16.w = VDMPY(V16.h, R10.h):sat               //
    }
    {
        V19.h = VSHUFF(V19.h)                         //
        V17.w = VDMPY(V17.h, R10.h):sat               //
    }
    {
        V20.h = VSHUFF(V20.h)                         //
        V18.w = VDMPY(V18.h, R10.h):sat               //
    }
    {
        V21.h = VSHUFF(V21.h)                         //
        V19.w = VDMPY(V19.h, R10.h):sat               //
    }
    {
        V22.h = VSHUFF(V22.h)                         //
        V20.w = VDMPY(V20.h, R10.h):sat               //
    }
    {
        V23.h = VSHUFF(V23.h)                         //
        V21.w = VDMPY(V21.h, R10.h):sat               //
    }
    {
        V24.h = VSHUFF(V24.h)                         //
        V22.w = VDMPY(V22.h, R10.h):sat               //
    }
    {
        V25.h = VSHUFF(V25.h)                         //
        V23.w = VDMPY(V23.h, R10.h):sat               //
    }
    {
        V26.h = VSHUFF(V26.h)                         //
        V24.w = VDMPY(V24.h, R10.h):sat               //
    }
    {
        V27.h = VSHUFF(V27.h)                         //
        V25.w = VDMPY(V25.h, R10.h):sat               //
    }
    {
        V28.h = VSHUFF(V28.h)                         //
        V26.w = VDMPY(V26.h, R10.h):sat               //
    }
    {
        V29.h = VSHUFF(V29.h)                         //
        V27.w = VDMPY(V27.h, R10.h):sat               //
    }
    {
        V30.h = VSHUFF(V30.h)                         //
        V28.w = VDMPY(V28.h, R10.h):sat               //
    }
    {
        V31.h = VSHUFF(V31.h)                         //
        V29.w = VDMPY(V29.h, R10.h):sat               //
        R28 = #32                                     //
    }
    {
        VSHUFF(V1,V0,R28)                             //
        V30.w = VDMPY(V30.h, R10.h):sat               //
    }
    {
        VSHUFF(V3,V2,R28)                             //
        V31.w = VDMPY(V31.h, R10.h):sat               //
    }
    {
        VSHUFF(V5,V4,R28)                             //
        V0.w = VADD(V1.w, V0.w)                       //
        V2.w = VADD(V3.w, V2.w)                       //
    }
    {
        VSHUFF(V7,V6,R28)                             //
        R7 = #64                                      //
    }
    {
        VSHUFF(V9,V8,R28)                             //
        V4.w = VADD(V5.w, V4.w)                       //
        V6.w = VADD(V7.w, V6.w)                       //
    }
    {
        VSHUFF(V11,V10,R28)                           //
    }
    {
        VSHUFF(V13,V12,R28)                           //
        V8.w = VADD(V9.w, V8.w)                       //
        V10.w = VADD(V11.w, V10.w)                    //
    }
    {
        VSHUFF(V15,V14,R28)                           //
    }
    {
        VSHUFF(V17,V16,R28)                           //
        V12.w = VADD(V13.w, V12.w)                    //
        V14.w = VADD(V15.w, V14.w)                    //
    }
    {
        VSHUFF(V19,V18,R28)                           //
    }
    {
        VSHUFF(V21,V20,R28)                           //
        V16.w = VADD(V17.w, V16.w)                    //
        V18.w = VADD(V19.w, V18.w)                    //
    }
    {
        VSHUFF(V23,V22,R28)                           //
    }
    {
        VSHUFF(V25,V24,R28)                           //
        V20.w = VADD(V21.w, V20.w)                    //
        V22.w = VADD(V23.w, V22.w)                    //
    }
    {
        VSHUFF(V27,V26,R28)                           //
    }
    {
        VSHUFF(V29,V28,R28)                           //
        V24.w = VADD(V25.w, V24.w)                    //
        V26.w = VADD(V27.w, V26.w)                    //
    }
    {
        VSHUFF(V31,V30,R28)                           //
    }
    {
        V28.w = VADD(V29.w, V28.w)                    //
        VSHUFF(V2, V0,R7)                             //
    }
    {
        V30.w = VADD(V31.w, V30.w)                    //
        VSHUFF(V6, V4,R7)                             //
        V0.w  = VADD(V0.w, V2.w)                      //
    }
    {
        VSHUFF(V10,V8,R7)                             //
        V1.tmp = VMEM(R4+#0)                          // update hist[0-15]
        V0.w  = VADD(V0.w, V1.w)                      //
        VMEM(R4++#1) = V0.new                         //
    }
    {
        VSHUFF(V14,V12,R7)                            //
        V4.w  = VADD(V4.w, V6.w)                      //
        V8.w  = VADD(V8.w, V10.w)                     //
    }
    {
        VSHUFF(V18,V16,R7)                            //
        V1.tmp = VMEM(R4+#0)                          // update hist[16-31]
        V4.w  = VADD(V4.w, V1.w)                      //
        VMEM(R4++#1) = V4.new                         //
    }
    {
        VSHUFF(V22,V20,R7)                            //
        V12.w = VADD(V12.w,V14.w)                     //
        V16.w = VADD(V16.w,V18.w)                     //
    }
    {
        VSHUFF(V26,V24,R7)                            //
        V1.tmp = VMEM(R4+#0)                          // update hist[32-47]
        V8.w  = VADD(V8.w, V1.w)                      //
        VMEM(R4++#1) = V8.new                         //
    }
    {
        VSHUFF(V30,V28,R7)                            //
        V1.tmp = VMEM(R4+#0)                          // update hist[48-63]
        V12.w  = VADD(V12.w, V1.w)                    //
        VMEM(R4++#1) = V12.new                        //
    }

    {
        V20.w = VADD(V20.w,V22.w)                     //
        V1.tmp = VMEM(R4+#0)                          // update hist[64-79]
        V16.w  = VADD(V16.w, V1.w)                    //
        VMEM(R4++#1) = V16.new                        //
    }
    {
        V24.w = VADD(V24.w,V26.w)                     //
        V1.tmp = VMEM(R4+#0)                          // update hist[80-95]
        V20.w  = VADD(V20.w, V1.w)                    //
        VMEM(R4++#1) = V20.new                        //
    }
    {
        V28.w = VADD(V28.w,V30.w)                     //
        V1.tmp = VMEM(R4+#0)                          // update hist[96-111]
        V24.w  = VADD(V24.w, V1.w)                    //
        VMEM(R4++#1) = V24.new                        //
    }
    {
        V1.tmp = VMEM(R4+#0)                          // update hist[112-127]
        V28.w  = VADD(V28.w, V1.w)                    //
        VMEM(R4++#1) = V28.new                        //
    }
    {
        JUMPR R31                                     // return
    }
    .size   HistogramPernRow, .-HistogramPernRow

/*[*****************************************************************************]*/
/*[  FUNCTION   : void HistogramBorderPernRow()                                 ]*/
/*[*****************************************************************************]*/
/*[  DESCRIPTION: calcluate histogram of a iaura block                          ]*/
/*[=============================================================================]*/
/*[  INPUTS     : R0 : unsigned char *inp  -- pointer to input iaura            ]*/
/*[               R1 : int stride_i        -- stride of iaura input             ]*/
/*[               R2 : int width           -- width of iaura block              ]*/
/*[               R3 : int height          -- height of iaura block             ]*/
/*[               R4 : int *hist           -- pointer to histogram array        ]*/
/*[               R5 : int border          -- pointer to histogram array        ]*/
/*[=============================================================================]*/
/*[                                                                             ]*/
/*[*****************************************************************************]*/
/* ============================================================ */

/* ============================================================ */
    .p2align 2
    .p2align 4,,15
    .globl  HistogramBorderPernRow
    .type   HistogramBorderPernRow, @function
HistogramBorderPernRow:
    {
        R2 = SUB(R2, R5)                              // width -border
        V1 = #0                                       //
        V0 = #0                                       //
    }
    {
        R2 = LSR(R2,#LOG2VLEN)                        // size/VLEN
        R6 = AND(R2,#VLEN-1)                          // size%VLEN
        P2 = CMP.GT(R2, #0)                           //
    }
    //[----------------------------------------------]
    //[ Step 1: Clean the whole vector Register file ]
    //[ to get ready for vhist                       ]
    //[----------------------------------------------]
    {
        V3:2 = V1:0                                   //
        V5:4 = V1:0                                   //
        P0 = CMP.GT(R2, #0)                           // P0 = (width/VLEN > 0)
        IF !P2 JUMPR R31                              // if width - border <= 0 return
    }
    {
        Q0 = VSETQ(R6)                                //for the last vector
        V7:6 = V1:0                                   //
        P1 = CMP.EQ(R6, #0)                           // P1 = (width%VLEN==0)
    }
    {   V9 = #0                                       //
        V8 = #0                                       //
        V10 = #0                                      //

        Q1 = VSETQ(R5)                                //for the first vector

    }
    {
        V13:12 = V1:0                                 //
        V15:14 = V1:0                                 //

    }
    {
        Q1 = not(Q1)                                  // set border for the fist Vector
        V11 = #0
    }
    {
        V17:16 = V1:0                                 //
        V19:18 = V1:0                                 //
    }
    {
        V21:20 = V1:0                                 //
        V23:22 = V1:0                                 //
        R7 = #1
    }
    {
        V29:28 = V1:0                                 //
        V31:30 = V1:0                                 //
        R10 = ADD(R0,R1)                              // R10 = &src[2*stride]
        IF !P0 JUMP .HistogramBorderLPOneVecter
    }
    //[----------------------------------------------]
    //[ Step 2: vhist                                ]
    //[----------------------------------------------]
    {
        V25:24 = V1:0                                 //
        V27:26 = V1:0
        R2 = SUB(R2, R7)
        LOOP1(.histogram_border_verticalLoop,R3)      // set loop1 with lc1=height
    }
    {
        P0 = CMP.GT(R2, #0)                           // P0 = (width/VLEN -1 > 0)
    }
    .falign
.histogram_border_verticalLoop:
    {
        V0.tmp = VMEM(R0++#1)                         //
        VHIST(Q1)                                     //
        LOOP0(.histogramborderLoop, R2)
        IF !P0 JUMP .histogramborderLPEND
    }
    .falign
.histogramborderLoop:
    {
        V0.tmp = VMEM(R0++#1)                         //
        VHIST                                         //
        NOP; NOP                                      //
    }:endloop0
    .falign
.histogramborderLPEND:
    {
        IF (P1) JUMP .borderskipRemainder             // if (width%VLEN==0) done with current row
    }
    {
        V0.tmp = VMEM(R0+#0)                          // otherwise process the rest pixels
        VHIST(Q0)                                     //
    }
    .falign
.borderskipRemainder:
    {
        R0 = R10                                      // set R0  = &src[(i+1)*stride]
        R10 = ADD(R10,R1)                             // update R10 = &src[(i+2)*stride]
    }:endloop1
    {
        JUMP .histogramborderEND
    }
    //[----------------------------------------------]
    //[        if width < 128                        ]
    //[----------------------------------------------]
    .falign
.HistogramBorderLPOneVecter:
    {
        V25:24 = V1:0                                 //
        V27:26 = V1:0
        LOOP0(.histogramborderOneVectorLoop,R3)       // set loop1 with lc1=height
        IF (P1) JUMP .histogramborderOneVectorLoop
    }
    {
         Q1 = AND(Q0, Q1)                             // for the first and end border
    }
    .falign
.histogramborderOneVectorLoop:
    {
        V0.tmp = VMEM(R0+#0)                          // otherwise process the rest pixels
        VHIST(Q1)                                     //
    }
    {
        R0 = R10                                      // set R0  = &src[(i+1)*stride]
        R10 = ADD(R10,R1)                             // update R10 = &src[(i+2)*stride]
    }:endloop0

    .falign
.histogramborderEND:
    //[----------------------------------------------]
    //[ Step 3:                                      ]
    //[  - Sum up the data in the register file      ]
    //[  - accumulate to hist[]                      ]
    //[----------------------------------------------]
    {
        V0.h = VSHUFF(V0.h)                           //
        R10 = ##0x00010001                            //
    }
    {
        V1.h = VSHUFF(V1.h)                           //
    }
    {
        V2.h = VSHUFF(V2.h)                           //
        V0.w = VDMPY(V0.h, R10.h):sat                 //
    }
    {
        V3.h = VSHUFF(V3.h)                           //
        V1.w = VDMPY(V1.h, R10.h):sat                 //
    }
    {
        V4.h = VSHUFF(V4.h)                           //
        V2.w = VDMPY(V2.h, R10.h):sat                 //
    }
    {
        V5.h = VSHUFF(V5.h)                           //
        V3.w = VDMPY(V3.h, R10.h):sat                 //
    }
    {
        V6.h = VSHUFF(V6.h)                           //
        V4.w = VDMPY(V4.h, R10.h):sat                 //
    }
    {
        V7.h = VSHUFF(V7.h)                           //
        V5.w = VDMPY(V5.h, R10.h):sat                 //
    }
    {
        V8.h = VSHUFF(V8.h)                           //
        V6.w = VDMPY(V6.h, R10.h):sat                 //
    }
    {
        V9.h = VSHUFF(V9.h)                           //
        V7.w = VDMPY(V7.h, R10.h):sat                 //
    }
    {
        V10.h = VSHUFF(V10.h)                         //
        V8.w = VDMPY(V8.h, R10.h):sat                 //
    }
    {
        V11.h = VSHUFF(V11.h)                         //
        V9.w = VDMPY(V9.h, R10.h):sat                 //
    }
    {
        V12.h = VSHUFF(V12.h)                         //
        V10.w = VDMPY(V10.h, R10.h):sat               //
    }
    {
        V13.h = VSHUFF(V13.h)                         //
        V11.w = VDMPY(V11.h, R10.h):sat               //
    }
    {
        V14.h = VSHUFF(V14.h)                         //
        V12.w = VDMPY(V12.h, R10.h):sat               //
    }
    {
        V15.h = VSHUFF(V15.h)                         //
        V13.w = VDMPY(V13.h, R10.h):sat               //
    }
    {
        V16.h = VSHUFF(V16.h)                         //
        V14.w = VDMPY(V14.h, R10.h):sat               //
    }
    {
        V17.h = VSHUFF(V17.h)                         //
        V15.w = VDMPY(V15.h, R10.h):sat               //
    }
    {
        V18.h = VSHUFF(V18.h)                         //
        V16.w = VDMPY(V16.h, R10.h):sat               //
    }
    {
        V19.h = VSHUFF(V19.h)                         //
        V17.w = VDMPY(V17.h, R10.h):sat               //
    }
    {
        V20.h = VSHUFF(V20.h)                         //
        V18.w = VDMPY(V18.h, R10.h):sat               //
    }
    {
        V21.h = VSHUFF(V21.h)                         //
        V19.w = VDMPY(V19.h, R10.h):sat               //
    }
    {
        V22.h = VSHUFF(V22.h)                         //
        V20.w = VDMPY(V20.h, R10.h):sat               //
    }
    {
        V23.h = VSHUFF(V23.h)                         //
        V21.w = VDMPY(V21.h, R10.h):sat               //
    }
    {
        V24.h = VSHUFF(V24.h)                         //
        V22.w = VDMPY(V22.h, R10.h):sat               //
    }
    {
        V25.h = VSHUFF(V25.h)                         //
        V23.w = VDMPY(V23.h, R10.h):sat               //
    }
    {
        V26.h = VSHUFF(V26.h)                         //
        V24.w = VDMPY(V24.h, R10.h):sat               //
    }
    {
        V27.h = VSHUFF(V27.h)                         //
        V25.w = VDMPY(V25.h, R10.h):sat               //
    }
    {
        V28.h = VSHUFF(V28.h)                         //
        V26.w = VDMPY(V26.h, R10.h):sat               //
    }
    {
        V29.h = VSHUFF(V29.h)                         //
        V27.w = VDMPY(V27.h, R10.h):sat               //
    }
    {
        V30.h = VSHUFF(V30.h)                         //
        V28.w = VDMPY(V28.h, R10.h):sat               //
    }
    {
        V31.h = VSHUFF(V31.h)                         //
        V29.w = VDMPY(V29.h, R10.h):sat               //
        R28 = #32                                     //
    }
    {
        VSHUFF(V1,V0,R28)                             //
        V30.w = VDMPY(V30.h, R10.h):sat               //
    }
    {
        VSHUFF(V3,V2,R28)                             //
        V31.w = VDMPY(V31.h, R10.h):sat               //
    }
    {
        VSHUFF(V5,V4,R28)                             //
        V0.w = VADD(V1.w, V0.w)                       //
        V2.w = VADD(V3.w, V2.w)                       //
    }
    {
        VSHUFF(V7,V6,R28)                             //
        R7 = #64                                      //
    }
    {
        VSHUFF(V9,V8,R28)                             //
        V4.w = VADD(V5.w, V4.w)                       //
        V6.w = VADD(V7.w, V6.w)                       //
    }
    {
        VSHUFF(V11,V10,R28)                           //
    }
    {
        VSHUFF(V13,V12,R28)                           //
        V8.w = VADD(V9.w, V8.w)                       //
        V10.w = VADD(V11.w, V10.w)                    //
    }
    {
        VSHUFF(V15,V14,R28)                           //
    }
    {
        VSHUFF(V17,V16,R28)                           //
        V12.w = VADD(V13.w, V12.w)                    //
        V14.w = VADD(V15.w, V14.w)                    //
    }
    {
        VSHUFF(V19,V18,R28)                           //
    }
    {
        VSHUFF(V21,V20,R28)                           //
        V16.w = VADD(V17.w, V16.w)                    //
        V18.w = VADD(V19.w, V18.w)                    //
    }
    {
        VSHUFF(V23,V22,R28)                           //
    }
    {
        VSHUFF(V25,V24,R28)                           //
        V20.w = VADD(V21.w, V20.w)                    //
        V22.w = VADD(V23.w, V22.w)                    //
    }
    {
        VSHUFF(V27,V26,R28)                           //
    }
    {
        VSHUFF(V29,V28,R28)                           //
        V24.w = VADD(V25.w, V24.w)                    //
        V26.w = VADD(V27.w, V26.w)                    //
    }
    {
        VSHUFF(V31,V30,R28)                           //
    }
    {
        V28.w = VADD(V29.w, V28.w)                    //
        VSHUFF(V2, V0,R7)                             //
    }
    {
        V30.w = VADD(V31.w, V30.w)                    //
        VSHUFF(V6, V4,R7)                             //
        V0.w  = VADD(V0.w, V2.w)                      //
    }
    {
        VSHUFF(V10,V8,R7)                             //
        V1.tmp = VMEM(R4+#0)                          // update hist[0-15]
        V0.w  = VADD(V0.w, V1.w)                      //
        VMEM(R4++#1) = V0.new                         //
    }
    {
        VSHUFF(V14,V12,R7)                            //
        V4.w  = VADD(V4.w, V6.w)                      //
        V8.w  = VADD(V8.w, V10.w)                     //
    }
    {
        VSHUFF(V18,V16,R7)                            //
        V1.tmp = VMEM(R4+#0)                          // update hist[16-31]
        V4.w  = VADD(V4.w, V1.w)                      //
        VMEM(R4++#1) = V4.new                         //
    }
    {
        VSHUFF(V22,V20,R7)                            //
        V12.w = VADD(V12.w,V14.w)                     //
        V16.w = VADD(V16.w,V18.w)                     //
    }
    {
        VSHUFF(V26,V24,R7)                            //
        V1.tmp = VMEM(R4+#0)                          // update hist[32-47]
        V8.w  = VADD(V8.w, V1.w)                      //
        VMEM(R4++#1) = V8.new                         //
    }
    {
        VSHUFF(V30,V28,R7)                            //
        V1.tmp = VMEM(R4+#0)                          // update hist[48-63]
        V12.w  = VADD(V12.w, V1.w)                    //
        VMEM(R4++#1) = V12.new                        //
    }

    {
        V20.w = VADD(V20.w,V22.w)                     //
        V1.tmp = VMEM(R4+#0)                          // update hist[64-79]
        V16.w  = VADD(V16.w, V1.w)                    //
        VMEM(R4++#1) = V16.new                        //
    }
    {
        V24.w = VADD(V24.w,V26.w)                     //
        V1.tmp = VMEM(R4+#0)                          // update hist[80-95]
        V20.w  = VADD(V20.w, V1.w)                    //
        VMEM(R4++#1) = V20.new                        //
    }
    {
        V28.w = VADD(V28.w,V30.w)                     //
        V1.tmp = VMEM(R4+#0)                          // update hist[96-111]
        V24.w  = VADD(V24.w, V1.w)                    //
        VMEM(R4++#1) = V24.new                        //
    }
    {
        V1.tmp = VMEM(R4+#0)                          // update hist[112-127]
        V28.w  = VADD(V28.w, V1.w)                    //
        VMEM(R4++#1) = V28.new                        //
    }
    {
        JUMPR R31                                     // return
    }
    .size   HistogramBorderPernRow, .-HistogramBorderPernRow

/* ============================================================ */
    .p2align 2
    .p2align 4,,15
    .globl  ClearHistogram
    .type   ClearHistogram, @function
ClearHistogram:
    {
        LOOP0(.clearHistogram_Loop,#256*4/VLEN)       //
        V0 = #0                                       //
    }
    .falign
.clearHistogram_Loop:
    {
        VMEM(R0++#1) = V0                             //hist[i] = 0
    }:endloop0

    {
        JUMPR R31                                     // return
    }
    .size   ClearHistogram, .-ClearHistogram
