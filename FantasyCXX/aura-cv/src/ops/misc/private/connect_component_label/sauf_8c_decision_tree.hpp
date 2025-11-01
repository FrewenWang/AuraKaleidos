/*
 * This file is generate by GRAPHGEN, which is licensed under the BSD 3-Clause License
 * For more information, see https://github.com/prittt/GRAPHGEN/tree/master
 * Copyright (c) 2020, the respective contributors, as shown by the AUTHORS file.
 * All rights reserved.
 */

if (CONDITION_X)
{
    if (CONDITION_Q)
    {
        // x <- q
        ACTION_4
    }
    else
    {
        // q = 0
        if (CONDITION_R)
        {
            if (CONDITION_P)
            {
                // x <- p + r
                ACTION_7
            }
            else
            {
                // p = q = 0
                if (CONDITION_S)
                {
                    // x <- s + r
                    ACTION_8
                }
                else
                {
                    // p = q = s = 0
                    // x <- r
                    ACTION_5
                }
            }
        }
        else
        {
            // r = q = 0
            if (CONDITION_P)
            {
                // x <- p
                ACTION_3
            }
            else
            {
                // r = q = p = 0
                if (CONDITION_S)
                {
                    // x <- s
                    ACTION_6
                }
                else
                {
                    // New label
                    ACTION_2
                }
            }
        }
    }
}