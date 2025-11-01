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
		if (CONDITION_S)
		{
			// x <- q + s
			ACTION_5
		}
		else
		{
			// x <- q
			ACTION_3
		}
	}
	else
	{
		// q = 0
		if (CONDITION_S)
		{
			// x <- s
			ACTION_4
		}
		else
		{
			// new label
			ACTION_2
		}
	}
}
else
{
	// Nothing to do, x is a background pixel
	ACTION_1
}