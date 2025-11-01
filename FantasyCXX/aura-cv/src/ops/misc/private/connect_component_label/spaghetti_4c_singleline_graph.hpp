/*
 * This file is generate by GRAPHGEN, which is licensed under the BSD 3-Clause License
 * For more information, see https://github.com/prittt/GRAPHGEN/tree/master
 * Copyright (c) 2020, the respective contributors, as shown by the AUTHORS file.
 * All rights reserved.
 */

goto cl_tree_0;
cl_tree_0 : if ((x += 1) >= width) goto cl_break;
if (CONDITION_X)
{
	if (CONDITION_Q)
	{
		ACTION_3
		goto cl_tree_1;
	}
	else
	{
		ACTION_2
		goto cl_tree_1;
	}
}
else
{
	ACTION_1
	goto cl_tree_0;
}
cl_tree_1 : if ((x += 1) >= width) goto cl_break;
if (CONDITION_X)
{
	if (CONDITION_Q)
	{
		ACTION_5
		goto cl_tree_1;
	}
	else
	{
		ACTION_4
		goto cl_tree_1;
	}
}
else
{
	ACTION_1
	goto cl_tree_0;
}
cl_break :;