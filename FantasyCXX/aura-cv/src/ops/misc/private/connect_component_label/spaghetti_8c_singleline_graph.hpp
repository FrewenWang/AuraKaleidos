/*
 * This file is generate by GRAPHGEN, which is licensed under the BSD 3-Clause License
 * For more information, see https://github.com/prittt/GRAPHGEN/tree/master
 * Copyright (c) 2020, the respective contributors, as shown by the AUTHORS file.
 * All rights reserved.
 */

sl_tree_0 : if ((x += 2) >= width - 2)
{
	if (x > width - 2)
	{
		goto sl_break_0_0;
	}
	else
	{
		goto sl_break_1_0;
	}
}
if (CONDITION_O)
{
	if (CONDITION_P)
	{
		ACTION_2
		goto sl_tree_1;
	}
	else
	{
		ACTION_2
		goto sl_tree_0;
	}
}
else
{
NODE_372:
	if (CONDITION_P)
	{
		ACTION_2
		goto sl_tree_1;
	}
	else
	{
		ACTION_1
		goto sl_tree_0;
	}
}
sl_tree_1 : if ((x += 2) >= width - 2)
{
	if (x > width - 2)
	{
		goto sl_break_0_1;
	}
	else
	{
		goto sl_break_1_1;
	}
}
if (CONDITION_O)
{
	if (CONDITION_P)
	{
		ACTION_6
		goto sl_tree_1;
	}
	else
	{
		ACTION_6
		goto sl_tree_0;
	}
}
else
{
	goto NODE_372;
}
sl_break_0_0 : if (CONDITION_O)
{
	ACTION_2
}
else
{
	ACTION_1
}
goto end_sl;
sl_break_0_1 : if (CONDITION_O)
{
	ACTION_6
}
else
{
	ACTION_1
}
goto end_sl;
sl_break_1_0 : if (CONDITION_O)
{
	if (CONDITION_P)
	{
		ACTION_2
	}
	else
	{
		ACTION_2
	}
}
else
{
NODE_375:
	if (CONDITION_P)
	{
		ACTION_2
	}
	else
	{
		ACTION_1
	}
}
goto end_sl;
sl_break_1_1 : if (CONDITION_O)
{
	if (CONDITION_P)
	{
		ACTION_6
	}
	else
	{
		ACTION_6
	}
}
else
{
	goto NODE_375;
}
goto end_sl;
end_sl :;
