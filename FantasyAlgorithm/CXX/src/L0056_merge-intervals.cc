//
// Created by frewen on 25-2-14.
//
//56. 合并区间
//题目链接：https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2&envId=top-interview-150
//以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
//
//
//
//示例 1：
//
//输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
//输出：[[1,6],[8,10],[15,18]]
//解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
//示例 2：
//
//输入：intervals = [[1,4],[4,5]]
//输出：[[1,5]]
//解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
//
//
//提示：
//
//1 <= intervals.length <= 104
//intervals[i].length == 2
//0 <= starti <= endi <= 104
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
		// 首先，针对区间进行排序，然后进行合并。
        // 排序很重要，刚开始没有做出来就是忘了进行排查，不排序会导致合并区间变得异常复杂。
        sort(intervals.begin(), intervals.end());
        // 特殊处理：如果只有一个区间，直接返回。
        vector<vector<int>> results;
        results.push_back(intervals[0]);
        int i = 1;
        int n = intervals.size();
        // 遍历后面待排序的数据。然后依次取出数据，和数据中最后一个数据进行比较。
        // 如果存储交叉区间，那么久更新区间，否则，直接添加。
        while(i < n) {
            vector<int> &temp_result = intervals[i];
            vector<int> &result =results.back();
           if (result[1] >= temp_result[0]) {
                result[0] = std::min(temp_result[0],result[0]);
                result[1] = std::max(temp_result[1],result[1]);
            } else {
                results.push_back(temp_result);
            }
            i++;

        }
        return results;
    }

    /**
	 * 使用排序法
	 */
    vector<vector<int>> mergeV2(vector<vector<int>>& intervals) {
		// 首先，针对区间进行排序，然后进行合并。
        // 排序很重要，刚开始没有做出来就是忘了进行排查，不排序会导致合并区间变得异常复杂。
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> merged;
        merged.push_back(intervals[0]);
        int i = 1;
        int n = intervals.size();
        // 遍历后面待排序的数据。然后依次取出数据，和数据中最后一个数据进行比较。
        // 如果存储交叉区间，那么久更新区间，否则，直接添加。
        while(i < n) {
            vector<int> &temp_result = intervals[i];
            vector<int> &result = merged.back();
           if (result[1] >= temp_result[0]) {
                result[0] = std::min(temp_result[0],result[0]);
                result[1] = std::max(temp_result[1],result[1]);
            } else {
                merged.push_back(temp_result);
            }
            i++;

        }
        return merged;
    }

};