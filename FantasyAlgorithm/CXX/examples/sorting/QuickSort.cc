//
// Created by Frewen.Wang on 25-3-13.
//
#include<iostream>

using namespace std;

class Solution {
public:
    void sort() {
      vector<int> nums = {};
      int l = 0, r=nums.size();
       QuickSort(nums,l,r);
    }

    void QuickSort(vector<int> &arr, int left, int right) {
    while (left < right) {
        int i, j, x;
        i = left;
        j = right;
        x = arr[i];

        while (i < j) {
            while (i < j && arr[j] > x)
                j--; // 从右向左找第一个小于x的数
            if (i < j)
                arr[i++] = arr[j];

            while (i < j && arr[i] < x)
                i++; // 从左向右找第一个大于x的数
            if (i < j)
                arr[j--] = arr[i];
        }
        arr[i] = x;//中值位归位
        QuickSort(arr, left, i - 1); /* 递归调用 */
        QuickSort(arr, i + 1, right); /* 递归调用 */
    }
}

};