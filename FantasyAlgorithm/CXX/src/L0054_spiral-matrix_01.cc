//
// Created by frewen on 25-2-28.
//
#include<vector>

using namespace std;

class Solution {
public:
  vector<int> spiralOrder(vector<vector<int>>& matrix) {
    // 我们定义一个来编辑我们所有的元素已经访问过的矩阵
    int rows = matrix.size();
    int columns = matrix[0].size();
    vector<vector<bool>> visit(rows,vector<bool>(columns));

    // 然后我们在定义一个方向的矩阵。用来标记顺时针行走的时候，对应的索引应该怎么变化
    vector<vector<int>> directions = {{0,1},{1,0},{0,-1},{-1,0}};

    // 初始化时刻，我们让最开始的索引落到[0,0]。同时我们方向向量的索引为0.也就是顺时针向右走
    int row=0,column = 0;
    int ditectionIndex = 0;
    int total = rows * columns;
    vector<int> ans(total);

    for(int i = 0;i<total;i++) {

      ans[i] = matrix[row][column];
      visit[row][column] = true;

      int nextRow = row + directions[ditectionIndex][0];
      int nextColumn = column + directions[ditectionIndex][1];

      if(nextRow < 0 || nextRow >= rows || nextColumn <0 || nextColumn >= columns || visit[nextRow][nextColumn]) {
        ditectionIndex = (ditectionIndex +1 ) % 4;
      }

      row = row + directions[ditectionIndex][0];
      column = column + directions[ditectionIndex][1];

    }

    return ans;
  }
};