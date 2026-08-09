//
// Created by Frewen.Wang on 25-2-25.
//
//54. 螺旋矩阵
//题目链接：https://leetcode.cn/problems/spiral-matrix/description/
//给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
//
//
//
//示例 1：
//
//
//输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
//输出：[1,2,3,6,9,8,7,4,5]
//示例 2：
//
//
//输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
//输出：[1,2,3,4,8,12,11,10,9,5,6,7]
//
//
//提示：
//
//m == matrix.length
//n == matrix[i].length
//1 <= m, n <= 10
//-100 <= matrix[i][j] <= 100

#include<algorithm>
#include<vector>

using namespace std;

class Solution{

enum Direction {
    RIGHT = 0,
    BOTTOM = 1,
    LEFT = 2,
    TOP = 3
};

public:
  /**
   * 算法核心思想： 本质上这种二维矩阵的遍历，主要就是看位置[row][column] 到底怎么去计算
   * 比如这个题目中：首先你要计算好方向。那这个方向既然是顺时针，那么我们所以就定义了一个ENUM变量来进行方向的约束
   * 但是这里面有个很尴尬的点：就是下一个方向的计算。需要+1循环遍历
   * 同时：所有的C++的算法题主要就是边界条件的考虑需要完善。
   * 那么这道题目里面的边界条件就是： 你下一个需要移到的位置上不能出现矩阵数据越界，也就是必须在[0,size-1]这个索引里面
   * 同时，需要记住：访问过的元素不能访问。
   * 思考：我们这个方向写的不是很好。这个有调整的空间。看看原题的解题思路
   */
//  vector<int> spiralOrder (vector<vector<int>> & matrix) {
//      // 计算这个矩阵的宽度和高度
//      int rows = matrix.size();
//      int columns = matrix[0].size();
//      // 定义一个同样的矩阵，这个矩阵李存放bool型的变量，用来标记对应的位置的元素是否已经遍历过
//      vector<vector<bool>> visited(rows,vector<bool>(columns));
//      // 同时我们准备一个数组向量，然后来存储我们遍历的结果
//      int total = rows * columns;
//      vector<int> result(total);
//      // 进行开始索引遍历
//      int direction = static_cast<int>(RIGHT);
//      int row = 0, column = 0;
//      for(int i=0;i<total;i++) {
//        // 将对应的行列数据复制给我们的结果数据
//        result[i] = matrix[row][column];
//        // 同时标记对应的行列数据为已经访问过的
//        visited[row][column] = true;
//
//        /// 我们根据我们索要运行的方向，来暂时计算一下下一个位置点
//        // 但是计算出来位置点之后，我们需要判断这个位置是否有效
//        int nextRow = row;
//        int nextColumn = column;
//        // 下面我们就要开始计算下一个位置到底在哪里
//        switch(direction) {
//          case 0:
//            nextColumn++;
//            break;
//          case 1:
//            nextRow++;
//            break;
//          case 2:
//            nextColumn--;
//            break;
//          case 3:
//            nextRow--;
//            break;
//        }
//        // 这个就是来进行标记说。他下一个需要移到的位置是否有效，如果无效，我们可以纪要调转方向了。
//        if(nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns ||visited[nextRow][nextColumn]) {
//          direction = static_cast<int>((direction+1) % 4);
//        }
//        // 调转方向之后，我们再真正的去计算下一个坐标点
//        switch(direction) {
//          case 0:
//            column++;
//          break;
//          case 1:
//            row++;
//          break;
//          case 2:
//            column--;
//          break;
//          case 3:
//            row--;
//          break;
//        }
//      }
//       return result;
//  }

    /**
    *  这个解决方法跟原题的题目解决是一样的。
    */
    vector<int> spiralOrder (vector<vector<int>> & matrix) {
        static constexpr int directions[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};
        // 计算这个矩阵的宽度和高度
        int rows = matrix.size();
        int columns = matrix[0].size();
        // 定义一个同样的矩阵，这个矩阵李存放bool型的变量，用来标记对应的位置的元素是否已经遍历过
        vector<vector<bool>> visited(rows,vector<bool>(columns));
        // 同时我们准备一个数组向量，然后来存储我们遍历的结果
        int total = rows * columns;
        vector<int> result(total);
        // 进行开始索引遍历
        int directionIndex = 0;
        int row = 0, column = 0;
        for(int i=0;i<total;i++) {
          // 将对应的行列数据复制给我们的结果数据
          result[i] = matrix[row][column];
          // 同时标记对应的行列数据为已经访问过的
          visited[row][column] = true;

          /// 我们根据我们索要运行的方向，来暂时计算一下下一个位置点
          // 但是计算出来位置点之后，我们需要判断这个位置是否有效
          int nextRow = row + directions[directionIndex][0];
          int nextColumn = column + directions[directionIndex][1];

          // 这个就是来进行标记说。他下一个需要移到的位置是否有效，如果无效，我们可以纪要调转方向了。
          if(nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns ||visited[nextRow][nextColumn]) {
            directionIndex = static_cast<int>((directionIndex+1) % 4);
          }
          row = row + directions[directionIndex][0];
          column = column + directions[directionIndex][1];
        }
         return result;
    }

};