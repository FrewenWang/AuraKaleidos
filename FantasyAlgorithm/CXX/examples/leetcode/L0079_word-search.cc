//
// Created by Frewen.Wang on 25-2-26.
//
//79. 单词搜索
//题目链接：https://leetcode.cn/problems/word-search/description/
//给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
//
//单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
//
//
//
//示例 1：
//
//
//输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
//输出：true
//示例 2：
//
//
//输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
//输出：true
//示例 3：
//
//
//输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
//输出：false
//
//
//提示：
//
//m == board.length
//n = board[i].length
//1 <= m, n <= 6
//1 <= word.length <= 15
//board 和 word 仅由大小写英文字母组成


//进阶：你可以使用搜索剪枝的技术来优化解决方案，使其在 board 更大的情况下可以更快解决问题？

#include<vector>
#include<string>

using namespace std;

class Solution {
public:
  bool exist(const vector<vector<char>>& board,string word) {
    // 计算出这个二维矩阵的宽度和高度
    int rows = board.size();
    int columns = board[0].size();
    // 同时，我们定义一个相同的矩阵。来记录某个数据被使用，避免重复使用
    vector<vector<bool>> visited(rows,vector<bool>(columns));
    // 定义一个存放第一个字符的索引位置的二维数组
    const int total = rows* columns;
    int initIndex = 0;
    vector<vector<int>> firstCharIndex(total,vector<int>(2,-1));

    char firstChar = word[0];
    for(int i=0;i< rows; i++) {
      for(int j=0;j< columns; j++) {
        if(firstChar == board[i][j]) {
          firstCharIndex[initIndex][0] = i;
          firstCharIndex[initIndex][1] = j;
          initIndex++;
        }
      }
    }


    bool find = false;
    // 我们依次遍历这个首字母所在的所有可能的位置
    for(int j=0; j< total;j++) {
      int row = firstCharIndex[j][0];
      int column = firstCharIndex[j][1];
      if (row < 0 || column < 0 || row > rows || column > columns) {
        continue;
      }
      if(firstChar != board[row][column] ) {
        continue;
      }
      visited[row][column] = true;
      // 我们就开始遍历后面的
      for(int i=1;i<word.size();i++) {
        // 上
        int nextRow = row-1;
        int nextColumn = column;
        if(nextRow >0 && nextRow< rows && nextColumn>0 && nextColumn< columns
          && !visited[nextRow][nextColumn] && word[i] == board[nextRow][nextColumn]) {
          row = nextRow;
          column = nextColumn;
          visited[row][column] = true;
          if(i == word.size()-1) {
            return true;
          }
          continue;
        }
        // 下
        nextRow = row+1;
        nextColumn = column;
        if(nextRow >0 && nextRow< rows && nextColumn>0 && nextColumn< columns
          && !visited[nextRow][nextColumn] && word[i] == board[nextRow][nextColumn]) {
          row = nextRow;
          column = nextColumn;
          visited[row][column] = true;
          if(i == word.size()-1) {
            return true;
          }
          continue;
        }
        // 左
        nextRow = row;
        nextColumn = column -1;
        if(nextRow >0 && nextRow< rows && nextColumn>0 && nextColumn< columns
          && !visited[nextRow][nextColumn] && word[i] == board[nextRow][nextColumn]) {
          row = nextRow;
          column = nextColumn;
          visited[row][column] = true;
          if(i == word.size()-1) {
            return true;
          }
          continue;
        }
        // 右
        nextRow = row;
        nextColumn = column +1;
        if(nextRow >0 && nextRow< rows && nextColumn>0 && nextColumn< columns
          && !visited[nextRow][nextColumn] && word[i] == board[nextRow][nextColumn]) {
          row = nextRow;
          column = nextColumn;
          visited[row][column] = true;
          if(i == word.size()-1) {
            return true;
          }
          continue;
        }
        break;
      }
    }
    return false;
  }
};

void testCases() {
  vector<vector<char>> case1 = {{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}};
  string word = "ABCCED";
  Solution solution;
  solution.exist(case1,word);

}

int main() {
  testCases();
}