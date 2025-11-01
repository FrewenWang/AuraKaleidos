//
// Created by Frewen.Wang on 25-2-25.
//

#include<algorithm>
#include<iostream>

using namespace std;

struct BBox {
  double x1,y1;
  double x2,y2;
};

double calIOU(const BBox &box1, const BBox &box2){
  // 计算出来IOU框相交电的左上角和右下角的坐标
  double pltx1 = max(box1.x1,box2.x1);
  double plty1 = max(box1.y1,box2.y1);

  double prbx2 = min(box1.x2,box2.x2);
  double prby2 = min(box1.y2,box2.y2);

  //  std::cout << "pltx1:" << pltx1 << std::endl;
  //  std::cout << "plty1:" << plty1 << std::endl;
  //  std::cout << "prbx2:" << prbx2 << std::endl;
  //  std::cout << "prby2:" << prby2 << std::endl;


  double interArea = max(0.0,(prbx2-pltx1)) * max(0.0,(prby2 -plty1));
  std::cout << "interArea:" << interArea << std::endl;

  double area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  double area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

  double unitArea = area1 + area2 - interArea;
  std::cout << "unitArea:" << unitArea << std::endl;

  return interArea/unitArea;
}

int main(int argc, const char** argv) {
  BBox box1 = {0,0,2,2};
  BBox box2 = {1,1,3,3};

  double iou = calIOU(box1,box2);
  std::cout << "IOU:" << iou << std::endl;

  return 0;
}
