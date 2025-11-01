#include "aura/aad/utils/math/aura_box2d.h"

#include <algorithm>
#include <cmath>

namespace aura{
namespace aad{

Box2d::Box2d(const Vector2d &center, const double length, const double width)
        : center_(center),
          length_(length),
          width_(width),
          half_length_(length / 2.0),
          half_width_(width / 2.0) {
}

Box2d::Box2d(const Vector2d &one_corner, const Vector2d &opposite_corner)
        : Box2d((one_corner + opposite_corner) / 2.0,
                std::abs(one_corner.x() - opposite_corner.x()),
                std::abs(one_corner.y() - opposite_corner.y())) { }

Box2d::Box2d(const safe::vector<Vector2d> &points) {
  double min_x = points[0].x();
  double max_x = points[0].x();
  double min_y = points[0].y();
  double max_y = points[0].y();
  for (const auto &point: points) {
    min_x = std::min(min_x, point.x());
    max_x = std::max(max_x, point.x());
    min_y = std::min(min_y, point.y());
    max_y = std::max(max_y, point.y());
  }
  
  center_ = {(min_x + max_x) / 2.0, (min_y + max_y) / 2.0};
  length_ = max_x - min_x;
  width_ = max_y - min_y;
  half_length_ = length_ / 2.0;
  half_width_ = width_ / 2.0;
}

void Box2d::GetAllCorners(safe::vector<Vector2d> *const corners) const {
  corners->reserve(4);
  corners->emplace_back(center_.x() + half_length_, center_.y() - half_width_);
  corners->emplace_back(center_.x() + half_length_, center_.y() + half_width_);
  corners->emplace_back(center_.x() - half_length_, center_.y() + half_width_);
  corners->emplace_back(center_.x() - half_length_, center_.y() - half_width_);
}

bool Box2d::IsPointIn(const Vector2d &point) const {
  return std::abs(point.x() - center_.x()) <= half_length_ + kMathEpsilon &&
         std::abs(point.y() - center_.y()) <= half_width_ + kMathEpsilon;
}

bool Box2d::IsPointOnBoundary(const Vector2d &point) const {
  const double dx = std::abs(point.x() - center_.x());
  const double dy = std::abs(point.y() - center_.y());
  return (std::abs(dx - half_length_) <= kMathEpsilon &&
          dy <= half_width_ + kMathEpsilon) ||
         (std::abs(dy - half_width_) <= kMathEpsilon &&
          dx <= half_length_ + kMathEpsilon);
}

double Box2d::DistanceTo(const Vector2d &point) const {
  const double dx = std::abs(point.x() - center_.x()) - half_length_;
  const double dy = std::abs(point.y() - center_.y()) - half_width_;
  if (dx <= 0.0) {
    return std::max(0.0, dy);
  }
  if (dy <= 0.0) {
    return dx;
  }
  return hypot(dx, dy);
}

double Box2d::DistanceTo(const Box2d &box) const {
  const double dx =
          std::abs(box.center_x() - center_.x()) - box.half_length() - half_length_;
  const double dy =
          std::abs(box.center_y() - center_.y()) - box.half_width() - half_width_;
  if (dx <= 0.0) {
    return std::max(0.0, dy);
  }
  if (dy <= 0.0) {
    return dx;
  }
  return hypot(dx, dy);
}

bool Box2d::HasOverlap(const Box2d &box) const {
  return std::abs(box.center_x() - center_.x()) <=
         box.half_length() + half_length_ &&
         std::abs(box.center_y() - center_.y()) <=
         box.half_width() + half_width_;
}

void Box2d::Shift(const Vector2d &shift_vec) { center_ += shift_vec; }

void Box2d::MergeFrom(const Box2d &other_box) {
  const double x1 = std::min(min_x(), other_box.min_x());
  const double x2 = std::max(max_x(), other_box.max_x());
  const double y1 = std::min(min_y(), other_box.min_y());
  const double y2 = std::max(max_y(), other_box.max_y());
  center_ = Vector2d((x1 + x2) / 2.0, (y1 + y2) / 2.0);
  length_ = x2 - x1;
  width_ = y2 - y1;
  half_length_ = length_ / 2.0;
  half_width_ = width_ / 2.0;
}

void Box2d::MergeFrom(const Vector2d &other_point) {
  const double x1 = std::min(min_x(), other_point.x());
  const double x2 = std::max(max_x(), other_point.x());
  const double y1 = std::min(min_y(), other_point.y());
  const double y2 = std::max(max_y(), other_point.y());
  center_ = Vector2d((x1 + x2) / 2.0, (y1 + y2) / 2.0);
  length_ = x2 - x1;
  width_ = y2 - y1;
  half_length_ = length_ / 2.0;
  half_width_ = width_ / 2.0;
}

void Box2d::set_center(const Vector2d &center) {
  center_.set_x(center.x());
  center_.set_y(center.y());
}

void Box2d::set_length(const double &length) {
  length_ = length;
  half_length_ = length_ / 2;
}

void Box2d::set_width(const double &width) {
  width_ = width;
  half_width_ = width_ / 2;
}

}  // namespace aura
}  // namespace aad

