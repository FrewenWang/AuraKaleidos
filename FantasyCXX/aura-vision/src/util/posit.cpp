#include <iostream>
#include <fstream>
#include <vector>
#include "posit.h"

void modern_posit(Mat_<float> &image_points, Mat_<float> &object_points, float focal_length, Point2f center,
                  Mat_<float> &rot, Mat_<float> &trans) {
    Mat_<float> centeredImage = image_points.clone();
    centeredImage.col(0) -= center.x;
    centeredImage.col(1) -= center.y;
    const double f = 1.0 / norm(centeredImage);
    centeredImage.convertTo(centeredImage, -1, f);
    centeredImage /= focal_length;
    Mat_<float> ui = centeredImage.col(0).clone();
    Mat_<float> vi = centeredImage.col(1).clone();
    Mat_<float> wi = Mat_<float>::ones(centeredImage.rows, 1);

    Mat_<float> homogeneousWorldPts = Mat_<float>::ones(object_points.rows, object_points.cols + 1);
    object_points.copyTo(homogeneousWorldPts.colRange(0, object_points.cols));

    Mat_<float> objectMat = homogeneousWorldPts.inv(DECOMP_SVD);
    Mat_<float> r1, r2, r3;
    float Tx = 0.0f;
    float Ty = 0.0f;
    float Tz = 0.0f;
    bool converged = false;
    int count = 0;
    while (!converged) {
        Mat_<float> r1T = objectMat * ui;
        Mat_<float> r2T = objectMat * vi;
        float Tz1 = 1.0f /
                    sqrt(r1T(0, 0) * r1T(0, 0) + r1T(1, 0) * r1T(1, 0) + r1T(2, 0) * r1T(2, 0)); // 1/Tz1 is norm of r1T
        float Tz2 = 1.0f / sqrt(r2T(0, 0) * r2T(0, 0) + r2T(1, 0) * r2T(1, 0) + r2T(2, 0) * r2T(2, 0));
        Tz = sqrt(Tz1 * Tz2); // geometric average instead of arythmetic average of classicPosit
        Mat_<float> r1N = r1T * Tz;
        Mat_<float> r2N = r2T * Tz;

        r1 = r1N.rowRange(0, 3).clone();
        r2 = r2N.rowRange(0, 3).clone();
        r3 = r1.cross(r2);

        Mat_<float> r3T(r3.rows + 1, r3.cols);
        r3.copyTo(r3T.rowRange(0, r3.rows));
        r3T(3, 0) = Tz;

        Tx = r1N(3, 0);
        Ty = r2N(3, 0);
        wi = homogeneousWorldPts * r3T / Tz;

        Mat_<float> oldUi = ui.clone();
        Mat_<float> oldVi = vi.clone();
        ui = wi.mul(centeredImage.col(0));
        vi = wi.mul(centeredImage.col(1));
        Mat_<float> deltaUi = ui - oldUi;
        Mat_<float> deltaVi = vi - oldVi;

        float delta = focal_length * focal_length * (deltaUi.dot(deltaUi) + deltaVi.dot(deltaVi));
        converged = (count > 0) & (delta < 1);
        count = count + 1;
        if (count > 10)
            break;
    }
    rot.create(3, 3);
    trans.create(3, 1);
    trans(0, 0) = Tx;
    trans(1, 0) = Ty;
    trans(2, 0) = Tz;
    r1.copyTo(rot.col(0));
    r2.copyTo(rot.col(1));
    r3.copyTo(rot.col(2));
    rot = rot.t();
}

//-----------------------------------------------------------------------------
//ref: Computing Euler angles from a rotation matrix--Gregory G. Slabaugh
void rotation_matrix_from_euler_angle(Mat_<float> &matrix, const float roll, const float pitch, const float yaw) {
    const double psi = pitch; //x
    const double theta = yaw; //y
    const double phi = roll; //z

    const double sinPhi = sin(phi);
    const double cosPhi = cos(phi);
    const double sinPsi = sin(psi);
    const double cosPsi = cos(psi);
    const double sinTheta = sin(theta);
    const double cosTheta = cos(theta);

    if (matrix.empty())
        matrix.create(3, 3);

    matrix(0, 0) = cosTheta * cosPhi;
    matrix(0, 1) = sinPsi * sinTheta * cosPhi - cosPsi * sinPhi;
    matrix(0, 2) = cosPsi * sinTheta * cosPhi + sinPsi * sinPhi;
    matrix(1, 0) = cosTheta * sinPhi;
    matrix(1, 1) = sinPsi * sinTheta * sinPhi + cosPsi * cosPhi;
    matrix(1, 2) = cosPsi * sinTheta * sinPhi - sinPsi * cosPhi;
    matrix(2, 0) = -sinTheta;
    matrix(2, 1) = sinPsi * cosTheta;
    matrix(2, 2) = cosPsi * cosTheta;
}

//-----------------------------------------------------------------------------
//ref: Computing Euler angles from a rotation matrix--Gregory G. Slabaugh
void rotation_matrix_to_euler_angle(const Mat &rot_matrix, float &roll, float &pitch, float &yaw) {
    // do we have to transpose here?
    const double a11 = rot_matrix.at<float>(0, 0), a12 = rot_matrix.at<float>(0, 1), a13 = rot_matrix.at<float>(0, 2);
    const double a21 = rot_matrix.at<float>(1, 0), a22 = rot_matrix.at<float>(1, 1), a23 = rot_matrix.at<float>(1, 2);
    const double a31 = rot_matrix.at<float>(2, 0), a32 = rot_matrix.at<float>(2, 1), a33 = rot_matrix.at<float>(2, 2);

    double psi = 0.0f;
    double theta = 0.0f;
    double phi = 0.0f;//euler angle radians about the x,y,z axis
    if (abs(1.0 - a31) <= epsilon) // special case a31 == +1
    {
        cerr << "gimbal lock case a31 == " << a31;
        phi = 0; //arbitrary value set to 0
        theta = -pi_over_2;
        psi = atan2(a12, a13) - phi;
    } else if (abs(-1.0 - a31) <= epsilon) // special case a31 == -1
    {
        cerr << "gimbal lock case a31 == " << a31;
        phi = 0; //arbitrary value set to 0
        theta = pi_over_2;
        psi = atan2(a12, a13) + phi;
    } else // standard case a31 != +/-1
    {
        theta = asin(-a31); //yaw in [-90 90]

        float cos_theta = cos(theta);
        psi = atan2(a32 / cos_theta, a33 / cos_theta); // pitch in [-90, 90]
        phi = atan2(a21 / cos_theta, a11 / cos_theta); //roll in [-180, 180]
        /*
        psi = a33 > 0 ? atan2(a32, a33) : atan2(-a32, -a33);
        phi = a11 > 0 ? atan2(a12, a11) : atan2(-a12, -a11);
                */
    }
    pitch = psi;
    yaw = theta;
    roll = phi;
}
