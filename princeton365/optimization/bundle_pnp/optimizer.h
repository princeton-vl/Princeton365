#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

// Intrinsics plus known distortion parameters (OpenCV 5-coeff model).
struct KnownIntrinsics {
  double fx;
  double fy;
  double cx;
  double cy;

  // Distortion: k1, k2, p1, p2, k3
  double k1;
  double k2;
  double p1;
  double p2;
  double k3;
};

// ---------------------------------------------------------------------
// COST FUNCTOR
// ---------------------------------------------------------------------
// We assume the camera is pre-calibrated, so these 5 distortion params and
// (fx,fy,cx,cy) are known constants. We do *not* solve for them.
//
// The only unknowns are:
//   - camera pose (6D: angle-axis + translation)
//   - board pose  (6D: angle-axis + translation)
//
// The measured 2D point is (observed_u_, observed_v_).
// The local 3D point on the board is p_local_.
//
// Steps:
//   1) board local -> world
//   2) world -> camera
//   3) project with known distortion -> predicted_u, predicted_v
//   4) residual = observed - predicted
//
struct DistortedReprojError {
  DistortedReprojError(double obs_u,
                       double obs_v,
                       const double* p_local,
                       const KnownIntrinsics& intr)
    : observed_u_(obs_u), observed_v_(obs_v),
      fx_(intr.fx), fy_(intr.fy), cx_(intr.cx), cy_(intr.cy),
      k1_(intr.k1), k2_(intr.k2), p1_(intr.p1), p2_(intr.p2), k3_(intr.k3)
  {
    // store 3D local point
    p_local_[0] = p_local[0];
    p_local_[1] = p_local[1];
    p_local_[2] = p_local[2];
  }

  // camera[0..5] = angle-axis + translation
  // board[0..5]  = angle-axis + translation
  template <typename T>
  bool operator()(const T* const camera,
                  const T* const board,
                  T* residual) const
  {
    // 1) board local -> world
    T p_local_t[3] = { T(p_local_[0]),
                       T(p_local_[1]),
                       T(p_local_[2]) };

    T p_world[3];
    ceres::AngleAxisRotatePoint(board, p_local_t, p_world);
    p_world[0] += board[3];
    p_world[1] += board[4];
    p_world[2] += board[5];


    // 2) world -> camera
    T p_cam[3];
    ceres::AngleAxisRotatePoint(camera, p_world, p_cam);
    p_cam[0] += camera[3];
    p_cam[1] += camera[4];
    p_cam[2] += camera[5];

    // 3) Perspective divide
    T xp = p_cam[0] / p_cam[2];
    T yp = p_cam[1] / p_cam[2];

    // OpenCV distortion:
    //   r^2 = x^2 + y^2
    //   x_distorted = x * (1 + k1*r^2 + k2*r^4 + k3*r^6) + ...
    //   y_distorted = y * (1 + k1*r^2 + k2*r^4 + k3*r^6) + ...
    T r2 = xp*xp + yp*yp;
    T r4 = r2*r2;
    T r6 = r4*r2;

    T radial = T(1.0) + T(k1_)*r2 + T(k2_)*r4 + T(k3_)*r6;
    // tangential
    T x_distorted = xp*radial
                    + T(2.0)*T(p1_)*xp*yp
                    + T(p2_)*(r2 + T(2.0)*xp*xp);
    T y_distorted = yp*radial
                    + T(p1_)*(r2 + T(2.0)*yp*yp)
                    + T(2.0)*T(p2_)*xp*yp;

    // 4) final pixel coords
    T predicted_u = T(fx_)*x_distorted + T(cx_);
    T predicted_v = T(fy_)*y_distorted + T(cy_);

    // residual
    residual[0] = T(observed_u_) - predicted_u;
    residual[1] = T(observed_v_) - predicted_v;

    return true;
  }

  // A convenience factory
  static ceres::CostFunction* Create(double obs_u,
                                     double obs_v,
                                     const double* p_local,
                                     const KnownIntrinsics& intr)
  {
    // residual dimension = 2
    // param blocks: camera(6), board(6)
    return new ceres::AutoDiffCostFunction<DistortedReprojError, 2, 6, 6>(
        new DistortedReprojError(obs_u, obs_v, p_local, intr));
  }

 private:
  double observed_u_;
  double observed_v_;
  double p_local_[3];

  double fx_, fy_, cx_, cy_;
  double k1_, k2_, p1_, p2_, k3_;
};

// ---------------------------------------------------------------------
// A generic function that solves any ceres::Problem you provide. 
// It does NOT assume anything about boards, cameras, etc. 
// It's just a wrapper around ceres::Solve(...).
void SolveProblem(ceres::Problem& problem,
                  const ceres::Solver::Options& options,
                  ceres::Solver::Summary& summary);
