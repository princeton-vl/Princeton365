#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "include/cnpy.h"  // Include the cnpy header

// Use nlohmann/json for JSON handling
using json = nlohmann::json;

// A known intrinsics + distortion structure
// We'll assume each camera has (fx,fy,cx,cy,k1,k2,p1,p2,k3).
struct KnownIntrinsics {
  double fx, fy;
  double cx, cy;
  double k1, k2, p1, p2, k3;
};

// ---------------------------------------------------------------------
// Helper transforms for angle-axis + translation.
//
// 1) TransformPoint(A_T_B, p_B, p_A):
//    Applies the 6D angle-axis+translation transform (B → A) to a 3D point p_B.
//
// 2) InvertTransform(A_T_B, B_T_A):
//    Given a transform A_T_B (B→A), produces its inverse B_T_A (A→B).
// ---------------------------------------------------------------------
template <typename T>
inline void TransformPoint(const T* A_T_B, const T* p_B, T* p_A)
{
  // Rotate
  ceres::AngleAxisRotatePoint(A_T_B, p_B, p_A);
  // Then translate
  p_A[0] += A_T_B[3];
  p_A[1] += A_T_B[4];
  p_A[2] += A_T_B[5];
}

template <typename T>
inline void InvertTransform(const T* A_T_B, T* B_T_A)
{
  // The inverse rotation is simply the negative of the angle-axis
  B_T_A[0] = -A_T_B[0];
  B_T_A[1] = -A_T_B[1];
  B_T_A[2] = -A_T_B[2];

  // The inverse translation is the negative of the old translation,
  // but we must rotate it by the *new* rotation (which is -axis).
  T negTrans[3] = { -A_T_B[3], -A_T_B[4], -A_T_B[5] };
  ceres::AngleAxisRotatePoint(B_T_A, negTrans, &B_T_A[3]);
}

// ---------------------------------------------------------------------
// parseBoardData (as provided)
// ---------------------------------------------------------------------
std::unordered_map<int, std::vector<double>> parseBoardData(const std::string& board_data_str) {
    std::unordered_map<int, std::vector<double>> board_map;

    // Split the input into node_id:pose pairs
    std::istringstream ss(board_data_str);
    std::string item;
    while (std::getline(ss, item, ';')) {
        // Split node_id and pose
        size_t colon_pos = item.find(':');
        if (colon_pos == std::string::npos) {
            std::cerr << "Error: Invalid board data format: " << item << "\n";
            continue;
        }

        int node_id = std::stoi(item.substr(0, colon_pos)); 
        std::string pose_str = item.substr(colon_pos + 1);

        // Parse pose into a vector of doubles
        std::vector<double> pose;
        std::istringstream pose_stream(pose_str);
        std::string value;
        while (std::getline(pose_stream, value, ',')) {
            pose.push_back(std::stod(value));
        }

        if (pose.size() != 6) {
            std::cerr << "Error: Pose for node " << node_id << " must have exactly 6 elements.\n";
            continue;
        }

        board_map[node_id] = pose;
    }

    return board_map;
}

// ---------------------------------------------------------------------
// Structure to hold observations
// ---------------------------------------------------------------------
struct Obs {
    int board_id;  // Board identifier
  int frame_idx; // Frame index
  int cam_idx;   // Camera index (0 or 1)
  double Xb, Yb, Zb; // 3D point in board coordinates
  double u, v;       // 2D observation in pixel coordinates
};

// ---------------------------------------------------------------------
// Read observations from JSON files
// ---------------------------------------------------------------------

std::vector<Obs> ReadObservationsFromJson(const std::vector<std::string>& filenames) {
  std::vector<Obs> observations;
  
  for (size_t j = 0; j < filenames.size(); ++j) {
    
    std::string filename = filenames[j];
    int cam_idx = j;
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Error opening file: " + filename);
    }
    std::cout << "Reading observations from " << filename << std::endl;
    json json_data;
    file >> json_data;
    file.close();
    int total_points = 0;
    
    // Process each frame/board group
    for (const auto& group : json_data) {
      int frame_idx = group["f"];

    // Handle string board IDs
    std::string board_id_str = group["b_id"];
    int board_id;
    if (board_id_str.find('_') != std::string::npos) {
        board_id = std::stoi(board_id_str.substr(0, board_id_str.find('_')));
    } else {
        try {
            board_id = std::stoi(board_id_str);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing board_id: " << board_id_str << std::endl;
            board_id = -1; // Or some default value
        }
    }
      // Get 2D and 3D point arrays (now they're arrays of arrays)
      const auto& points_2d = group["2d"];
      const auto& points_3d = group["3d"];
      
      if (points_2d.size() != points_3d.size()) {
        std::cerr << "Warning: Mismatch in 2D/3D point counts in " << filename 
                  << " for frame " << frame_idx << ", board " << board_id << std::endl;
        continue;
      }
      
      // Process each point pair
      for (size_t i = 0; i < points_3d.size(); ++i) {
        Obs ob;
        ob.cam_idx = cam_idx;
        ob.frame_idx = frame_idx;
        ob.board_id = board_id;
        
        // Get 3D point from [x,y,z] array
        const auto& point3d = points_3d[i];
        ob.Xb = point3d[0];
        ob.Yb = point3d[1];
        ob.Zb = point3d[2];
        
        // Get 2D point from [u,v] array
        const auto& point2d = points_2d[i];
        ob.u = point2d[0];
        ob.v = point2d[1];
        
        observations.push_back(ob);
        total_points++;
      }
    }
    
    std::cout << "Read " << json_data.size() << " board/frame groups with " 
              << total_points << " total points from " << filename << std::endl;
  }
  
  return observations;
}

// Function to split a string by a delimiter
std::vector<double> splitAndConvert(const std::string& str, char delimiter) {
  std::vector<double> elements;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    elements.push_back(std::stod(item));
  }
  return elements;
}

KnownIntrinsics parseCameraIntrinsics(const std::string& matrix_str, const std::string& dist_coeffs_str) {
  // Parse camera matrix
  std::vector<double> camera_matrix_flat = splitAndConvert(matrix_str, ',');
  if (camera_matrix_flat.size() != 9) {
      throw std::runtime_error("Error: Camera matrix must have exactly 9 elements.");
  }

  Eigen::Matrix3d camera_matrix;
  camera_matrix << camera_matrix_flat[0], camera_matrix_flat[1], camera_matrix_flat[2],
                   camera_matrix_flat[3], camera_matrix_flat[4], camera_matrix_flat[5],
                   camera_matrix_flat[6], camera_matrix_flat[7], camera_matrix_flat[8];

  // Parse distortion coefficients
  std::vector<double> dist_coeffs = splitAndConvert(dist_coeffs_str, ',');
  if (dist_coeffs.size() != 5) {
      throw std::runtime_error("Error: Distortion coefficients must have exactly 5 elements.");
  }

  // Store intrinsics
  KnownIntrinsics intr;
  intr.fx = camera_matrix(0, 0);
  intr.fy = camera_matrix(1, 1);
  intr.cx = camera_matrix(0, 2);
  intr.cy = camera_matrix(1, 2);
  intr.k1 = dist_coeffs[0];
  intr.k2 = dist_coeffs[1];
  intr.p1 = dist_coeffs[2];
  intr.p2 = dist_coeffs[3];
  intr.k3 = dist_coeffs[4];

  return intr;
}



// Read camera intrinsics from JSON file
std::unordered_map<int, KnownIntrinsics> readCameraIntrinsics(const std::string& filename) {
  std::unordered_map<int, KnownIntrinsics> intrinsics;
  
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening file: " + filename);
  }
  
  json json_data;
  file >> json_data;
  file.close();
  
  for (auto it = json_data.begin(); it != json_data.end(); ++it) {
    int cam_idx = std::stoi(it.key());
    const auto& cam_data = it.value();
    
    KnownIntrinsics intr;
    intr.fx = cam_data["fx"];
    intr.fy = cam_data["fy"];
    intr.cx = cam_data["cx"];
    intr.cy = cam_data["cy"];
    intr.k1 = cam_data["k1"];
    intr.k2 = cam_data["k2"];
    intr.p1 = cam_data["p1"];
    intr.p2 = cam_data["p2"];
    intr.k3 = cam_data["k3"];
    
    intrinsics[cam_idx] = intr;
  }
  
  std::cout << "Read intrinsics for " << intrinsics.size() << " cameras from " << filename << std::endl;
  return intrinsics;
}


void PrintObservations(const std::vector<Obs>& observations) {
  std::cout << "=== Observations ===" << std::endl;
  for (const auto &obs : observations) {
    std::cout << "Board ID: " << obs.board_id
              << ", Frame: " << obs.frame_idx
              << ", Camera Index: " << obs.cam_idx
              << ", 3D Point (Xb, Yb, Zb): ("
              << obs.Xb << ", " << obs.Yb << ", " << obs.Zb << ")"
              << ", 2D Observation (u, v): ("
              << obs.u << ", " << obs.v << ")"
              << std::endl;
  }
}


void PrintWorldTBoardsEst(const std::unordered_map<int, std::vector<double>>& world_T_boards_est) {
  std::cout << "=== world_T_boards_est ===" << std::endl;
  for (const auto& entry : world_T_boards_est) {
    int board_id = entry.first;
    const std::vector<double>& pose = entry.second;
    std::cout << "Board " << board_id << " pose: [";
    for (size_t i = 0; i < pose.size(); ++i) {
      std::cout << pose[i];
      if (i != pose.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
}

void PrintBoardPose(const std::unordered_map<int, std::vector<double>>& world_T_boards_est, int board_id) {
  auto it = world_T_boards_est.find(board_id);
  if (it != world_T_boards_est.end()) {
    const std::vector<double>& pose = it->second;
    std::cout << "Board " << board_id << " pose: [";
    for (size_t i = 0; i < pose.size(); ++i) {
      std::cout << pose[i];
      if (i < pose.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  } else {
    std::cout << "Board with id " << board_id << " not found." << std::endl;
  }
}


void loadTransformationToAngleAxis(const char* matrix_str, double transform[6]) {
  std::istringstream iss(matrix_str);
  double data[16];

  for (int i = 0; i < 16; ++i) {
      if (!(iss >> data[i])) {
          throw std::runtime_error("Error: Failed to parse 16 doubles from matrix string.");
      }
  }

  // Construct the Eigen 4x4 matrix
  Eigen::Matrix4d T;
  for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
          T(i, j) = data[i * 4 + j];
      }
  }
  
  // Extract the rotation (upper-left 3x3 block) and translation (first 3 elements of the last column)
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::Vector3d t = T.block<3, 1>(0, 3);
  
  // Convert the rotation matrix to an angle-axis representation
  Eigen::AngleAxisd aa(R);
  Eigen::Vector3d aaVec = aa.axis() * aa.angle();
  
  // Store the rotation (angle-axis) and translation into the 6-element array
  transform[0] = aaVec[0];
  transform[1] = aaVec[1];
  transform[2] = aaVec[2];
  transform[3] = t[0];
  transform[4] = t[1];
  transform[5] = t[2];
}


// ---------------------------------------------------------------------
// Helper to measure rotation/translation difference
// ---------------------------------------------------------------------
static void ComparePose6D(const double* transformGT, const double* transformEst)
{
  auto length3 = [&](const double* v){
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  };

  // We'll do R_err = R_est * R_gt^-1 in angle-axis
  auto invertAA = [&](const double* aaIn, double* aaOut){
    aaOut[0] = -aaIn[0];
    aaOut[1] = -aaIn[1];
    aaOut[2] = -aaIn[2];
  };
  auto axisAngleMultiply = [&](const double* A, const double* B, double* AB){
    // AB = A * B in angle-axis. We'll do a matrix-based approach for simplicity
    auto toMat = [&](const double* ax, double M[9]){
      double th = length3(ax);
      double x=0, y=0, z=0;
      if(th>1e-14){
        x=ax[0]/th; y=ax[1]/th; z=ax[2]/th;
      }
      double c=std::cos(th), s=std::sin(th), nc=1.0-c;
      M[0]=c+x*x*nc;   M[1]=x*y*nc -z*s; M[2]=x*z*nc +y*s;
      M[3]=y*x*nc +z*s;M[4]=c+y*y*nc;    M[5]=y*z*nc -x*s;
      M[6]=z*x*nc -y*s;M[7]=z*y*nc +x*s; M[8]=c+z*z*nc;
    };
    auto fromMat = [&](const double M[9], double* aa){
      double trace= M[0] + M[4] + M[8];
      double cosA = 0.5*(trace - 1.0);
      if(cosA > 1.0)  cosA=1.0;
      if(cosA < -1.0) cosA=-1.0;
      double angle=std::acos(cosA);
      if(angle < 1e-14){
        aa[0]=0; aa[1]=0; aa[2]=0;
        return;
      }
      double sinA=std::sin(angle);
      double denom=2.0*sinA;
      aa[0]=(M[5]-M[7])/denom;
      aa[1]=(M[6]-M[2])/denom;
      aa[2]=(M[1]-M[3])/denom;
      aa[0]*=angle; aa[1]*=angle; aa[2]*=angle;
    };
    double MA[9], MB[9], MAB[9];
    toMat(A, MA);
    toMat(B, MB);
    for(int r=0; r<3; r++){
      for(int c=0; c<3; c++){
        double val=0;
        for(int k=0; k<3; k++){
          val += MA[3*r + k]*MB[3*k + c];
        }
        MAB[3*r + c] = val;
      }
    }
    fromMat(MAB, AB);
  };

  double negGT[3];
  invertAA(transformGT, negGT);
  double Rerr[3];
  axisAngleMultiply(transformEst, negGT, Rerr);
  double rotErrDeg = length3(Rerr)*180.0/M_PI;

  double dx= transformEst[3] - transformGT[3];
  double dy= transformEst[4] - transformGT[4];
  double dz= transformEst[5] - transformGT[5];
  double transErr = std::sqrt(dx*dx + dy*dy + dz*dz);

  std::cout << "  Rotation Error: " << rotErrDeg
            << " deg,  Translation Error: " << transErr << "\n";
}

struct TumEntry {
  double timestamp;
  double tx, ty, tz;
  double qx, qy, qz, qw; // (qw is real part for TUM)
};

// This version inverts each line => final angle-axis
static std::vector<double> parseTumFile(const std::string& filename)
{
  std::ifstream ifs(filename);
  if(!ifs.is_open()){
    throw std::runtime_error("Cannot open TUM file: " + filename);
  }
  std::vector<double> out;
  std::string line;
  while(std::getline(ifs, line)) {
    if(line.empty()) continue;

    std::istringstream iss(line);
    TumEntry e;
    iss >> e.timestamp >> e.tx >> e.ty >> e.tz
    >> e.qx >> e.qy >> e.qz >> e.qw;
    if(iss.fail()) {
      double nan_val = std::numeric_limits<double>::quiet_NaN();
      for (int i = 0; i < 6; i++) {
        out.push_back(nan_val);
      }
      // std::cerr << "Skipping invalid TUM line: " << line << "\n";
      continue;
    }

    Eigen::Quaterniond q(e.qw, e.qx, e.qy, e.qz);
    Eigen::Vector3d t(e.tx, e.ty, e.tz);
    Eigen::Quaterniond q_inv = q.inverse();
    Eigen::Vector3d t_inv = -(q_inv * t);

    // to angle-axis
    Eigen::AngleAxisd aa(q_inv);
    double rx = aa.axis()[0] * aa.angle();
    double ry = aa.axis()[1] * aa.angle();
    double rz = aa.axis()[2] * aa.angle();

    out.push_back(rx);
    out.push_back(ry);
    out.push_back(rz);
    out.push_back(t_inv[0]);
    out.push_back(t_inv[1]);
    out.push_back(t_inv[2]);
  }
  ifs.close();
  return out;
}


void PrintCam0TWorldEst(const std::vector<double>& cam0_T_world_est) {
  std::cout << "=== cam0_T_world_est ===" << std::endl;
  int numFrames = cam0_T_world_est.size() / 6;
  for (int f = 0; f < numFrames; ++f) {
    std::cout << "Frame " << f << ": ";
    for (int i = 0; i < 6; ++i) {
      std::cout << cam0_T_world_est[f*6 + i] << " ";
    }
    std::cout << std::endl;
  }
}

// ---------------------------------------------------------------------
// Multi-camera rig cost functor for reprojection error
// ---------------------------------------------------------------------
struct RigDistortedReprojError {
  RigDistortedReprojError(int cam_idx,
                          double obs_u, double obs_v,
                          double Xb,  double Yb,  double Zb,
                          const KnownIntrinsics& intr)
    : cam_idx_(cam_idx),
      obs_u_(obs_u), obs_v_(obs_v),
      Xb_(Xb), Yb_(Yb), Zb_(Zb),
      fx_(intr.fx), fy_(intr.fy),
      cx_(intr.cx), cy_(intr.cy),
      k1_(intr.k1), k2_(intr.k2), p1_(intr.p1), p2_(intr.p2), k3_(intr.k3)
  {}

  template<typename T>
  bool operator()(const T* const world_T_board,   // 6D for specific board
                  const T* const cam0_T_world,    // 6D
                  const T* const cam0_T_camI,     // 6D for camera>0
                  T* residuals) const
  {
    // 1) board local -> world
    T p_board[3] = { T(Xb_), T(Yb_), T(Zb_) };
    T p_world[3];
    TransformPoint(world_T_board, p_board, p_world);

    // 2) world -> cam0
    T p_cam0[3];
    TransformPoint(cam0_T_world, p_world, p_cam0);

    // 3) If cam_idx_ == 0 => p_cam = p_cam0
    //    If cam_idx_ > 0 => invert (camI->cam0) to get (cam0->camI), then apply
    T p_cam[3];
    if (cam_idx_ == 0) {
      p_cam[0] = p_cam0[0];
      p_cam[1] = p_cam0[1];
      p_cam[2] = p_cam0[2];
    } else {
      T camI_T_cam0[6];
      InvertTransform(cam0_T_camI, camI_T_cam0);
      TransformPoint(camI_T_cam0, p_cam0, p_cam);
    }

    // 4) Project with distortion
    T xp = p_cam[0] / p_cam[2];
    T yp = p_cam[1] / p_cam[2];

    T r2 = xp*xp + yp*yp;
    T r4 = r2*r2;
    T r6 = r2*r4;
    T radial = T(1.0) + T(k1_)*r2 + T(k2_)*r4 + T(k3_)*r6;

    // tangential terms
    T x_tang = T(2.0)*T(p1_)*xp*yp + T(p2_)*(r2 + T(2.0)*xp*xp);
    T y_tang = T(p1_)*(r2 + T(2.0)*yp*yp) + T(2.0)*T(p2_)*xp*yp;

    T x_dist = xp*radial + x_tang;
    T y_dist = yp*radial + y_tang;

    T pred_u = T(fx_)*x_dist + T(cx_);
    T pred_v = T(fy_)*y_dist + T(cy_);

    // 5) residual = observed - predicted
    residuals[0] = T(obs_u_) - pred_u;
    residuals[1] = T(obs_v_) - pred_v;
    return true;
  }

  static ceres::CostFunction* Create(int cam_idx,
                                     double obs_u, double obs_v,
                                     double Xb, double Yb, double Zb,
                                     const KnownIntrinsics& intr)
  {
    // 2 residuals, 3 param blocks each of size 6
    return new ceres::AutoDiffCostFunction<RigDistortedReprojError, 2, 6, 6, 6>(
      new RigDistortedReprojError(cam_idx, obs_u, obs_v, Xb, Yb, Zb, intr));
  }

private:
  int cam_idx_;
  double obs_u_, obs_v_;
  double Xb_, Yb_, Zb_;
  double fx_, fy_, cx_, cy_;
  double k1_, k2_, p1_, p2_, k3_;
};

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);

  // ============================================================
  // Read observations back from JSON files
  // ============================================================
  std::cout << "\n=== Reading observations from JSON files ===\n";
  std::vector<std::string> observation_files = {argv[1], argv[2]}; // Input 1, 2: JSON files with the detected points
  std::vector<Obs> observations = ReadObservationsFromJson(observation_files); 

  std::unordered_map<int, KnownIntrinsics> camera_intrinsics;
  camera_intrinsics[0] = parseCameraIntrinsics(argv[3], argv[4]); // Input 3, 4: GT View Camera intrinsics
  camera_intrinsics[1] = parseCameraIntrinsics(argv[5], argv[6]); // Input 5, 6: User View Camera intrinsics
  auto world_T_boards_est = parseBoardData(argv[7]); // Input 7: Graph poses 
  auto cam0_T_world_est = parseTumFile(argv[8]); //Input 8: List of camera poses [6*frame + i ] from camera to world
  
  int numFrames = cam0_T_world_est.size() / 6;
  double cam0_T_cam1_est[6];
  loadTransformationToAngleAxis(argv[9], cam0_T_cam1_est);// 

  //Inital guess for cam0_T_cam1_est
  cam0_T_cam1_est[3] = 0;
  cam0_T_cam1_est[4] = 0;
  cam0_T_cam1_est[5] = -0.019;

  std::cout << "\n=== Initial guesses for cam_est ===\n";
  for(int i=0; i<6; i++){
    std::cout << cam0_T_cam1_est[i] << " ";
  }
  std::cout << std::endl;

  // Save a copy for printing
  double cam0_T_cam1_init[6];
  for(int i=0; i<6; i++){
    cam0_T_cam1_init[i] = cam0_T_cam1_est[i];
  }

  // Build the ceres problem
  ceres::Problem problem;
  
  // Add parameter blocks for each board
  for (auto& [board_id, board_pose_est] : world_T_boards_est) {
    problem.AddParameterBlock(board_pose_est.data(), 6);
    problem.SetParameterBlockConstant(board_pose_est.data());
  }
  
  // Add parameter blocks for camera poses
  for(int f=0; f<numFrames; f++){
    int start_index = f * 6;
    bool hasNan = false;
    for (int i = 0; i < 6; ++i) {
      if (std::isnan(cam0_T_world_est[start_index + i])) {
        hasNan = true;
        break;
      }
    }
    if (hasNan) {
      continue;
    }
    problem.AddParameterBlock(&cam0_T_world_est[start_index], 6);
  }
  
  // Add parameter block for camera offset
  problem.AddParameterBlock(cam0_T_cam1_est, 6);

  // Add residual blocks
  for(const auto &ob : observations){

    int start_index = ob.frame_idx * 6;
    bool skipObservation = false;
    for (int i = 0; i < 6; ++i) {
      if (std::isnan(cam0_T_world_est[start_index + i])) {
        skipObservation = true;
        break;
      }
    }
    
    if (skipObservation) {
      continue;
    }

    const KnownIntrinsics K = camera_intrinsics[ob.cam_idx];
    
    // Create the cost function
    ceres::CostFunction* costFunc =
      RigDistortedReprojError::Create(ob.cam_idx,
                                      ob.u, ob.v,
                                      ob.Xb, ob.Yb, ob.Zb,
                                      K);
    // Add residual block with the correct board parameter block
    problem.AddResidualBlock(costFunc,
                            nullptr,
                             world_T_boards_est[ob.board_id].data(),
                             &cam0_T_world_est[ob.frame_idx*6],
                             cam0_T_cam1_est);
  }

  // Solve
  ceres::Solver::Options options;
  options.max_num_iterations = 50;
  options.function_tolerance = 1e-7;
  options.minimizer_progress_to_stdout = true;
  options.linear_solver_type = ceres::DENSE_SCHUR;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "\n==== FINAL REPORT ====\n" << summary.FullReport() << "\n";

  std::cout << "Final offset: ";
    for(int i=0; i<6; i++){
        std::cout << cam0_T_cam1_est[i] << " ";
    }

  ceres::Problem::EvaluateOptions eval_options;
  
  return 0;
}