#include <iostream>
#include <cmath>
#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include "optimizer.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath> // For std::isnan



#include <iomanip>

// Callback to monitor parameter block values
class DebugParameterBlockCallback : public ceres::IterationCallback {
public:
    DebugParameterBlockCallback(double* parameter_block, int size)
        : parameter_block_(parameter_block), size_(size) {}

    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
        std::cout << "Parameter Block Values (Size " << size_ << "):\n";
        for (int i = 0; i < size_; ++i) {
            std::cout << "  [" << i << "]: " << std::setprecision(10) << parameter_block_[i] << "\n";
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    double* parameter_block_;
    int size_;
};

using json = nlohmann::json;

// Define types for better readability
using Points3D = std::vector<std::vector<double>>;
using Points2D = std::vector<std::vector<double>>;

// Struct to hold board data
struct BoardData {
    std::string board_id;
    Points3D points_3d;
    Points2D points_2d;
};

// Struct to hold frame data
struct FrameData {
    int frame_idx;
    std::vector<BoardData> boards;
};

// Function to parse JSON file and organize data by frames
std::vector<FrameData> parse_json_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename);
    }

    // Parse JSON
    json j;
    file >> j;

    // Vector to store frames in order
    std::vector<FrameData> frames;

    for (const auto& item : j) {
        int frame_idx = item["f"];  // "f" for frame_idx
        std::string board_id = item["b_id"];  // "b_id" for board_id
        Points3D board_3d_points = item["3d"].get<Points3D>();  // "3d" for 3D points
        Points2D board_2d_points = item["2d"].get<Points2D>();  // "2d" for 2D points

        // Check if the frame already exists in the vector
        auto it = std::find_if(frames.begin(), frames.end(), [frame_idx](const FrameData& f) {
            return f.frame_idx == frame_idx;
        });

        if (it == frames.end()) {
            // Frame not found, create a new one
            frames.push_back({frame_idx, {}});
            it = std::prev(frames.end());
        }

        // Add the board data to the frame
        it->boards.push_back({board_id, board_3d_points, board_2d_points});
    }

    return frames;
}

// Helper to apply a 6D transform (angle-axis + translation)
static void TransformPoint(const double* pose_6,
                           const double* in,
                           double* out)
{
  double p[3];
  ceres::AngleAxisRotatePoint(pose_6, in, p);
  out[0] = p[0] + pose_6[3];
  out[1] = p[1] + pose_6[4];
  out[2] = p[2] + pose_6[5];
}

struct TumData {
    double timestamp;  
    double tx, ty, tz;
    double qx, qy, qz, qw;
};



// Function to parse node data string (node_id:pose)
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



std::vector<TumData> parseTumFile(const std::string& filepath) {
    std::ifstream tum_file(filepath);
    if (!tum_file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filepath);
    }

    std::vector<TumData> tum_entries;
    std::string line;
    while (std::getline(tum_file, line)) {
        std::istringstream line_stream(line);
        TumData entry;

        line_stream >> entry.timestamp >> entry.tx >> entry.ty >> entry.tz
                    >> entry.qx >> entry.qy >> entry.qz >> entry.qw;

        if (line_stream.fail()) {
            std::cerr << "Warning: Invalid line in TUM file: " << line << "\n";
            entry.tx = std::numeric_limits<double>::quiet_NaN();
            entry.ty = std::numeric_limits<double>::quiet_NaN();
            entry.tz = std::numeric_limits<double>::quiet_NaN();
            entry.qx = std::numeric_limits<double>::quiet_NaN();
            entry.qy = std::numeric_limits<double>::quiet_NaN();
            entry.qz = std::numeric_limits<double>::quiet_NaN();
            entry.qw = std::numeric_limits<double>::quiet_NaN();
        } else {
            // Convert quaternion to angle-axis
            Eigen::Quaterniond quaternion(entry.qw, entry.qx, entry.qy, entry.qz);
            Eigen::Vector3d translation(entry.tx, entry.ty, entry.tz);

            // Invert the quaternion
            Eigen::Quaterniond inverse_quaternion = quaternion.inverse();

            // Compute the new translation: -R^T * t
            Eigen::Vector3d inverse_translation = -(inverse_quaternion * translation);

            // Update the entry with the inverse pose
            entry.tx = inverse_translation[0];
            entry.ty = inverse_translation[1];
            entry.tz = inverse_translation[2];
            entry.qx = inverse_quaternion.x();
            entry.qy = inverse_quaternion.y();
            entry.qz = inverse_quaternion.z();
            entry.qw = inverse_quaternion.w();

            // Convert quaternion to angle-axis
            Eigen::AngleAxisd angle_axis(inverse_quaternion);
            entry.qx = angle_axis.axis()[0] * angle_axis.angle(); // ax
            entry.qy = angle_axis.axis()[1] * angle_axis.angle(); // ay
            entry.qz = angle_axis.axis()[2] * angle_axis.angle(); // az
            entry.qw = 0.0; // No longer needed

            // Eigen::Quaterniond quaternion(entry.qw, entry.qx, entry.qy, entry.qz);
            // Eigen::AngleAxisd angle_axis(quaternion);

            // entry.qx = angle_axis.axis()[0]; //* angle_axis.angle(); // ax
            // entry.qy = angle_axis.axis()[1]; //* angle_axis.angle(); // ay
            // entry.qz = angle_axis.axis()[2]; //* angle_axis.angle(); // az
            // entry.qw = 0.0; // No longer needed
        }
        tum_entries.push_back(entry);
    }
    tum_file.close();
    return tum_entries;
}

std::unordered_map<int, size_t> BoardIdToIndexMap(const std::unordered_map<int, std::vector<double>>& board_graph_poses) {
    std::unordered_map<int, size_t> board_id_to_index;
    size_t board_index = 0;

    for (const auto& [board_id, pose] : board_graph_poses) {
        board_id_to_index[board_id] = board_index++;
    }

    return board_id_to_index;
}


void printTumData(const std::vector<TumData>& tum_data, const std::string& label) {
    std::cout << "==== " << label << " ====\n";
    for (const auto& entry : tum_data) {
        std::cout << "Timestamp: " << entry.timestamp
                  << ", Translation: (" << entry.tx << ", " << entry.ty << ", " << entry.tz << ")"
                  << ", Angle-Axis: (" << entry.qx << ", " << entry.qy << ", " << entry.qz << ", " << entry.qw << ")\n";
    }
    std::cout << "=========================\n";
}


void printJsondData(const std::vector<FrameData>& frames) {
    for (const auto& frame : frames) {
        std::cout << "Frame Index: " << frame.frame_idx << "\n";

        for (const auto& board : frame.boards) {
            std::cout << "  Board ID: " << board.board_id << "\n";

            std::cout << "    3D Points:\n";
            for (const auto& point : board.points_3d) {
                std::cout << "      (" << point[0] << ", " << point[1] << ", " << point[2] << ")\n";
            }

            std::cout << "    2D Points:\n";
            for (const auto& point : board.points_2d) {
                std::cout << "      (" << point[0] << ", " << point[1] << ")\n";
            }
        }
        std::cout << "\n";
    }
}


void printBoardGraphPoses(const std::unordered_map<int, std::vector<double>>& board_graph_poses) {
    for (const auto& [node_id, pose] : board_graph_poses) {
        std::cout << "Node " << node_id << ": [";
        for (size_t i = 0; i < pose.size(); ++i) {
            std::cout << pose[i];
            if (i < pose.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

void PrintOptimizedCameraPoses(const std::vector<double>& frame_poses, size_t num_frames) {
    std::cout << "Optimized Camera Poses:\n";
    for (size_t i = 0; i < num_frames; ++i) {
        std::cout << "Camera " << i << ": [";
        for (size_t j = 0; j < 6; ++j) {
            std::cout << frame_poses[i * 6 + j] << (j < 5 ? ", " : "");
        }
        std::cout << "]\n";
    }
}

void PrintOptimizedBoardPoses(const std::vector<double>& board_poses, size_t num_boards) {
    std::cout << "Optimized Board Poses:\n";
    for (size_t i = 0; i < num_boards; ++i) {
        std::cout << "Board " << i << ": [";
        for (size_t j = 0; j < 6; ++j) {
            std::cout << board_poses[i * 6 + j] << (j < 5 ? ", " : "");
        }
        std::cout << "]\n";
    }
}

Eigen::Quaterniond AngleAxisToQuaternion(const double* angle_axis) {
    Eigen::AngleAxisd rotation(Eigen::Vector3d(angle_axis[0], angle_axis[1], angle_axis[2]).norm(),
                               Eigen::Vector3d(angle_axis[0], angle_axis[1], angle_axis[2]).normalized());
    return Eigen::Quaterniond(rotation);
}


void SaveOptimizedPoses(const std::vector<double>& frame_poses, size_t num_frames, const std::string& base_filename) {
    // Insert "global_optimized_" before "final_pose_scenario" in base_filename
    std::string output_filename = base_filename;
    // size_t pos = output_filename.find("final_pose_scenario");
    // if (pos != std::string::npos) {
    //     output_filename.insert(pos, "global_optimized_");
    // } else {
    //     throw std::runtime_error("Error: 'final_pose_scenario' not found in the provided filename.");
    // }

    // Open file for writing
    std::ofstream file(output_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + output_filename);
    }
    

    for (size_t i = 0; i < num_frames; ++i) {
        const double* pose = &frame_poses[i * 6];

        // Convert angle-axis to quaternion
        Eigen::Quaterniond quaternion = AngleAxisToQuaternion(pose);

        // Compute inverse quaternion (conjugate)
        Eigen::Quaterniond inverse_quaternion = quaternion.conjugate();

        // Compute inverse translation: -R^T * t
        Eigen::Vector3d translation(pose[3], pose[4], pose[5]);
        Eigen::Vector3d inverse_translation = -(inverse_quaternion * translation);

        // Write timestamp
        file << std::fixed << std::setprecision(1) << static_cast<double>(i) << " ";

        // Write inverse position and quaternion components with high precision
        file << std::fixed << std::setprecision(15)
             << inverse_translation[0] << " " << inverse_translation[1] << " " << inverse_translation[2] << " "  // Inverse Translation
             << inverse_quaternion.x() << " " << inverse_quaternion.y() << " "
             << inverse_quaternion.z() << " " << inverse_quaternion.w() << "\n";  // Inverse Quaternion
    }

    file.close();
    std::cout << "Optimized poses saved to " << output_filename << "\n";
}


// A small struct for storing 2D detections
struct Detection2D {
  double u;       // measured x
  double v;       // measured y
  int board_id;   // 1-based index
  double p_local[3]; // local 3D coords
};

int main(int argc, char** argv)
{

    if (argc != 10) { // Expect three arguments
        std::cerr << "Usage: " << argv[0] << " <camera_matrix_str> <dist_coeffs_str> <camera_trajectory_file_path> <board_graph> <graph_camera_trajectory> <detected_points_camera> <detected_points_graph> <camera_matrix_close_str> <dist_coeffs_close_str> <>\n";
        return 1;
    }

    
    // --------------------------------------------
    // 1) Parse input data
    // --------------------------------------------

    KnownIntrinsics known_intr = parseCameraIntrinsics(argv[1], argv[2]);  // Input 1: Camera Intrinsics
    auto camera_trajectory = parseTumFile(argv[3]); // Input 2: Camera Trajectory (timestamp, tx, ty, tz, qx, qy, qz, qw)
    auto board_graph_poses = parseBoardData(argv[4]);  // Input 3: Graph poses 
    auto graph_cam_traj = parseTumFile(argv[5]);   // Input 4: Graph Close Camera Trajectory (timestamp, tx, ty, tz, qx, qy, qz, qw)
    auto detected_points_camera = parse_json_file(argv[6]); // Input 5: Detected Points Camera for Camera Trajectory
    auto detected_points_graph = parse_json_file(argv[7]); // Input 6: Detected Points for Graph Close Camera Trajectory
    KnownIntrinsics known_intr_close = parseCameraIntrinsics(argv[8], argv[9]); // Input 7: Camera Intrinsics Close
    

    // --------------------------------------------
    // 2) Build a Problem
    // --------------------------------------------
    // 2.1) Initialize Graph and Frame Poses  
    // --------------------------------------------
    google::InitGoogleLogging(argv[0]);

    ceres::Problem problem;
    
    size_t num_frames = camera_trajectory.size() + graph_cam_traj.size();
    size_t num_boards = board_graph_poses.size();

    std::vector<double> frame_poses(num_frames * 6, 0.0);
    std::vector<double> board_poses(num_boards * 6, 0.0);
    auto board_id_to_index = BoardIdToIndexMap(board_graph_poses);
    
    for (size_t i = 0; i < camera_trajectory.size(); i++) { 
        frame_poses[i * 6 + 0] = camera_trajectory[i].qx; // Angle-axis x
        frame_poses[i * 6 + 1] = camera_trajectory[i].qy; // Angle-axis y
        frame_poses[i * 6 + 2] = camera_trajectory[i].qz; // Angle-axis z
        frame_poses[i * 6 + 3] = camera_trajectory[i].tx; 
        frame_poses[i * 6 + 4] = camera_trajectory[i].ty; 
        frame_poses[i * 6 + 5] = camera_trajectory[i].tz;
    }

    size_t N = camera_trajectory.size();

    for (size_t i = 0; i < graph_cam_traj.size(); i++) {
        frame_poses[( N + i) * 6 + 0] = graph_cam_traj[i].qx;
        frame_poses[(N + i) * 6 + 1] = graph_cam_traj[i].qy;
        frame_poses[(N + i) * 6 + 2] = graph_cam_traj[i].qz;
        frame_poses[(N + i) * 6 + 3] = graph_cam_traj[i].tx;
        frame_poses[(N + i) * 6 + 4] = graph_cam_traj[i].ty;
        frame_poses[(N + i) * 6 + 5] = graph_cam_traj[i].tz;
    }


    for (size_t i = 0; i < num_frames; i++) {
        if (std::isnan(frame_poses[i * 6 + 0]) || std::isnan(frame_poses[i * 6 + 1]) ||
            std::isnan(frame_poses[i * 6 + 2]) || std::isnan(frame_poses[i * 6 + 3]) ||
            std::isnan(frame_poses[i * 6 + 4]) || std::isnan(frame_poses[i * 6 + 5])) {
            continue;
        }
            problem.AddParameterBlock(&frame_poses[i * 6], 6);
    }

    std::cout << "Number of frames: " << num_frames << "\n";
    for (size_t i = 0; i < num_boards; i++) {
            problem.AddParameterBlock(&board_poses[i * 6], 6);
    }
    std::cout << "Number of boards: " << num_boards << "\n";

    for (const auto& [board_id, pose] : board_graph_poses) {
        size_t idx = board_id_to_index[board_id] * 6; // Map board_id to a contiguous index
        board_poses[idx + 0] = pose[0];
        board_poses[idx + 1] = pose[1];
        board_poses[idx + 2] = pose[2];
        board_poses[idx + 3] = pose[3];
        board_poses[idx + 4] = pose[4];
        board_poses[idx + 5] = pose[5]; 
        if (pose[0] == 0 && pose[1] == 0 && pose[2] == 0 && pose[3] == 0 && pose[4] == 0 && pose[5] == 0) {
            problem.SetParameterBlockConstant(&board_poses[idx]);
        }
    }



    // --------------------------------------------
    // 2.2) Add residuals
    // --------------------------------------------

    //   std::vector<std::vector<Detection2D>> all_detections(num_frames);

    for (size_t frame_idx = 0; frame_idx < detected_points_camera.size(); frame_idx++) {
        for (const auto& board : detected_points_camera[frame_idx].boards) {
            std::string board_id_str = board.board_id;
            size_t underscore_pos = board_id_str.find('_');
            if (underscore_pos != std::string::npos) {
                board_id_str = board_id_str.substr(0, underscore_pos);
            }
            int board_id = std::stoi(board_id_str);  
            if (board_id_to_index.find(board_id) == board_id_to_index.end()) {
                throw std::runtime_error("Error: board_id not found in board_id_to_index map.");
            }
            size_t board_index = board_id_to_index[board_id];
            if (board.points_2d.size() == 0) {
            continue;
            }
            for (size_t point_idx = 0; point_idx < board.points_2d.size(); point_idx++) {
                const auto& point_2d = board.points_2d[point_idx];
                const auto& point_3d = board.points_3d[point_idx];

                if (std::isnan(point_2d[0]) || std::isnan(point_2d[1]) ||
                    std::isnan(point_3d[0]) || std::isnan(point_3d[1]) || std::isnan(point_3d[2])) {
                    throw std::runtime_error("Error: NaN detected in input points.");
                }

                Detection2D detection;
                detection.u = point_2d[0];
                detection.v = point_2d[1];
                detection.board_id = board_index;
                detection.p_local[0] = point_3d[0];
                detection.p_local[1] = point_3d[1];
                detection.p_local[2] = point_3d[2];
                //   all_detections[frame_idx].push_back(detection);
        
                // Add reprojection residual
                auto* cost_function = DistortedReprojError::Create(
                    detection.u, detection.v, detection.p_local, known_intr);
                problem.AddResidualBlock(cost_function,
                                            nullptr,
                                        &frame_poses[detected_points_camera[frame_idx].frame_idx * 6],
                                        &board_poses[board_index * 6]);
                // problem.AddResidualBlock(cost_function,
                //                             nullptr,
                //                         &frame_poses[frame_idx * 6],
                //                         &board_poses[board_index * 6]);
                // // Calculate residual for printing
                // double residual[2];
                // double camera_params[6], board_params[6];
                // std::copy(&frame_poses[frame_idx * 6], &frame_poses[frame_idx * 6 + 6], camera_params);
                // std::copy(&board_poses[board_index * 6], &board_poses[board_index * 6 + 6], board_params);

                // std::cout<< "--------------------------------------------\n";

                // DistortedReprojError error(detection.u, detection.v, detection.p_local, known_intr);
                // error(camera_params, board_params, residual);
                
                // // Print reprojection error
                // std::cout << "Frame: " << frame_idx << ", Board: " << board_id
                //         << ", Point: " << point_idx << "\n"
                //         <<" 3d: [" << point_3d[0] << ", " << point_3d[1] << ", " << point_3d[2] << "]\n"
                //         << ", 2d: [" << point_2d[0] << ", " << point_2d[1] << "]\n"
                //         << "Frame Pose: [" << camera_params[0] << ", " << camera_params[1] << ", " << camera_params[2] << ", " << camera_params[3] << ", " << camera_params[4] << ", " << camera_params[5] << "]\n"
                //         << "Board Pose: [" << board_params[0] << ", " << board_params[1] << ", " << board_params[2] << ", " << board_params[3] << ", " << board_params[4] << ", " << board_params[5] << "]\n"
                //         << ", Residual: [" << residual[0] << ", " << residual[1] << "]\n";
 
            }
        }
        // break;
    }
    
    for (size_t frame_idx = 0; frame_idx < detected_points_graph.size();frame_idx++) {
        for (const auto& board : detected_points_graph[frame_idx].boards) {
            std::string board_id_str = board.board_id;
            size_t underscore_pos = board_id_str.find('_');
            if (underscore_pos != std::string::npos) {
                board_id_str = board_id_str.substr(0, underscore_pos);
            }
            int board_id = std::stoi(board_id_str);
            if (board_id_to_index.find(board_id) == board_id_to_index.end()) {
                throw std::runtime_error("Error: board_id not found in board_id_to_index map.");
            }
            size_t board_index = board_id_to_index[board_id];
            for (size_t point_idx = 0; point_idx < board.points_2d.size(); point_idx++) {
                const auto& point_2d = board.points_2d[point_idx];
                const auto& point_3d = board.points_3d[point_idx];

                Detection2D detection;
                detection.u = point_2d[0];
                detection.v = point_2d[1];
                detection.board_id = board_index;
                detection.p_local[0] = point_3d[0];
                detection.p_local[1] = point_3d[1];
                detection.p_local[2] = point_3d[2];
                //   all_detections[frame_idx].push_back(detection);

                // Add reprojection residual
                auto* cost_function = DistortedReprojError::Create(
                    detection.u, detection.v, detection.p_local, known_intr_close);
                problem.AddResidualBlock(cost_function,
                                            nullptr,
                                            &frame_poses[(N + detected_points_graph[frame_idx].frame_idx) * 6],
                                            &board_poses[board_index * 6]);
            }
        }
        
    }
   

    // --------------------------------------------
    // 3) Solve the problem
    // --------------------------------------------

    ceres::Solver::Options options;
    options.max_num_iterations = 30;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ceres::Problem::EvaluateOptions eval_options;
    // Get all residuals
    std::vector<double> residuals;
    problem.Evaluate(eval_options, nullptr, &residuals, nullptr, nullptr);

    std::cout << summary.FullReport() << "\n";
    SaveOptimizedPoses(frame_poses, N, argv[3]);



    // Print residuals
    // std::cout << "Residuals:\n";
    // std::cout << residuals.size() << "\n";
    // for (size_t i = 0; i < 10; ++i) {
    //     std::cout << "  Residual " << i << ": " << residuals[i] << "\n";
    // }

    // PrintOptimizedCameraPoses(frame_poses, num_frames);
    // PrintOptimizedBoardPoses(board_poses, num_boards);

    return 0;
}
