#include "optimizer.h"

void SolveProblem(ceres::Problem& problem,
                  const ceres::Solver::Options& options,
                  ceres::Solver::Summary& summary)
{
  ceres::Solve(options, &problem, &summary);
}
