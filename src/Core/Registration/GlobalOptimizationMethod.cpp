// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncle@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "GlobalOptimizationMethod.h"

//#include <iostream>
//#include <fstream>
//#include <json/json.h>
#include <Core/Utility/Console.h>
#include <Core/Utility/Eigen.h>
#include <Core/Utility/Timer.h>
#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>
#include <Core/Registration/GlobalOptimizationOption.h>

namespace three{

std::shared_ptr<PoseGraph> 
		GraphOptimizationMethodGaussNewton::OptimizePoseGraph(
		const PoseGraph &pose_graph,
		const GlobalOptimizationOption &option) const
{
	//OptimizationStatusGaussNewton status;

	//status.Init(pose_graph, option);
	//status.PrintInit();
	//status.Checkb();

	//status.timer_overall_.Start();
	//for (status.iter_ = 0; !status.stop_; status.iter_++) {
	//	status.timer_iter_.Start();

	//	status.SolveLinearSystemInClass();

	//	if (!status.CheckRelative()) {
	//		status.UpdatePoseGraphInClass();
	//		status.CheckRelativeResidual();
	//		status.UpdateCurrentInClass();
	//		status.ComputeLinearSystemInClass();
	//		status.Checkb();
	//	}
	//	if (!status.stop_) {
	//		status.timer_iter_.Stop();
	//		status.PrintStatus();
	//	}
	//	status.CheckAbsoluteResidual();
	//	status.CheckMaxIteration();
	//}
	//status.timer_overall_.Stop();
	//status.PrintOverallTime();

	//std::shared_ptr<PoseGraph> pose_graph_refined_pruned =
	//	status.UpdatePruneInClass();
	//return pose_graph_refined_pruned;

	std::shared_ptr<PoseGraph> pose_graph_refined_pruned;
	return pose_graph_refined_pruned;
}

std::shared_ptr<PoseGraph> 
		GraphOptimizationLevenbergMethodMarquardt::OptimizePoseGraph(
		const PoseGraph &pose_graph,
		const GlobalOptimizationOption &option) const
{
	//OptimizationStatusLevenbergMarquardt status;

	//status.Init(pose_graph, option);
	//status.InitLM();
	//status.PrintInit();
	//status.Checkb();

	//status.timer_overall_.Start();
	//for (status.iter_ = 0; !status.stop_; status.iter_++) {
	//	status.timer_iter_.Start();
	//	status.lm_count_ = 0;
	//	do 
	//	{
	//		status.SolveLinearSystemInClass();
	//		if (!status.CheckRelative()) {
	//			status.UpdatePoseGraphInClass();
	//			status.ComputeRho();
	//			if (status.rho_ > 0) {
	//				status.CheckRelativeResidual();
	//				status.ComputeGain();
	//				status.UpdateCurrentInClass();
	//				status.ComputeLinearSystemInClass();
	//				status.Checkb();
	//			} else {
	//				status.ResetGain();
	//			}
	//		}
	//		status.lm_count_++;
	//		status.CheckMaxIterationInnerLoop();
	//	} while (!((status.rho_ > 0) || status.stop_));
	//	if (!status.stop_) {
	//		status.timer_iter_.Stop();
	//		status.PrintStatus();
	//	}
	//	status.CheckAbsoluteResidual();
	//	status.CheckMaxIteration();
	//}	
	//status.timer_overall_.Stop();
	//status.PrintOverallTime();

	//std::shared_ptr<PoseGraph> pose_graph_refined_pruned =
	//	status.UpdatePruneInClass();
	//return pose_graph_refined_pruned;

	std::shared_ptr<PoseGraph> pose_graph_refined_pruned;
	return pose_graph_refined_pruned;
}

}	// namespace three
