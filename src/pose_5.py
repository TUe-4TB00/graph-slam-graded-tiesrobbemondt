import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    print("Optimization Result:\n", result)

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None      
    best_landmark = None    
    best_score = float('inf')
    sum_of_marginals = 0
    
    for current_pose_name in pose_options.keys():
        for current_landmark in (1, 2):
            pose_5 = pose_options[current_pose_name]
            
            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_estimate = gtsam.Values(initial_estimate)
            
            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)
            result = optimize(test_graph, test_estimate)
            
            test_graph = add_landmark_measurement(test_graph, result, pose_5, current_landmark)
            result = optimize(test_graph, test_estimate)

            # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
            current_marginals = gtsam.Marginals(test_graph, result)
            
            # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
            score = np.trace(current_marginals.marginalCovariance(L(1))) + np.trace(current_marginals.marginalCovariance(L(2)))
            total_sum = current_marginals.marginalCovariance(L(1)).sum() + current_marginals.marginalCovariance(L(2)).sum()
            
            if score < best_score:
                best_score = score
                best_pose = current_pose_name
                best_landmark = current_landmark
                sum_of_marginals = total_sum

    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None      
    best_landmark = None    
    sum_of_errors = float('inf')
    
    for current_pose_name in pose_options.keys():
        for current_landmark in (1, 2):
            pose_5 = pose_options[current_pose_name]
            
            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_estimate = gtsam.Values(initial_estimate)
            
            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)
            result = optimize(test_graph, test_estimate)
            
            test_graph = add_landmark_measurement(test_graph, result, pose_5, current_landmark)
            result = optimize(test_graph, test_estimate)
            
            current_marginals = gtsam.Marginals(test_graph, result)

            # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
            list_of_errors = [
                np.trace(current_marginals.marginalCovariance(X(1))),
                np.trace(current_marginals.marginalCovariance(X(2))),
                np.trace(current_marginals.marginalCovariance(X(3)))
            ]
            
            # TODO: compute the sum of the errors and return it along with the best pose and landmark
            current_sum_of_errors = sum(list_of_errors)
            
            if current_sum_of_errors < sum_of_errors:
                sum_of_errors = current_sum_of_errors
                best_pose = current_pose_name
                best_landmark = current_landmark

    return best_pose, best_landmark, sum_of_errors