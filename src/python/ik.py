# ik.py

# IMPORTS
import numpy as np
import time
import cma
import fk
import copy
HAS_CMA = True

def forward_kinematics_vector(pressures, params, segments):
    pA, pB, pC, pD = pressures
    
    channel_area = fk.channel_area(params['channel_radius'])
    centroid_dist = fk.centroid_distance(params['channel_radius'], params['septum_thickness'])
    
    M_ac = fk.directional_moment(pA, pC, channel_area, centroid_dist)
    M_bd = fk.directional_moment(pB, pD, channel_area, centroid_dist)
    M_res = fk.resultant_moment(M_ac, M_bd)
    phi = fk.bending_plane_ang(M_ac, M_bd)

    epsilon_pre = params['epsilon_pre']
    
    # Compute curvature and arc angle for each segment
    for seg in segments:
        seg.curvature = fk.curvature(M_res, seg.EI)
        seg.theta = fk.arc_angle(seg.curvature, fk.prestrained_length(seg.length, epsilon_pre))
    
    # Chain transformation matrices
    T = np.eye(4)
    
    for seg in segments:
        kappa = seg.curvature
        theta = seg.theta
        
        if abs(kappa) < 1e-6:
            # Straight segment
            T_seg = np.eye(4)
            T_seg[2, 3] = seg.length * (1 + epsilon_pre)
        else:
            # Curved segment (PCC model)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            T_seg = np.array([
                [cos_phi**2 + sin_phi**2 * cos_theta,
                -sin_phi * cos_phi * (1 - cos_theta),
                cos_phi * sin_theta,
                cos_phi * (1 - cos_theta) / kappa],

                [-sin_phi * cos_phi * (1 - cos_theta),
                sin_phi**2 + cos_phi**2 * cos_theta,
                sin_phi * sin_theta,
                sin_phi * (1 - cos_theta) / kappa],

                [-cos_phi * sin_theta,
                -sin_phi * sin_theta,
                cos_theta,
                sin_theta / kappa],

                [0, 0, 0, 1]
            ])
        
        T = T @ T_seg
    
    return T[:3, 3]


def compute_jacobian(pressures, params, segments, delta=500.0):
    J = np.zeros((3, 4))
    
    for i in range(4):
        p_plus = pressures.copy()
        p_minus = pressures.copy()
        
        # Perturb safely
        p_plus[i] = min(pressures[i] + delta, 100e3)
        p_minus[i] = max(pressures[i] - delta, 0)
        
        pos_plus = forward_kinematics_vector(p_plus, params, copy.deepcopy(segments))
        pos_minus = forward_kinematics_vector(p_minus, params, copy.deepcopy(segments))
        
        actual_delta = p_plus[i] - p_minus[i]
        if actual_delta > 0:
            J[:, i] = (pos_plus - pos_minus) / actual_delta
    
    return J

def gradient_descent_ik(target_pos, params, segments, pressure_bounds,
                        max_iterations=200, tolerance=1e-5, 
                        verbose=True):
    p_min = pressure_bounds[:, 0]
    p_max = pressure_bounds[:, 1]

    # Initialize at center of pressure range
    current_p = (p_min + p_max) / 2
    
    learning_rate = 0.1 * (p_max - p_min).mean()
    best_p = current_p.copy()
    best_error = float('inf')
    error_history = []
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Current position and error
        current_pos = forward_kinematics_vector(current_p, params, copy.deepcopy(segments))
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)
        error_history.append(error_norm)
        
        # Track best solution
        if error_norm < best_error:
            best_error = error_norm
            best_p = current_p.copy()
        
        # Check convergence
        if error_norm < tolerance:
            if verbose:
                print(f"  Converged at iteration {iteration}, error: {error_norm*1000:.4f} mm")
            break
        
        # Compute Jacobian and gradient direction
        J = compute_jacobian(current_p, params, segments)
        
        # Gradient descent step: move in direction J^T @ error
        # This is the direction that reduces ||error||^2 fastest
        step = J.T @ error
        
        # Normalize and scale by learning rate
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            if verbose:
                print(f"  Zero gradient at iteration {iteration}")
            break
        
        step = step / step_norm * learning_rate
        
        # Line search to find good step size
        best_alpha = 0.0
        best_trial_error = error_norm
        
        for alpha in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            trial_p = current_p + alpha * step
            trial_p = np.clip(trial_p, p_min, p_max)
            
            trial_pos = forward_kinematics_vector(trial_p, params, copy.deepcopy(segments))
            trial_error = np.linalg.norm(target_pos - trial_pos)
            
            if trial_error < best_trial_error:
                best_trial_error = trial_error
                best_alpha = alpha
        
        if best_alpha == 0:
            # No improvement found - reduce learning rate
            learning_rate *= 0.5
            if learning_rate < 100:
                if verbose:
                    print(f"  Learning rate too small at iteration {iteration}")
                break
        else:
            # Apply the best step
            current_p = current_p + best_alpha * step
            current_p = np.clip(current_p, p_min, p_max)
        
        if verbose and iteration % 20 == 0:
            print(f"  Iter {iteration}: error = {error_norm*1000:.4f} mm, "
                  f"lr = {learning_rate/1e3:.2f} kPa, alpha = {best_alpha:.2f}")
    
    elapsed_time = time.time() - start_time
    
    info_dict = {
        'iterations': iteration,
        'time': elapsed_time,
        'converged': best_error < tolerance,
        'error_history': error_history,
        'final_learning_rate': learning_rate
    }
    
    return best_p, best_error, info_dict


def levenberg_marquardt_ik(target_pos, params, segments, pressure_bounds,
                           max_iterations=100, tolerance=1e-6,
                           lambda_init=1e-4, verbose=True):
    """
    Levenberg-Marquardt IK solver with properly scaled damping.
    
    The key fix: damping is scaled relative to the Jacobian magnitude.
    """
    p_min = pressure_bounds[:, 0]
    p_max = pressure_bounds[:, 1]
    
    # Initialize at center of pressure range
    current_p = (p_min + p_max) / 2
    
    lambda_damping = lambda_init
    best_p = current_p.copy()
    best_error = float('inf')
    error_history = []
    lambda_history = []
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Current position and error
        current_pos = forward_kinematics_vector(current_p, params, segments)
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)
        
        error_history.append(error_norm)
        lambda_history.append(lambda_damping)
        
        # Track best solution
        if error_norm < best_error:
            best_error = error_norm
            best_p = current_p.copy()
        
        # Check convergence
        if error_norm < tolerance:
            if verbose:
                print(f"  LM converged at iteration {iteration}, error: {error_norm*1000:.4f} mm")
            break
        
        # Compute Jacobian
        J = compute_jacobian(current_p, params, segments)
        
        # Form normal equations: J^T J
        JTJ = J.T @ J
        JTe = J.T @ error
        
        # CRITICAL FIX: Scale damping relative to JTJ magnitude
        scale = np.trace(JTJ) / 4 + 1e-20
        lambda_scaled = lambda_damping * scale
        
        # Damped normal equations
        damped_matrix = JTJ + lambda_scaled * np.eye(4)
        
        try:
            delta_p = np.linalg.solve(damped_matrix, JTe)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  Singular matrix at iteration {iteration}")
            lambda_damping *= 10
            continue
        
        # Limit step size
        max_step = 0.2 * (p_max - p_min).mean()
        if np.abs(delta_p).max() > max_step:
            delta_p = delta_p * (max_step / np.abs(delta_p).max())
        
        # Try the step
        new_p = current_p + delta_p
        new_p = np.clip(new_p, p_min, p_max)
        
        new_pos = forward_kinematics_vector(new_p, params, copy.deepcopy(segments))
        new_error = np.linalg.norm(target_pos - new_pos)
        
        # Adapt damping
        if new_error < error_norm:
            current_p = new_p
            lambda_damping = max(lambda_damping / 2, 1e-10)
        else:
            lambda_damping = min(lambda_damping * 3, 1e3)
        
        if verbose and iteration % 10 == 0:
            print(f"  Iter {iteration}: error = {error_norm*1000:.4f} mm, "
                  f"lambda = {lambda_damping:.2e}")
    
    elapsed_time = time.time() - start_time
    
    info_dict = {
        'iterations': iteration,
        'time': elapsed_time,
        'converged': best_error < tolerance,
        'error_history': error_history,
        'lambda_history': lambda_history
    }
    
    return best_p, best_error, info_dict


def hybrid_cmaes_gradient_solver(target_pos, params, segments, pressure_bounds,
                                  cma_iterations=50, cma_popsize=20,
                                  gd_tolerance=1e-6, verbose=True):
    """
    Two-phase hybrid IK solver: CMA-ES global search + gradient descent refinement.
    """
    p_min = pressure_bounds[:, 0]
    p_max = pressure_bounds[:, 1]
    
    def cost_function(pressures):
        try:
            pos = forward_kinematics_vector(pressures, params, segments)
            return np.linalg.norm(pos - target_pos)
        except:
            return 1e10
    
    if verbose:
        print("\n" + "=" * 60)
        print("HYBRID CMA-ES + GRADIENT DESCENT IK SOLVER")
        print("=" * 60)
    
    total_start = time.time()
    
    # Phase 1: CMA-ES
    if HAS_CMA and cma_iterations > 0:
        if verbose:
            print("\n[PHASE 1] CMA-ES Global Search")
            print("-" * 60)
        
        x0 = (p_min + p_max) / 2
        sigma0 = 0.25 * (p_max - p_min).mean()
        
        opts = {
            'bounds': [p_min.tolist(), p_max.tolist()],
            'maxiter': cma_iterations,
            'popsize': cma_popsize,
            'verb_disp': 0,
            'verbose': -9
        }
        
        cma_start = time.time()
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        cma_costs = []
        while not es.stop():
            solutions = es.ask()
            costs = [cost_function(x) for x in solutions]
            es.tell(solutions, costs)
            cma_costs.append(min(costs))
        
        cma_time = time.time() - cma_start
        
        best_cma = es.result.xbest
        cost_cma = es.result.fbest
        cma_iters = es.result.iterations
        
        if verbose:
            print(f"  CMA-ES final error: {cost_cma*1000:.3f} mm")
            print(f"  CMA-ES iterations: {cma_iters}")
            print(f"  CMA-ES time: {cma_time:.2f}s")
    else:
        best_cma = (p_min + p_max) / 2
        cost_cma = cost_function(best_cma)
        cma_time = 0
        cma_iters = 0
        cma_costs = []
    
    # Phase 2: Gradient descent from CMA-ES result
    if verbose:
        print(f"\n[PHASE 2] Gradient Descent Refinement")
        print("-" * 60)
    
    gd_start = time.time()
    
    current_p = best_cma.copy()
    learning_rate = 0.1 * (p_max - p_min).mean()
    best_p = current_p.copy()
    best_error = float('inf')
    error_history = []
    
    for iteration in range(100):
        current_pos = forward_kinematics_vector(current_p, params, copy.deepcopy(segments))
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)
        error_history.append(error_norm)
        
        if error_norm < best_error:
            best_error = error_norm
            best_p = current_p.copy()
        
        if error_norm < gd_tolerance:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break
        
        J = compute_jacobian(current_p, params, segments)
        step = J.T @ error
        
        step_norm = np.linalg.norm(step)
        if step_norm > 1e-12:
            step = step / step_norm * learning_rate
        else:
            # very small gradient
            break
        
        # Line search
        best_alpha = 0.0
        best_trial_error = error_norm
        
        for alpha in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            trial_p = current_p + alpha * step
            trial_p = np.clip(trial_p, p_min, p_max)
            
            trial_pos = forward_kinematics_vector(trial_p, params, segments)
            trial_error = np.linalg.norm(target_pos - trial_pos)
            
            if trial_error < best_trial_error:
                best_trial_error = trial_error
                best_alpha = alpha
        
        if best_alpha == 0:
            learning_rate *= 0.5
            if learning_rate < 100:
                break
        else:
            current_p = current_p + best_alpha * step
            current_p = np.clip(current_p, p_min, p_max)
        
        if verbose and iteration % 10 == 0:
            print(f"  Iter {iteration}: error = {error_norm*1000:.4f} mm")
    
    gd_time = time.time() - gd_start
    total_time = time.time() - total_start
    
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"  After CMA-ES:           {cost_cma*1000:.3f} mm")
        print(f"  After Gradient Descent: {best_error*1000:.3f} mm")
        print(f"  Total time:             {total_time:.2f}s")
        print(f"\n  Solution pressures (kPa):")
        print(f"    pA = {best_p[0]/1e3:.2f}")
        print(f"    pB = {best_p[1]/1e3:.2f}")
        print(f"    pC = {best_p[2]/1e3:.2f}")
        print(f"    pD = {best_p[3]/1e3:.2f}")
    
    info_dict = {
        'cma_cost': cost_cma,
        'cma_iterations': cma_iters,
        'cma_time': cma_time,
        'dls_iterations': iteration,
        'dls_time': gd_time,
        'converged': best_error < gd_tolerance,
        'error_history_cma': cma_costs,
        'error_history_dls': error_history
    }
    
    return best_p, best_error, info_dict

def estimate_workspace_bounds(params, segments, pressure_bounds):
    epsilon_pre = params['epsilon_pre']
    total_arc_length = sum(seg.length * (1 + epsilon_pre) for seg in segments)

    p_min = pressure_bounds[:, 0].min()
    p_max = pressure_bounds[:, 1].max()
    max_pressure_diff = p_max - p_min

    channel_area = fk.channel_area(params['channel_radius'])
    centroid_dist = fk.centroid_distance(params['channel_radius'], params['septum_thickness'])

    M_max = max_pressure_diff * channel_area * centroid_dist
    min_EI = min(seg.EI for seg in segments)
    kappa_max = M_max / min_EI

    # Max total arc angle
    max_total_theta = sum(kappa_max * seg.length * (1 + epsilon_pre) for seg in segments)

    # XY reach using circular arc projection
    if kappa_max < 1e-6:
        max_xy_reach = total_arc_length
    else:
        # radius of curvature
        rho = 1 / kappa_max
        max_theta_clip = min(max_total_theta, 2 * np.pi)  # clip to 360 deg max
        max_xy_reach = 2 * rho * np.sin(max_theta_clip / 2)

    return {
        'total_arc_length': total_arc_length,
        'max_xy_reach': max_xy_reach,
        'max_pressure_diff': max_pressure_diff,
        'max_moment': M_max,
        'max_curvature': kappa_max,
        'max_total_theta': max_total_theta,
        'min_z': 0,
        'max_z': total_arc_length,
        'min_EI': min_EI
    }


def check_target_reachability(target_pos, workspace_bounds, verbose=True):
    """Check if target is reachable."""
    warnings = []
    
    target_xy_dist = np.linalg.norm(target_pos[:2])
    max_reach = workspace_bounds['max_xy_reach']
    
    is_reachable = target_xy_dist <= max_reach and 0 <= target_pos[2] <= workspace_bounds['max_z']
    
    if not is_reachable:
        warnings.append(f"Target may be outside workspace")
    
    if verbose and warnings:
        print(f"Reachability: {'OK' if is_reachable else 'WARNING'}")
        for w in warnings:
            print(f"  {w}")
    
    return is_reachable, warnings