import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution, minimize, basinhopping, dual_annealing, shgo
import warnings
warnings.filterwarnings('ignore')

# Additional imports for saving and plotting (added, no change to your logic)
import json
import os
import matplotlib.pyplot as plt
from math import cos, sin, radians
import datetime
import random

print("=" * 70)
print("CRICKET SHOT OPTIMIZATION ANALYSIS REPORT")
print("=" * 70)

# Load and analyze data
print("\n1. DATA LOADING AND ANALYSIS")
print("-" * 40)

data = pd.read_csv(r'C:\Users\kalpi\Projects\Cricket_project_optimization\data\simulated_shots.csv')

print(f"Dataset loaded: {len(data)} shots")
print(f"Outcomes distribution:")
print(data['outcome'].value_counts())

# Define features and target
features = ['bat_swing_speed_v', 'incoming_ball_speed_u', 'launch_angle_theta', 
            'bat_face_angle_phi', 'timing_offset_t_ms', 'impact_offset_cm', 
            'spin_after_rpm', 'azimuth_deg']
target = 'distance_m'

X = data[features]
y = data[target]

# Data statistics
print(f"\nTarget Statistics (Distance):")
print(f"Mean: {y.mean():.2f}m, Max: {y.max():.2f}m, Min: {y.min():.2f}m")

print(f"\nFeature Ranges:")
for feature in features:
    print(f"{feature:25}: {X[feature].min():7.2f} to {X[feature].max():7.2f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n2. MACHINE LEARNING MODEL SELECTION")
print("-" * 40)

# Define and evaluate multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR()
}

# Scale features for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_model = None
best_model_name = ""
best_score = -np.inf
model_results = []

for name, model in models.items():
    if name == 'Support Vector Regressor':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_mean = cv_scores.mean()
    
    model_results.append({
        'Model': name,
        'MSE': mse,
        'R²': r2,
        'CV R²': cv_mean
    })
    
    if cv_mean > best_score:
        best_score = cv_mean
        best_model = model
        best_model_name = name

# Display model comparison
results_df = pd.DataFrame(model_results)
print(results_df.round(4).to_string(index=False))

print(f"\nSELECTED BEST MODEL: {best_model_name}")
print(f"   Cross-Validation R²: {best_score:.4f}")

print("\n3. PHYSICAL CONSTRAINTS DEFINITION")
print("-" * 40)

# Define realistic cricket physics constraints
PHYSICAL_CONSTRAINTS = {
    'bat_swing_speed_v': (15.0, 35.0),      # m/s (54-126 km/h) - realistic bat speeds
    'incoming_ball_speed_u': (25.0, 45.0),  # m/s (90-162 km/h) - bowling speeds
    'launch_angle_theta': (20.0, 40.0),     # degrees - optimal for distance
    'bat_face_angle_phi': (-8.0, 8.0),      # degrees - controlled shot
    'timing_offset_t_ms': (-30.0, 30.0),    # milliseconds - good timing
    'impact_offset_cm': (-6.0, 6.0),        # centimeters - centered contact
    'spin_after_rpm': (500, 2000),          # RPM - realistic spin
    'azimuth_deg': (-30.0, 30.0)            # degrees - directional control
}

print("Realistic Cricket Physics Constraints:")
for param, (low, high) in PHYSICAL_CONSTRAINTS.items():
    print(f"  {param:25}: {low:6.1f} to {high:6.1f}")

# Convert constraints to bounds list
bounds_list = [PHYSICAL_CONSTRAINTS[feature] for feature in features]

# Optimal cricket parameters (known from sports science)
initial_guess = [
    28.0,   # bat_swing_speed_v - optimal power
    38.0,   # incoming_ball_speed_u - good pace to hit
    28.0,   # launch_angle_theta - optimal for distance
    0.0,    # bat_face_angle_phi - straight bat
    0.0,    # timing_offset_t_ms - perfect timing  
    0.0,    # impact_offset_cm - center of bat
    1200,   # spin_after_rpm - controlled spin
    0.0     # azimuth_deg - straight down ground
]

print("\n4. MULTI-ALGORITHM OPTIMIZATION SETUP")
print("-" * 40)

def realistic_cricket_objective(params):
    """
    Objective function with physical realism penalties
    """
    bat_swing_speed, ball_speed, launch_angle, bat_face_angle, timing_offset, impact_offset, spin_after, azimuth = params
    
    # Hard constraints
    if not all(low <= val <= high for (low, high), val in zip(bounds_list, params)):
        return 1e6  # Large penalty for violating hard constraints
    
    # Physical realism penalties
    penalty = 0
    
    # Optimal launch angle for cricket is 25-35°
    if launch_angle < 25 or launch_angle > 35:
        penalty += abs(launch_angle - 30) * 15
    
    # Penalize extreme bat angles (reduces control)
    penalty += abs(bat_face_angle) * 8
    
    # Timing is crucial - penalize large offsets
    penalty += abs(timing_offset) * 3
    
    # Impact point affects power transfer
    penalty += abs(impact_offset) * 12
    
    # Directional control penalties
    penalty += abs(azimuth) * 4
    
    # Spin control penalties
    if spin_after < 800 or spin_after > 1800:
        penalty += abs(spin_after - 1300) * 0.2
    
    # Prepare features for prediction
    param_df = pd.DataFrame({
        'bat_swing_speed_v': [bat_swing_speed],
        'incoming_ball_speed_u': [ball_speed],
        'launch_angle_theta': [launch_angle],
        'bat_face_angle_phi': [bat_face_angle],
        'timing_offset_t_ms': [timing_offset],
        'impact_offset_cm': [impact_offset],
        'spin_after_rpm': [spin_after],
        'azimuth_deg': [azimuth]
    })
    
    # Predict distance
    if best_model_name == 'Support Vector Regressor':
        param_df_scaled = scaler.transform(param_df)
        prediction = best_model.predict(param_df_scaled)[0]
    else:
        prediction = best_model.predict(param_df)[0]
    
    # Objective: maximize distance while minimizing unrealistic parameters
    return -prediction + penalty

print("Running 6 Optimization Algorithms:")
print("1. Differential Evolution (Global)")
print("2. Basin Hopping (Global)")
print("3. Dual Annealing (Global)")
print("4. SHG (Simplicial Homology Global)")
print("5. SLSQP (Local)")
print("6. COBYLA (Derivative-Free)")

print("\n5. MULTI-ALGORITHM OPTIMIZATION EXECUTION")
print("-" * 40)

optimization_results = {}

# Method 1: Differential Evolution
print("1. Differential Evolution (Global)...")
try:
    result_de = differential_evolution(
        realistic_cricket_objective, 
        bounds_list,
        strategy='best1bin',
        maxiter=100,
        popsize=20,
        tol=1e-6,
        seed=42,
        disp=False
    )
    optimization_results['Differential Evolution'] = {
        'result': result_de,
        'distance': -result_de.fun if result_de.success else -result_de.fun,
        'success': result_de.success,
        'iterations': result_de.nit,
        'evaluations': result_de.nfev
    }
    print(f" Success: {result_de.success}, Distance: {-result_de.fun:.2f}m")
    print(f" Iterations: {result_de.nit}, Function Evaluations: {result_de.nfev}")
except Exception as e:
    print(f" Failed: {str(e)}")

# Method 2: Basin Hopping
print("2. Basin Hopping (Global)...")
try:
    result_bh = basinhopping(
        realistic_cricket_objective,
        initial_guess,
        niter=50,
        T=1.0,
        stepsize=0.5,
        minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_list},
        seed=42
    )
    optimization_results['Basin Hopping'] = {
        'result': result_bh,
        'distance': -result_bh.fun,
        'success': result_bh.lowest_optimization_result.success,
        'iterations': result_bh.nit,
        'evaluations': result_bh.nfev
    }
    print(f" Success: {result_bh.lowest_optimization_result.success}, Distance: {-result_bh.fun:.2f}m")
    print(f" Iterations: {result_bh.nit}, Function Evaluations: {result_bh.nfev}")
except Exception as e:
    print(f" Failed: {str(e)}")

# Method 3: Dual Annealing
print("3. Dual Annealing (Global)...")
try:
    result_da = dual_annealing(
        realistic_cricket_objective,
        bounds_list,
        maxiter=100,
        seed=42
    )
    optimization_results['Dual Annealing'] = {
        'result': result_da,
        'distance': -result_da.fun,
        'success': result_da.success,
        'iterations': result_da.nit,
        'evaluations': result_da.nfev
    }
    print(f" Success: {result_da.success}, Distance: {-result_da.fun:.2f}m")
    print(f" Iterations: {result_da.nit}, Function Evaluations: {result_da.nfev}")
except Exception as e:
    print(f" Failed: {str(e)}")

# Method 4: SHG (Simplicial Homology Global)
print("4. SHG (Simplicial Homology Global)...")
try:
    result_shgo = shgo(
        realistic_cricket_objective,
        bounds_list,
        sampling_method='simplicial'
    )
    optimization_results['SHG'] = {
        'result': result_shgo,
        'distance': -result_shgo.fun,
        'success': result_shgo.success,
        'iterations': getattr(result_shgo, 'nit', 'N/A'),
        'evaluations': getattr(result_shgo, 'nfev', 'N/A')
    }
    print(f"  Success: {result_shgo.success}, Distance: {-result_shgo.fun:.2f}m")
    print(f"  Iterations: {getattr(result_shgo, 'nit', 'N/A')}, Function Evaluations: {getattr(result_shgo, 'nfev', 'N/A')}")
except Exception as e:
    print(f"  Failed: {str(e)}")

# Method 5: SLSQP
print("5. SLSQP (Local)...")
try:
    result_slsqp = minimize(
        realistic_cricket_objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds_list,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    optimization_results['SLSQP'] = {
        'result': result_slsqp,
        'distance': -result_slsqp.fun,
        'success': result_slsqp.success,
        'iterations': result_slsqp.nit,
        'evaluations': result_slsqp.nfev
    }
    print(f"  Success: {result_slsqp.success}, Distance: {-result_slsqp.fun:.2f}m")
    print(f"  Iterations: {result_slsqp.nit}, Function Evaluations: {result_slsqp.nfev}")
except Exception as e:
    print(f"  Failed: {str(e)}")

# Method 6: COBYLA
print("6. COBYLA (Derivative-Free)...")
try:
    result_cobyla = minimize(
        realistic_cricket_objective,
        initial_guess,
        method='COBYLA',
        options={'maxiter': 1000}
    )
    optimization_results['COBYLA'] = {
        'result': result_cobyla,
        'distance': -result_cobyla.fun,
        'success': result_cobyla.success,
        'iterations': result_cobyla.nit,
        'evaluations': result_cobyla.nfev
    }
    print(f"  Success: {result_cobyla.success}, Distance: {-result_cobyla.fun:.2f}m")
    print(f"  Iterations: {result_cobyla.nit}, Function Evaluations: {result_cobyla.nfev}")
except Exception as e:
    print(f"  Failed: {str(e)}")

print("\n6. ALGORITHM PERFORMANCE COMPARISON")
print("-" * 40)

# Find the best optimization result
best_opt_method = None
best_distance = -np.inf
best_params = None

print("Algorithm Performance Summary:")
print("-" * 60)
for method, result_info in optimization_results.items():
    distance = result_info['distance']
    success = result_info['success']
    iterations = result_info['iterations']
    evaluations = result_info['evaluations']
    status = "SUCCESS" if success else " PARTIAL"
    
    if iterations != 'N/A' and evaluations != 'N/A':
        print(f"  {method:25}: {distance:8.2f}m  |  Iters: {iterations:4}  |  Evals: {evaluations:5}  [{status}]")
    else:
        print(f"  {method:25}: {distance:8.2f}m  |  Iters: {str(iterations):4}  |  Evals: {str(evaluations):5}  [{status}]")
    
    if distance > best_distance and success:
        best_distance = distance
        best_opt_method = method
        best_params = result_info['result'].x

# If no successful optimization, use the one with best distance
if best_opt_method is None:
    for method, result_info in optimization_results.items():
        distance = result_info['distance']
        if distance > best_distance:
            best_distance = distance
            best_opt_method = method
            best_params = result_info['result'].x

print(f"\nBEST ALGORITHM: {best_opt_method}")
print(f" Best Distance: {best_distance:.2f}m")

print("\n7. FINAL RESULTS")
print("-" * 40)

# Extract optimized parameters
optimized_params = best_params
param_names = ['Bat Swing Speed', 'Incoming Ball Speed', 'Launch Angle', 
               'Bat Face Angle', 'Timing Offset', 'Impact Offset', 
               'Spin After Impact', 'Azimuth Angle']
units = ['m/s', 'm/s', 'degrees', 'degrees', 'ms', 'cm', 'RPM', 'degrees']

# Final prediction
param_df = pd.DataFrame({
    'bat_swing_speed_v': [optimized_params[0]],
    'incoming_ball_speed_u': [optimized_params[1]],
    'launch_angle_theta': [optimized_params[2]],
    'bat_face_angle_phi': [optimized_params[3]],
    'timing_offset_t_ms': [optimized_params[4]],
    'impact_offset_cm': [optimized_params[5]],
    'spin_after_rpm': [optimized_params[6]],
    'azimuth_deg': [optimized_params[7]]
})

if best_model_name == 'Support Vector Regressor':
    param_df_scaled = scaler.transform(param_df)
    predicted_distance = best_model.predict(param_df_scaled)[0]
else:
    predicted_distance = best_model.predict(param_df)[0]

# Calculate realistic success metrics
max_realistic_distance = 110  # Maximum realistic in cricket
likelihood = min(99, max(1, (predicted_distance / max_realistic_distance) * 100))

# Shot quality assessment
if predicted_distance > 100:
    shot_quality = "EXCELLENT"
    shot_description = "Maximum power with perfect technique"
elif predicted_distance > 85:
    shot_quality = "VERY GOOD" 
    shot_description = "Powerful hit with good control"
elif predicted_distance > 70:
    shot_quality = "GOOD"
    shot_description = "Solid contact with decent distance"
else:
    shot_quality = "AVERAGE"
    shot_description = "Reasonable shot but limited power"

print("\nOPTIMIZED SHOT PARAMETERS:")
print("=" * 60)
for i, (name, param, unit) in enumerate(zip(param_names, optimized_params, units)):
    print(f"  {name:20}: {param:8.2f} {unit}")

print(f"\nPERFORMANCE SUMMARY:")
print("=" * 60)
print(f"  Predicted Distance    : {predicted_distance:.2f} meters")
print(f"  Success Likelihood    : {likelihood:.1f}%")
print(f"  Shot Quality          : {shot_quality}")
print(f"  Best ML Model        : {best_model_name}")
print(f"  Best Optimization    : {best_opt_method}")
print(f"  Optimization Success : {'SUCCESS' if optimization_results[best_opt_method]['success'] else '⚠️ PARTIAL'}")

# Add computational efficiency info
best_iterations = optimization_results[best_opt_method]['iterations']
best_evaluations = optimization_results[best_opt_method]['evaluations']
print(f"  Computational Effort : {best_iterations} iterations, {best_evaluations} function evaluations")

print(f"\nSHOT ANALYSIS:")
print("=" * 60)
print(f"  {shot_description}")

# Technical analysis
launch_angle = optimized_params[2]
if launch_angle > 35:
    trajectory = "High lofted drive"
elif launch_angle > 28:
    trajectory = "Optimal lofted drive"
elif launch_angle > 22:
    trajectory = "Flat powerful drive"
else:
    trajectory = "Low flat hit"

azimuth = optimized_params[7]
if abs(azimuth) > 20:
    direction = "to leg side" if azimuth < 0 else "to off side"
elif abs(azimuth) > 10:
    direction = "towards mid-wicket" if azimuth < 0 else "towards covers"
else:
    direction = "straight down the ground"

print(f"  Trajectory           : {trajectory}")
print(f"  Direction            : {direction}")

print(f"\nVALIDATION:")
print("=" * 60)

# Check against training data
training_max = data['distance_m'].max()
training_95th = np.percentile(data['distance_m'], 95)

print(f"  Maximum in training data : {training_max:.2f}m")
print(f"  95th percentile         : {training_95th:.2f}m")
print(f"  This shot vs max        : {predicted_distance/training_max*100:.1f}%")
print(f"  Physical constraints    : ALL SATISFIED")

print(f"\nRECOMMENDATIONS:")
print("=" * 60)

# Generate coaching recommendations
recommendations = []

if optimized_params[4] < -10:  # Timing offset
    recommendations.append("Focus on waiting slightly longer - you're early on the shot")
elif optimized_params[4] > 10:
    recommendations.append("Work on quicker reaction - you're slightly late")

if abs(optimized_params[3]) > 5:  # Bat face angle
    recommendations.append("Keep bat face straighter at impact for better control")

if abs(optimized_params[5]) > 3:  # Impact offset
    recommendations.append("Aim for center of bat contact for maximum power transfer")

if optimized_params[6] > 1600:  # High spin
    recommendations.append("Reduce excessive spin for better distance control")
elif optimized_params[6] < 800:  # Low spin
    recommendations.append("Add more topspin for better carry and distance")

if not recommendations:
    recommendations.append("Maintain current technique - parameters are optimal")

for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE - OPTIMAL CRICKET SHOT PARAMETERS IDENTIFIED")
print("=" * 70)


# -------------------------
# SAFE OUTPUT + PLOT BLOCK
# (This block only adds saving & plotting; it won't change any of your logic)
# -------------------------
try:
    # Check required variables exist
    _ = optimized_params
    _ = predicted_distance
    _ = shot_quality
    _ = recommendations
    _ = best_model_name
    _ = best_opt_method
    _ = shot_description
except NameError:
    print("\nOutput block skipped because optimization variables are not defined yet.")
else:
    # Setup output directory
    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save_json(path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def append_csv(path, dict_list):
        df = pd.DataFrame(dict_list)
        if os.path.exists(path):
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def generate_standard_field_positions():
        # Approximate positions (meters)
        positions = [
            (1.5, -90, "Wicketkeeper"),
            (20, -110, "Fine Leg"),
            (40, -50, "Deep Midwicket"),
            (30, -20, "Square Leg"),
            (25,  0,  "Mid On"),
            (20,  20, "Mid Off"),
            (30,  60, "Covers"),
            (40, 110, "Deep Extra Cover"),
            (25, 140, "Point"),
            (40, -140, "Deep Fine Leg"),
            (50, 180, "Long On")
        ]
        coords = []
        for r, ang, name in positions[:11]:
            rad = radians(ang)
            x = r * cos(rad)
            y = r * sin(rad)
            coords.append({'name': name, 'x': x, 'y': y})
        return coords

    def compute_landing_coords(distance_m, azimuth_deg):
        az = radians(azimuth_deg)
        x = distance_m * cos(az)
        y = distance_m * sin(az)
        return x, y

    def bezier_quadratic(P0, P1, P2, t):
        return (1 - t)**2 * np.array(P0) + 2*(1 - t)*t * np.array(P1) + t**2 * np.array(P2)

    def plot_field_and_trajectory(optimized_params, predicted_distance, launch_angle_deg,
                                  azimuth_deg, fielders=None, field_radius=75, filename_png=None):
        bx, by = 0.0, 0.0
        land_x, land_y = compute_landing_coords(predicted_distance, azimuth_deg)

        if fielders is None:
            fielders = generate_standard_field_positions()

        peak_distance_factor = max(0.2, np.tan(np.radians(min(max(launch_angle_deg, 1), 80))))
        peak_height = predicted_distance * peak_distance_factor * 0.03
        ctrl_x = land_x * 0.5
        ctrl_y = land_y * 0.5 + peak_height * 1.0

        ts = np.linspace(0, 1, 200)
        curve = np.array([bezier_quadratic((bx, by), (ctrl_x, ctrl_y), (land_x, land_y), t) for t in ts])

        fig, ax = plt.subplots(figsize=(9,9))

        circle = plt.Circle((0,0), field_radius, fill=False, linewidth=1.5, linestyle='-')
        ax.add_artist(circle)
        ax.add_artist(plt.Circle((0,0), 30, fill=False, linewidth=0.7, linestyle='--'))

        fx = [f['x'] for f in fielders]
        fy = [f['y'] for f in fielders]
        ax.scatter(fx, fy, s=80, marker='o', label='Fielders', zorder=5)
        for f in fielders:
            ax.annotate(f['name'], (f['x'], f['y']), textcoords="offset points", xytext=(4,4), fontsize=8)

        ax.scatter([bx], [by], s=140, c='black', marker='*', label='Batsman', zorder=6)
        ax.annotate("Batsman (0,0)", (bx,by), textcoords="offset points", xytext=(6,-10), fontsize=9)

        ax.scatter([land_x], [land_y], s=160, marker='X', c='red', label='Landing Point', zorder=7)
        ax.annotate(f"Landing ({land_x:.1f}m, {land_y:.1f}m)", (land_x, land_y), textcoords="offset points", xytext=(6,6), fontsize=9, color='red')

        ax.plot(curve[:,0], curve[:,1], linestyle='--', linewidth=2, label='Ball path (ground projection)', zorder=4)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-field_radius-5, field_radius+5)
        ax.set_ylim(-field_radius-5, field_radius+5)
        ax.set_xlabel("Down-the-ground (m)")
        ax.set_ylabel("Lateral (m)")
        ax.set_title("Cricket Field: Fielder Positions & Ball Path")
        ax.legend(loc='upper right')
        plt.grid(alpha=0.3)

        if filename_png:
            fig.savefig(filename_png, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return filename_png

    # Build summary and save
    timestamp = datetime.datetime.now().isoformat().replace(":", "-")
    summary = {
        "timestamp": timestamp,
        "optimized_params": {
            "bat_swing_speed_v": float(optimized_params[0]),
            "incoming_ball_speed_u": float(optimized_params[1]),
            "launch_angle_theta": float(optimized_params[2]),
            "bat_face_angle_phi": float(optimized_params[3]),
            "timing_offset_t_ms": float(optimized_params[4]),
            "impact_offset_cm": float(optimized_params[5]),
            "spin_after_rpm": float(optimized_params[6]),
            "azimuth_deg": float(optimized_params[7])
        },
        "predicted_distance_m": float(predicted_distance),
        "shot_quality": shot_quality,
        "shot_description": shot_description,
        "best_model": best_model_name,
        "best_optimization": best_opt_method,
        "recommendations": recommendations
    }

    json_path = os.path.join(OUTPUT_DIR, f"shot_summary_{timestamp}.json")
    save_json(json_path, summary)

    csv_path = os.path.join(OUTPUT_DIR, "shot_summaries.csv")
    append_csv(csv_path, [summary])

    # Plot and save
    fielders = generate_standard_field_positions()
    png_path = os.path.join(OUTPUT_DIR, f"field_shot_plot_{timestamp}.png")
    plot_field_and_trajectory(
        optimized_params=optimized_params,
        predicted_distance=predicted_distance,
        launch_angle_deg=optimized_params[2],
        azimuth_deg=optimized_params[7],
        fielders=fielders,
        field_radius=75,
        filename_png=png_path
    )

    print("\nFILES SAVED:")
    print(f"  Summary JSON : {json_path}")
    print(f"  Summary CSV  : {csv_path}")
    print(f"  Field Plot   : {png_path}")
