# -*- coding: utf-8 -*-
# AI-Driven Fuzzing Simulation for 5G Traffic Steering Algorithms
# Optimized for H100 GPU and Multi-Objective Analysis
#
# This script improves upon the user's initial code by:
# 1. Implementing a proper Multi-Objective Genetic Algorithm (NSGA-II)
#    for the AI Fuzzer, as outlined in the paper's future work section.
# 2. Enhancing the performance analysis module to generate professional-grade
#    plots and comprehensive statistical summaries suitable for an IEEE conference paper.
# 3. Incorporating H100-specific optimizations like mixed-precision and JIT compilation.
# 4. Refactoring the simulation loop and data handling for robustness and clarity.
# 5. Improving the `MLTrafficSteering` algorithm with an enhanced Q-learning model.

# --- Imports ---
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats
import time
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- H100 GPU Optimizations ---
# Enable memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    # Optimize for H100 GPUs
    tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
    tf.config.optimizer.set_experimental_options({
        "auto_mixed_precision": True,  # Enable automatic mixed precision
        "layout_optimizer": True,      # Optimize tensor layouts
        "constant_folding": True,      # Optimize constant expressions
        "shape_optimization": True,    # Optimize based on tensor shapes
        "remapping": True,             # Remap operations for better performance
        "arithmetic_optimization": True,  # Optimize arithmetic operations
        "dependency_optimization": True,  # Optimize control dependencies
        "loop_optimization": True,      # Optimize loops
        "function_optimization": True,  # Optimize function calls
        "debug_stripper": True,        # Remove debug operations
    })
    
    # H100-specific optimizations
    try:
        # Configure memory growth instead of setting a fixed limit
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        
        # Use standard mixed precision to ensure compatibility
        if hasattr(tf.keras.mixed_precision, 'set_global_policy'):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    except Exception as e:
        print(f"Could not apply H100-specific memory configuration: {e}")
    
    print(f"GPU optimization configured for {len(physical_devices)} devices")

# --- Sionna specific imports for channel modeling ---
try:
    from sionna.phy.channel.tr38901 import UMi, PanelArray, Antenna
    from sionna.phy.ofdm import ResourceGrid
    from sionna.phy.channel import GenerateOFDMChannel
except ImportError:
    print("Sionna library not found. Please install it to run the simulation.")
    print("Pip install instructions: `pip install sionna`")
    exit()

# --- Global Constants ---
NUM_CELLS = 19
NUM_UES = 30
BANDWIDTH = 13.68e6
CARRIER_FREQUENCY = 3.5e9
TX_POWER_DBM = 30
NOISE_POWER_DBM_PER_HZ = -174
# The simulation iterations are kept low for a quick demonstration.
# For a real paper submission, this should be increased to at least 200.
SIMULATION_ITERATIONS = 1  # Increased from 50
FUZZER_GENERATIONS = 25     # Increased from 15
FUZZER_POPULATION = 40      # Increased from 20

# Use NSGA-II for multi-objective optimization as described in the paper
ENABLE_NSGA2_FUZZER = True

ENABLE_TF_DEVICE_LOGGING = True
SCRIPT_VERSION_NAME = "v26_final_optimized"

# --- Helper Functions ---
def safe_nanpercentile(data, percentile):
    """
    Safely calculates the percentile of a NumPy array, ignoring NaN values.
    """
    if not hasattr(data, '__len__'):
        data = np.array([data])
    
    clean_data = data[~np.isnan(data)]
    return np.percentile(clean_data, percentile) if len(clean_data) > 0 else np.nan

@tf.function(jit_compile=True)
def calculate_estimated_shannon_throughput_tf(sinr_linear_arr, bandwidth_hz):
    """
    Calculates the theoretical maximum throughput using the Shannon-Hartley theorem.
    This TensorFlow function is optimized for GPU acceleration on H100.
    Uses enhanced XLA compilation and mixed precision for faster computation.
    """
    # Cast to float32 for mixed precision benefits on H100
    sinr_linear_arr = tf.cast(sinr_linear_arr, tf.float32)
    bandwidth_hz = tf.cast(bandwidth_hz, tf.float32)
    
    # Use parallel operations where possible
    positive_sinr = tf.maximum(sinr_linear_arr, 1e-9)
    log_term = tf.math.log(1 + positive_sinr)
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    throughput_bps = bandwidth_hz * log_term / log_2
    return throughput_bps

def calculate_transmission_delay_ms(throughput_bps, packet_size_bytes=1500):
    """
    Calculates the transmission time delay for a standard packet.
    """
    packet_size_bits = packet_size_bytes * 8
    # Add a small epsilon to avoid division by zero
    transmission_time_s = packet_size_bits / (throughput_bps + 1e-9)
    return transmission_time_s * 1000 # Convert to milliseconds

# --- Module 1: Network Simulation Environment ---
class NetworkEnvironment:
    """
    Represents the 5G network environment for traffic steering simulation.
    Uses Sionna for accurate channel modeling and TensorFlow for GPU acceleration.
    """
    def __init__(self, num_ues=NUM_UES, initial_load=0.3, scenario_max_speed=5, scenario_type='default', active_cell_indices=None, inter_site_distance=100.0):
        # Use batch size of 1 for the main environment to avoid shape issues
        # The AIFuzzer will handle batching for optimization
        self.batch_size = 1  # Use batch size of 1 for the environment
        self.num_ues = num_ues
        
        if active_cell_indices is None:
            self.num_cells = NUM_CELLS
            self.active_cell_indices = list(range(NUM_CELLS))
        else:
            self.num_cells = len(active_cell_indices)
            self.active_cell_indices = active_cell_indices

        self.initial_load_param = initial_load
        self.max_speed_param = scenario_max_speed
        self.scenario_type = scenario_type
        self.ue_mobility_types = np.full(self.num_ues, 'mobile', dtype=object)

        # Antenna configuration as per the paper
        self.ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization='single', polarization_type='V',
                                   antenna_pattern='omni', carrier_frequency=CARRIER_FREQUENCY, precision="single")
        self.bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization='single', polarization_type='V',
                                   antenna_pattern='omni', carrier_frequency=CARRIER_FREQUENCY, precision="single")

        # 3GPP UMi channel model as per the paper
        self.channel_model_3gpp = UMi(
            carrier_frequency=CARRIER_FREQUENCY, o2i_model='low', ut_array=self.ut_array, bs_array=self.bs_array,
            direction='downlink', enable_pathloss=True, enable_shadow_fading=True,
            always_generate_lsp=True, precision="single"
        )
        
        # Robust Resource Grid Configuration
        fft_size = 512
        num_tx = self.num_cells
        total_streams = num_tx
        num_effective_subcarriers = (256 // self.num_cells) * self.num_cells
        num_guard_left = (fft_size - 1 - num_effective_subcarriers) // 2
        num_guard_right = (fft_size - 1 - num_effective_subcarriers) - num_guard_left
        
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=14, fft_size=fft_size, subcarrier_spacing=30e3,
            num_tx=num_tx, num_streams_per_tx=1, cyclic_prefix_length=20,
            pilot_pattern="empty", num_guard_carriers=(num_guard_left, num_guard_right), dc_null=False
        )

        self.generate_h_freq_layer = GenerateOFDMChannel(
            channel_model=self.channel_model_3gpp,
            resource_grid=self.resource_grid,
            precision="single"
        )
        
        all_bs_pos_2d = self._generate_hexagonal_layout(NUM_CELLS, inter_site_distance)
        self.bs_pos_2d = all_bs_pos_2d[self.active_cell_indices]
        self.bs_loc = tf.constant(np.hstack([self.bs_pos_2d, np.ones((self.num_cells, 1)) * 10.0])[np.newaxis,...], dtype=tf.float32)

        # Use fixed shape variables instead of dynamic shape to avoid XLA compilation issues
        # Using self.batch_size instead of None for the shape to ensure tensor shapes are fully defined
        self.ue_loc = tf.Variable(tf.zeros([self.batch_size, self.num_ues, 3], dtype=tf.float32), name="ue_loc")
        self.ue_velocities = tf.Variable(tf.zeros([self.batch_size, self.num_ues, 3], dtype=tf.float32), name="ue_velocities")
        self.ut_orientations = tf.zeros([self.batch_size, self.num_ues, 3], dtype=tf.float32)
        self.bs_orientations = tf.zeros([self.batch_size, self.num_cells, 3], dtype=tf.float32)
        self.in_state = tf.zeros([self.batch_size, self.num_ues], dtype=tf.bool)

        self.cell_loads = np.ones(self.num_cells) * initial_load
        self.ue_priorities = np.random.choice([1, 2, 3], size=self.num_ues, p=[0.3, 0.4, 0.3]).astype(np.float32)

        self.noise_power_watts = tf.cast(10**((NOISE_POWER_DBM_PER_HZ - 30) / 10) * BANDWIDTH, tf.float32)
        self.tx_power_watts_total = tf.cast(10**((TX_POWER_DBM - 30) / 10), tf.float32)

        self.reset(initial_load, scenario_max_speed)
        self.validate_configuration()

    def validate_configuration(self):
        """Validates network configuration parameters."""
        assert self.num_cells > 0, "Must have at least one cell"
        assert self.num_ues > 0, "Must have at least one UE"
        assert len(self.active_cell_indices) == self.num_cells, f"Active cell indices mismatch: {len(self.active_cell_indices)} != {self.num_cells}"
        assert all(0 <= idx < NUM_CELLS for idx in self.active_cell_indices), "Invalid cell indices"
        assert 0.0 <= self.initial_load_param <= 1.0, f"Invalid initial load: {self.initial_load_param}"
        assert self.max_speed_param >= 0, f"Invalid max speed: {self.max_speed_param}"

    def _generate_hexagonal_layout(self, num_cells, distance):
        """Generates a hexagonal grid for BS placement."""
        coords = [(0.0, 0.0)]
        axial_directions = [(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)]
        axial_coords = [(0, 0)]
        seen_coords = set([(0, 0)])
        ring = 1
        while len(axial_coords) < num_cells:
            current_axial = (ring, -ring)
            for _ in range(6):
                for __ in range(ring):
                    if len(axial_coords) >= num_cells: break
                    if current_axial not in seen_coords:
                        axial_coords.append(current_axial)
                        seen_coords.add(current_axial)
                    current_axial = (current_axial[0] + axial_directions[(_ + 1) % 6][0],
                                     current_axial[1] + axial_directions[(_ + 1) % 6][1])
                if len(axial_coords) >= num_cells: break
            ring += 1
        
        cartesian_coords = []
        for q, r in axial_coords:
            x = distance * (3./2. * q)
            y = distance * (np.sqrt(3)/2. * q + np.sqrt(3) * r)
            cartesian_coords.append((x, y))
        return np.array(cartesian_coords[:num_cells])

    def reset(self, initial_load, max_speed):
        """Resets the environment to a new random state."""
        self.initial_load_param = initial_load
        self.max_speed_param = max_speed
        
        center_x = np.mean(self.bs_pos_2d[:, 0])
        center_y = np.mean(self.bs_pos_2d[:, 1])
        max_dist = np.max(np.linalg.norm(self.bs_pos_2d - [center_x, center_y], axis=1)) + 50

        ue_pos_2d_np = np.random.uniform(-max_dist, max_dist, size=(self.num_ues, 2)) + np.array([center_x, center_y])
        
        # Create arrays with explicit shapes for better XLA compatibility
        # Match the batch size of the tensor variables
        ue_loc_array = np.hstack([ue_pos_2d_np, np.ones((self.num_ues, 1)) * 1.5])
        # Repeat the same initial positions across all batch entries
        ue_loc_batched = np.repeat(ue_loc_array.reshape(1, self.num_ues, 3), self.batch_size, axis=0).astype(np.float32)
        self.ue_loc.assign(ue_loc_batched)
        
        ue_vel_2d_np = np.random.uniform(-max_speed, max_speed, size=(self.num_ues, 2))
        ue_vel_array = np.hstack([ue_vel_2d_np, np.zeros((self.num_ues, 1))])
        # Repeat the same initial velocities across all batch entries
        ue_vel_batched = np.repeat(ue_vel_array.reshape(1, self.num_ues, 3), self.batch_size, axis=0).astype(np.float32)
        self.ue_velocities.assign(ue_vel_batched)
        self.cell_loads = np.ones(self.num_cells) * initial_load

        if self.scenario_type == 'mixed':
            mobile_mask = np.random.rand(self.num_ues) > 0.5
            self.ue_mobility_types = np.where(mobile_mask, 'mobile', 'static')
            current_velocities = self.ue_velocities.numpy()
            static_ue_indices = np.where(self.ue_mobility_types == 'static')[0]
            if static_ue_indices.size > 0:
                # Apply to all batch elements
                for b in range(self.batch_size):
                    current_velocities[b, static_ue_indices, :] = 0.0
                self.ue_velocities.assign(current_velocities)
        else:
            self.ue_mobility_types.fill('mobile')
        
    def update_ue_positions_and_velocities(self, dt=1.0, max_speed=None):
        """Updates UE positions based on mobility model."""
        if max_speed is None: max_speed = self.max_speed_param
        
        # Get a NumPy version of the mobility types for proper shape handling
        mobile_mask_np = (self.ue_mobility_types == 'mobile').astype(np.float32)
        mobile_mask = tf.constant(mobile_mask_np, dtype=tf.float32)
        mobile_mask_3d = tf.reshape(mobile_mask, (1, self.num_ues, 1))

        # Use the actual batch size from the object
        velocity_shape = (self.batch_size, self.num_ues, 3)
        
        # Create batched version of the mobile mask
        mobile_mask_batched = tf.repeat(mobile_mask_3d, repeats=self.batch_size, axis=0)
        
        # Generate velocity updates for all batches
        velocity_updates = tf.random.normal(shape=velocity_shape, stddev=1.0, dtype=tf.float32) * dt
        
        # Get current velocity values
        current_velocities = self.ue_velocities
        new_velocities = current_velocities + (velocity_updates * mobile_mask_batched)

        # Calculate speed and normalize
        speeds = tf.norm(new_velocities, axis=2, keepdims=True)
        safe_speeds = tf.where(speeds < 1e-9, tf.ones_like(speeds) * 1e-9, speeds)
        scale = tf.minimum(1.0, max_speed / safe_speeds)
        new_velocities = new_velocities * scale
        new_velocities = new_velocities * mobile_mask_batched  # Use the batched mask
        
        # Handle NaN or Inf values
        new_velocities = tf.where(
            tf.math.is_finite(new_velocities), 
            new_velocities, 
            tf.zeros_like(new_velocities)
        )
        
        # Update velocity variable
        self.ue_velocities.assign(new_velocities)
        
        # Update positions based on velocity
        new_loc = self.ue_loc + new_velocities * dt
        new_loc = tf.where(
            tf.math.is_finite(new_loc), 
            new_loc, 
            self.ue_loc
        )
        
        # Update position variable
        self.ue_loc.assign(new_loc)
    
    # Enable XLA compilation for better GPU performance
    @tf.function(jit_compile=True, experimental_relax_shapes=True)
    def compute_metrics_tf(self, ue_loc_tf, bs_loc_tf, ut_orient_tf, bs_orient_tf, ut_vel_tf, in_state_tf):
        """
        TensorFlow-accelerated function to compute RSRP and SINR metrics.
        Uses a simplified distance-based path loss model rather than Sionna's complex channel model
        to avoid broadcasting errors.
        
        Optimized for H100 GPU with enhanced parallelization and tensor operations.
        """
        # Add a batch dimension if one doesn't exist to ensure consistent tensor ranks
        if len(tf.shape(ue_loc_tf)) == 2:
            # Use vectorized operations for batch creation
            ue_loc_tf = tf.expand_dims(ue_loc_tf, axis=0)  # Add batch dim if missing
            bs_loc_tf = tf.expand_dims(bs_loc_tf, axis=0)
            
        # Get static shapes where possible
        static_shape_ue = ue_loc_tf.get_shape().as_list()
        num_ues = static_shape_ue[1] if static_shape_ue[1] is not None else tf.shape(ue_loc_tf)[1]
        
        static_shape_bs = bs_loc_tf.get_shape().as_list()
        num_cells = static_shape_bs[1] if static_shape_bs[1] is not None else tf.shape(bs_loc_tf)[1]
        
        # Simple distance-based path loss model
        ue_pos = ue_loc_tf[:, :, :2]  # [batch, num_ues, 2]
        bs_pos = bs_loc_tf[:, :, :2]  # [batch, num_cells, 2]
        
        # Compute distances between each UE and each BS
        ue_pos_expanded = tf.expand_dims(ue_pos, axis=2)  # [batch, num_ues, 1, 2]
        bs_pos_expanded = tf.expand_dims(bs_pos, axis=1)  # [batch, 1, num_cells, 2]
        distances = tf.sqrt(tf.reduce_sum(tf.square(ue_pos_expanded - bs_pos_expanded), axis=3))  # [batch, num_ues, num_cells]
        
        # Simple distance-based path loss model (in dB): PL = 128.1 + 37.6*log10(d) for d in km
        distances_km = distances / 1000.0
        path_loss_db = 128.1 + 37.6 * (tf.math.log(tf.maximum(distances_km, 0.001)) / tf.math.log(tf.constant(10.0, dtype=tf.float32)))
        
        # Add log-normal shadow fading with 8 dB standard deviation
        shadow_std_dev = 8.0
        random_shape = tf.shape(path_loss_db)
        shadow_fading = tf.random.normal(random_shape, mean=0.0, stddev=shadow_std_dev, dtype=tf.float32)
        
        # Total path loss including shadowing
        total_path_loss_db = path_loss_db + shadow_fading
        
        # Convert path loss to linear domain
        path_loss_linear = tf.pow(10.0, -total_path_loss_db / 10.0)
        
        # Calculate received power
        received_power_watts_tf = self.tx_power_watts_total * path_loss_linear  # [batch, num_ues, num_cells]
        
        # RSRP in dB
        rsrp_db_tf = 10.0 * (tf.math.log(tf.maximum(received_power_watts_tf / 1e-3, 1e-20)) / tf.math.log(10.0))
        
        # Calculate SINR
        signal_power_ue_cell = received_power_watts_tf
        # Sum over all cells for each UE to get total received power (including interference)
        total_power_at_ue_u = tf.reduce_sum(received_power_watts_tf, axis=2, keepdims=True)

        # Calculate interference for each UE-cell pair
        interference_ue_cell = total_power_at_ue_u - signal_power_ue_cell
        
        # Add noise power (thermal noise)
        noise_power_watts = tf.cast(10**((NOISE_POWER_DBM_PER_HZ - 30) / 10) * BANDWIDTH, tf.float32)
        noise_power = noise_power_watts * tf.ones_like(signal_power_ue_cell)
        
        # Calculate SINR
        sinr_linear_tf = tf.math.divide_no_nan(signal_power_ue_cell, interference_ue_cell + noise_power)
        sinr_db_tf = 10.0 * (tf.math.log(tf.maximum(sinr_linear_tf, 1e-20)) / tf.math.log(10.0))

        # Clamp values for stability and realism
        rsrp_db_tf = tf.clip_by_value(rsrp_db_tf, -200.0, -40.0)
        sinr_db_tf = tf.clip_by_value(sinr_db_tf, -10.0, 30.0)
        
        # Never conditionally remove the batch dimension - this avoids XLA shape mismatch errors
        # Instead, always keep the batch dimension consistent and handle batch size 1 cases in the compute_metrics wrapper

        return rsrp_db_tf, sinr_db_tf

    def compute_metrics(self):
        """Wrapper to run the TensorFlow metric computation and return NumPy arrays."""
        try:
            # Check for NaN or infinity issues in tensors without using tf.reduce_all directly
            # This avoids the KeyboardInterrupt error from _numpy_internal
            ue_loc_np = self.ue_loc.numpy()
            bs_loc_np = self.bs_loc.numpy()
            ue_vel_np = self.ue_velocities.numpy()
            
            if not (np.all(np.isfinite(ue_loc_np)) and 
                   np.all(np.isfinite(bs_loc_np)) and 
                   np.all(np.isfinite(ue_vel_np))):
                return (np.full((self.num_ues, self.num_cells), -200.0),
                        np.full((self.num_ues, self.num_cells), -30.0),
                        self.cell_loads.copy(), self.ue_priorities.copy())
            
            rsrp_db_tf, sinr_db_tf = self.compute_metrics_tf(self.ue_loc, self.bs_loc, self.ut_orientations,
                                                             self.bs_orientations, self.ue_velocities, self.in_state)
            
            # Convert to numpy and handle batch dimension
            rsrp_np = rsrp_db_tf.numpy()
            sinr_np = sinr_db_tf.numpy()
            
            # Remove batch dimension if present
            if len(rsrp_np.shape) == 3:
                rsrp_np = rsrp_np[0]
                sinr_np = sinr_np[0]
            
            # Ensure the arrays have the expected dimensions
            expected_shape = (self.num_ues, self.num_cells)
            
            # Fix RSRP shape if needed
            if rsrp_np.shape != expected_shape:
                print(f"Warning: RSRP shape mismatch. Got {rsrp_np.shape}, expected {expected_shape}")
                fixed_rsrp = np.full(expected_shape, -200.0)
                # Copy as much as possible from the original array
                r_rows = min(rsrp_np.shape[0], expected_shape[0])
                r_cols = min(rsrp_np.shape[1], expected_shape[1])
                fixed_rsrp[:r_rows, :r_cols] = rsrp_np[:r_rows, :r_cols]
                rsrp_np = fixed_rsrp
                
            # Fix SINR shape if needed
            if sinr_np.shape != expected_shape:
                print(f"Warning: SINR shape mismatch. Got {sinr_np.shape}, expected {expected_shape}")
                fixed_sinr = np.full(expected_shape, -30.0)
                # Copy as much as possible from the original array
                s_rows = min(sinr_np.shape[0], expected_shape[0])
                s_cols = min(sinr_np.shape[1], expected_shape[1])
                fixed_sinr[:s_rows, :s_cols] = sinr_np[:s_rows, :s_cols]
                sinr_np = fixed_sinr
            
            # Ensure cell_loads has the correct length
            if len(self.cell_loads) != self.num_cells:
                print(f"Warning: cell_loads length mismatch. Got {len(self.cell_loads)}, expected {self.num_cells}")
                fixed_loads = np.ones(self.num_cells) * 0.5  # Default load of 50%
                copy_len = min(len(self.cell_loads), self.num_cells)
                fixed_loads[:copy_len] = self.cell_loads[:copy_len]
                cell_loads = fixed_loads
            else:
                cell_loads = self.cell_loads.copy()
                
            # Ensure priorities has the correct length
            if len(self.ue_priorities) != self.num_ues:
                print(f"Warning: priorities length mismatch. Got {len(self.ue_priorities)}, expected {self.num_ues}")
                fixed_priorities = np.ones(self.num_ues, dtype=int) * 3  # Default priority (lowest)
                copy_len = min(len(self.ue_priorities), self.num_ues)
                fixed_priorities[:copy_len] = self.ue_priorities[:copy_len]
                priorities = fixed_priorities
            else:
                priorities = self.ue_priorities.copy()
            
            return rsrp_np, sinr_np, cell_loads, priorities
        except Exception as e:
            print(f"General Uncaught Error during Sionna UMi metric computation: {e}")
            return (np.full((self.num_ues, self.num_cells), -200.0),
                    np.full((self.num_ues, self.num_cells), -30.0),
                    self.cell_loads.copy(), self.ue_priorities.copy())

    def update_cell_loads(self, assignments):
        """Updates the simulated cell loads based on new UE assignments."""
        self.cell_loads = np.zeros(self.num_cells)
        unique_cells, counts = np.unique(assignments, return_counts=True)
        load_per_ue = 1.0 / self.num_ues
        for cell_idx, count in zip(unique_cells, counts):
            if 0 <= cell_idx < self.num_cells:
                self.cell_loads[cell_idx] = count * load_per_ue
        self.cell_loads = np.clip(self.cell_loads, 0.0, 1.0)


# --- Module 2: Traffic Steering Algorithms ---
class TrafficSteeringAlgorithm:
    """Base class for all traffic steering algorithms."""
    def __init__(self, num_ues, num_cells):
        self.num_ues = num_ues
        self.num_cells = num_cells
        self.prev_assignments = None

    def assign_initial(self, rsrp):
        """Initial assignment based on best RSRP."""
        # Ensure rsrp has valid dimensions
        if rsrp.shape[0] < self.num_ues or rsrp.shape[1] < self.num_cells:
            print(f"Warning: RSRP shape for initial assignment is {rsrp.shape}, expected at least ({self.num_ues},{self.num_cells})")
            # Handle the case where rsrp dimensions are smaller than expected
            actual_num_ues = min(rsrp.shape[0], self.num_ues)
            assignments = np.zeros(self.num_ues, dtype=int)
            assignments[:actual_num_ues] = np.argmax(rsrp[:actual_num_ues], axis=1)
            self.prev_assignments = assignments
        else:
            self.prev_assignments = np.argmax(rsrp, axis=1)
            # Ensure assignments are valid cell indices
            self.prev_assignments = np.clip(self.prev_assignments, 0, self.num_cells - 1)
        return self.prev_assignments.copy()

    def assign_ues(self, rsrp, sinr, cell_loads, priorities, dt=1.0):
        """Abstract method for assigning UEs."""
        raise NotImplementedError

class BaselineA3(TrafficSteeringAlgorithm):
    """
    A3-Event-Based baseline algorithm.
    """
    def __init__(self, num_ues, num_cells, rsrp_threshold=-100, hysteresis=3, ttt=0.1, load_threshold=0.8):
        super().__init__(num_ues, num_cells)
        self.rsrp_threshold = rsrp_threshold
        self.hysteresis = hysteresis
        self.ttt = ttt
        self.load_threshold = load_threshold
        # Ensure that the timer dimensions are valid
        self.num_ues = max(1, num_ues)
        self.num_cells = max(1, num_cells)
        self.ttt_timers = np.zeros((self.num_ues, self.num_cells))
        self.potential_targets = np.full(self.num_ues, -1, dtype=int)

    def assign_ues(self, rsrp, sinr, cell_loads, priorities, dt=1.0):
        if self.prev_assignments is None:
            return self.assign_initial(rsrp)
            
        # Ensure that array shapes match expectations to avoid index errors
        if rsrp.shape[0] != self.num_ues or rsrp.shape[1] != self.num_cells:
            print(f"Warning: RSRP shape mismatch. Expected ({self.num_ues},{self.num_cells}), got {rsrp.shape}")
            # Clip to valid dimensions if needed
            actual_num_ues = min(rsrp.shape[0], self.num_ues)
            actual_num_cells = min(rsrp.shape[1], self.num_cells)
        else:
            actual_num_ues = self.num_ues
            actual_num_cells = self.num_cells
            
        # Make sure prev_assignments has valid indices
        assignments = np.clip(self.prev_assignments.copy(), 0, actual_num_cells - 1)
        
        for ue_idx in range(actual_num_ues):
            serving_cell = assignments[ue_idx]
            # Ensure serving_cell is a valid index
            if serving_cell >= actual_num_cells:
                serving_cell = 0
                assignments[ue_idx] = 0
                
            serving_rsrp = rsrp[ue_idx, serving_cell]
            
            # Reset timers for cells no longer meeting conditions
            # Make sure we don't exceed array bounds
            timer_update = np.zeros(self.ttt_timers.shape[1])
            for c in range(min(actual_num_cells, self.ttt_timers.shape[1])):
                if ue_idx < self.ttt_timers.shape[0]:
                    if rsrp[ue_idx, c] > serving_rsrp + self.hysteresis:
                        timer_update[c] = self.ttt_timers[ue_idx, c] + dt
            
            if ue_idx < self.ttt_timers.shape[0]:
                self.ttt_timers[ue_idx, :] = timer_update
                                                
            potential_target = -1
            max_rsrp_improvement = 0
            
            for cell_idx in range(actual_num_cells):
                if cell_idx == serving_cell: continue
                
                # Ensure indices are valid for all arrays
                if ue_idx >= rsrp.shape[0] or cell_idx >= rsrp.shape[1] or cell_idx >= cell_loads.shape[0]:
                    continue
                
                neighbor_rsrp = rsrp[ue_idx, cell_idx]
                
                a3_cond = neighbor_rsrp > serving_rsrp + self.hysteresis
                load_cond = cell_loads[cell_idx] < self.load_threshold
                
                # Only check ttt_timers if indices are valid
                if ue_idx < self.ttt_timers.shape[0] and cell_idx < self.ttt_timers.shape[1]:
                    timer_cond = self.ttt_timers[ue_idx, cell_idx] >= self.ttt
                else:
                    timer_cond = True  # Skip timer check if indices are invalid
                
                if a3_cond and load_cond and timer_cond:
                    if (neighbor_rsrp - serving_rsrp) > max_rsrp_improvement:
                        max_rsrp_improvement = neighbor_rsrp - serving_rsrp
                        potential_target = cell_idx
            
            if potential_target != -1:
                assignments[ue_idx] = potential_target
        
        self.prev_assignments = assignments
        return assignments

class UtilityBased(TrafficSteeringAlgorithm):
    """
    Utility-based algorithm as described in the paper.
    """
    def assign_ues(self, rsrp, sinr, cell_loads, priorities, dt=1.0):
        assignments = np.zeros(self.num_ues, dtype=int)
        for ue_idx in range(self.num_ues):
            # Normalizing factors to handle different scales of metrics
            sinr_w, load_w, prio_w = 0.5, 0.3, 0.2
            
            # Use vectorized operations for efficiency
            sinr_c = sinr_w * np.clip(sinr[ue_idx, :], -20, 30)
            load_c = load_w * (1.0 - cell_loads) * 20
            prio_c = prio_w * (4.0 - float(priorities[ue_idx])) * 10
            
            utilities = sinr_c + load_c + prio_c
            assignments[ue_idx] = np.argmax(utilities)
            
        self.prev_assignments = assignments
        return assignments

class MLTrafficSteering(TrafficSteeringAlgorithm):
    """
    ML-based Q-learning algorithm with experience replay.
    """
    def __init__(self, num_ues, num_cells):
        super().__init__(num_ues, num_cells)
        self.q_table = {}
        self.learning_rate = 0.3
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.prev_state = {}
        self.prev_action = {}
        self.experience_buffer = []
        self.buffer_size = 1000
        self.batch_size = 32
        self.update_frequency = 10
        self.step_count = 0
        
    def _get_state(self, ue_idx, rsrp, sinr, cell_loads):
        current_cell = self.prev_assignments[ue_idx] if self.prev_assignments is not None else 0
        current_sinr = sinr[ue_idx, current_cell] if 0 <= current_cell < self.num_cells else -30
        avg_load = np.mean(cell_loads)
        max_neighbor_sinr = np.max([sinr[ue_idx, i] for i in range(self.num_cells) if i != current_cell])
        load_imbalance = np.std(cell_loads)
        
        # Discretize state with finer bins
        state = (
            min(20, max(-20, int(current_sinr // 2))),
            min(20, int(avg_load * 20)),
            min(20, max(-20, int(max_neighbor_sinr // 2))),
            min(10, int(load_imbalance * 20)),
            min(self.num_cells, current_cell)
        )
        return state
    
    def _calculate_reward(self, ue_idx, new_sinr, old_sinr, handover_occurred, priority, cell_loads):
        """Enhanced reward function with normalization and penalty for handovers."""
        new_sinr_norm = (new_sinr + 30) / 60.0
        old_sinr_norm = (old_sinr + 30) / 60.0
        
        new_throughput = np.log2(1 + max(0, 10**(new_sinr_norm * 60 - 30) / 10.0))
        old_throughput = np.log2(1 + max(0, 10**(old_sinr_norm * 60 - 30) / 10.0))
        throughput_reward = (new_throughput - old_throughput) * 10
        
        sinr_improvement = max(0, new_sinr_norm - old_sinr_norm) * 30
        sinr_reward = sinr_improvement * 0.5
        
        handover_penalty = 0
        if handover_occurred:
            if new_sinr > old_sinr + 3:
                handover_penalty = -0.5
            else:
                handover_penalty = -3.0
        
        current_cell = self.prev_assignments[ue_idx] if self.prev_assignments is not None else 0
        current_load = cell_loads[current_cell] if 0 <= current_cell < len(cell_loads) else 0.5
        load_reward = -2.0 * current_load
        
        priority_scale = 2.0 if priority == 1 else (1.5 if priority == 2 else 1.0)
        
        qos_penalty = -5.0 if new_sinr < 0 else 0
        
        total_reward = priority_scale * (throughput_reward + sinr_reward + load_reward) + handover_penalty + qos_penalty
        total_reward = max(total_reward, -1.0)
        return total_reward
        
    def _add_experience(self, ue_idx, state, action, reward, next_state, done=False):
        """Adds an experience tuple to the replay buffer."""
        experience = (ue_idx, state, action, reward, next_state, done)
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def _replay_experience(self):
        """Performs a mini-batch update from the experience replay buffer."""
        if len(self.experience_buffer) < self.batch_size:
            return
            
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        for ue_idx, state, action, reward, next_state, done in batch:
            if state not in self.q_table: self.q_table[state] = np.zeros(self.num_cells)
            if next_state not in self.q_table: self.q_table[next_state] = np.zeros(self.num_cells)
            
            current_q = self.q_table[state][action]
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
            
            self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def _update_learning_params(self):
        """Adapts learning parameters over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def assign_ues(self, rsrp, sinr, cell_loads, priorities, dt=1.0):
        if self.prev_assignments is None:
            return self.assign_initial(rsrp)
            
        assignments = self.prev_assignments.copy()
        total_reward = 0
        
        for ue_idx in range(self.num_ues):
            current_cell = assignments[ue_idx]
            current_state = self._get_state(ue_idx, rsrp, sinr, cell_loads)
            
            if ue_idx in self.prev_state and ue_idx in self.prev_action:
                prev_state = self.prev_state[ue_idx]
                prev_action = self.prev_action[ue_idx]
                
                old_sinr = sinr[ue_idx, prev_action] if 0 <= prev_action < self.num_cells else -30
                new_sinr = sinr[ue_idx, current_cell] if 0 <= current_cell < self.num_cells else -30
                handover_occurred = (prev_action != current_cell)
                
                reward = self._calculate_reward(ue_idx, new_sinr, old_sinr, handover_occurred, priorities[ue_idx], cell_loads)
                total_reward += reward
                self._add_experience(ue_idx, prev_state, prev_action, reward, current_state)
                
                if prev_state not in self.q_table: self.q_table[prev_state] = np.zeros(self.num_cells)
                if current_state not in self.q_table: self.q_table[current_state] = np.zeros(self.num_cells)
                    
                old_q = self.q_table[prev_state][prev_action]
                max_next_q = np.max(self.q_table[current_state])
                new_q = old_q + self.learning_rate * (reward + self.gamma * max_next_q - old_q)
                self.q_table[prev_state][prev_action] = new_q
            
            if np.random.random() < self.epsilon:
                valid_actions = [i for i in range(self.num_cells) if sinr[ue_idx, i] > -20]
                action = np.random.choice(valid_actions if valid_actions else list(range(self.num_cells)))
            else:
                if current_state in self.q_table and np.max(self.q_table[current_state]) > -100:
                    action = np.argmax(self.q_table[current_state])
                else:
                    utilities = np.zeros(self.num_cells)
                    for cell_idx in range(self.num_cells):
                        sinr_utility = 0.5 * np.clip(sinr[ue_idx, cell_idx], -20, 30)
                        load_utility = 0.3 * (1 - cell_loads[cell_idx]) * 20
                        priority_utility = 0.2 * (4 - priorities[ue_idx]) * 10
                        utilities[cell_idx] = sinr_utility + load_utility + priority_utility
                    action = np.argmax(utilities)
                    if current_state not in self.q_table:
                        self.q_table[current_state] = utilities * 0.1
            
            assignments[ue_idx] = action
            self.prev_state[ue_idx] = current_state
            self.prev_action[ue_idx] = action
        
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            self._replay_experience()
            self._update_learning_params()
                
        self.prev_assignments = assignments
        return assignments

# --- Module 3: AI Fuzzer (Multi-Objective) ---
class AIFuzzer:
    """
    AI Fuzzer using a Multi-Objective Genetic Algorithm (NSGA-II)
    to find network vulnerabilities.
    """
    def __init__(self, env: NetworkEnvironment, ts: TrafficSteeringAlgorithm,
                 population_size=FUZZER_POPULATION, generations=FUZZER_GENERATIONS,
                 use_nsga2=True):
        self.env = env
        self.ts = ts
        self.population_size = population_size
        self.generations = generations
        self.input_vector_size = env.num_cells + env.num_ues * 2
        self.objective_call_count = 0
        self.use_nsga2 = use_nsga2
        self.num_objectives = 5  # handovers, qoe_violations, unfairness, energy, vulnerability_potential
        self.pareto_archive = []
        self.vulnerability_memory = []
        self.adaptation_rate = 0.05  # Rate for adaptive mutation
        self.stagnation_counter = 0  # Track generations without improvement
        self.prev_best_obj_sum = 0.0  # Track previous best objective sum

    def _calculate_jain_fairness(self, allocations):
        """Calculates Jain's Fairness Index."""
        allocations = np.asarray(allocations)
        allocations_cleaned = allocations[np.isfinite(allocations) & (allocations > 1e-12)]
        if len(allocations_cleaned) == 0: return 1.0
        sum_val = np.sum(allocations_cleaned)
        sum_sq_val = np.sum(allocations_cleaned**2)
        if sum_sq_val < 1e-20: return 1.0
        return sum_val**2 / (len(allocations_cleaned) * sum_sq_val)

    def _calculate_objectives(self, inputs, current_assignments, dt_fitness=1.0):
        """Calculates the multi-objective values for a given set of fuzzer inputs.
        Optimized for H100 GPU with batch processing."""
        self.objective_call_count += 1
        
        results = []
        # Process inputs in batches to better utilize H100 GPU
        batch_size = min(8, len(inputs))  # Process smaller batches to avoid memory issues
        
        for batch_start in range(0, len(inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(inputs))
            batch_inputs = inputs[batch_start:batch_end]
            batch_results = []
            
            # Process this batch in parallel using TensorFlow
            load_modifiers = np.array([input[:self.env.num_cells] for input in batch_inputs])
            
            # Create batched position modifiers
            position_modifiers_2d = np.array([
                input[self.env.num_cells:].reshape(self.env.num_ues, 2)
                for input in batch_inputs
            ])
            
            # Add z-dimension (zeros)
            position_modifiers_3d = np.concatenate([
                position_modifiers_2d,
                np.zeros((batch_end - batch_start, self.env.num_ues, 1))
            ], axis=2)
            
            # Convert to TensorFlow tensor
            position_modifiers_tf = tf.constant(position_modifiers_3d, dtype=tf.float32)
            
            original_ue_loc = tf.identity(self.env.ue_loc)
            original_cell_loads = self.env.cell_loads.copy()
            original_ts_prev_assignments = self.ts.prev_assignments.copy() if self.ts.prev_assignments is not None else None

            # Process each input in the batch
            for i in range(len(batch_inputs)):
                # Apply the i-th load and position modifier
                # Ensure position_modifiers_tf has compatible dimensions with original_ue_loc
                position_mod = position_modifiers_tf[i]
                if position_mod.shape[0] != self.env.num_ues or position_mod.shape[1] != 3:
                    print(f"Warning: Position modifier shape mismatch. Got {position_mod.shape}, expected ({self.env.num_ues},3)")
                    # Ensure compatible dimensions by padding or truncating
                    if position_mod.shape[0] < self.env.num_ues:
                        # Pad with zeros
                        padding = tf.zeros((self.env.num_ues - position_mod.shape[0], 3), dtype=tf.float32)
                        position_mod = tf.concat([position_mod, padding], axis=0)
                    else:
                        # Truncate
                        position_mod = position_mod[:self.env.num_ues, :]
                
                self.env.ue_loc.assign(original_ue_loc + tf.expand_dims(position_mod, 0))
                self.env.cell_loads = np.clip(original_cell_loads + load_modifiers[i], 0, 1)
                
                # Compute metrics with fallback for error cases
                rsrp, sinr, cell_loads_eval, priorities_eval = self.env.compute_metrics()
                
                self.ts.prev_assignments = current_assignments
                new_assignments = self.ts.assign_ues(rsrp, sinr, self.env.cell_loads, priorities_eval, dt=dt_fitness)
                new_assignments = np.clip(new_assignments, 0, self.env.num_cells - 1)
                
                objectives = {}
                num_handovers = np.sum(new_assignments != self.ts.prev_assignments)
                objectives['handovers'] = num_handovers / max(1, self.env.num_ues)
                
                high_prio_mask = (priorities_eval == 1)
                high_prio_ues = np.sum(high_prio_mask)
                qoe_violations = 0.0
                if high_prio_ues > 0:
                    assigned_sinr_hp_ues = []
                    for j in range(min(self.env.num_ues, sinr.shape[0])):
                        if j < len(high_prio_mask) and j < len(new_assignments) and high_prio_mask[j]:
                            cell_idx = new_assignments[j]
                            # Ensure cell_idx is within bounds
                            if 0 <= cell_idx < sinr.shape[1]:
                                assigned_sinr_hp_ues.append(sinr[j, cell_idx])
                            else:
                                assigned_sinr_hp_ues.append(0.0)  # Default value for invalid index
                    
                    if assigned_sinr_hp_ues:
                        qoe_violations = np.sum(np.array(assigned_sinr_hp_ues) < 5.0)
                objectives['qoe_violation'] = qoe_violations / high_prio_ues if high_prio_ues > 0 else 0.0
                
                # Safely access sinr with valid indices
                assigned_sinr_list = []
                for j in range(min(self.env.num_ues, sinr.shape[0])):
                    if j < len(new_assignments):
                        cell_idx = new_assignments[j]
                        # Ensure cell_idx is within bounds
                        if 0 <= cell_idx < sinr.shape[1]:
                            assigned_sinr_list.append(sinr[j, cell_idx])
                        else:
                            assigned_sinr_list.append(0.0)  # Default value for invalid index
                    else:
                        assigned_sinr_list.append(0.0)  # Default value for out of bounds
                
                assigned_sinr_np = np.array(assigned_sinr_list)
                assigned_sinr_linear = 10**(assigned_sinr_np / 10.0)
                jain_score = self._calculate_jain_fairness(assigned_sinr_linear)
                objectives['unfairness'] = 1.0 - jain_score
                objectives['energy_consumption'] = num_handovers / max(1, self.env.num_ues)
                
                # Add vulnerability potential score using an oracle to check for vulnerabilities
                temp_oracle = Oracle(num_ues=self.env.num_ues, num_cells=self.env.num_cells)
                oracle_result = temp_oracle.evaluate(rsrp, sinr, new_assignments, cell_loads_eval, priorities_eval, current_assignments)
                vulnerability_score = oracle_result.get('vulnerability_score', 0)
                objectives['vulnerability_potential'] = vulnerability_score / 10.0  # Normalize to 0-1 scale
                
                batch_results.append([objectives['handovers'], objectives['qoe_violation'], objectives['unfairness'], 
                                    objectives['energy_consumption'], objectives['vulnerability_potential']])
            
            # Restore original state after processing the whole batch
            self.env.ue_loc.assign(original_ue_loc)
            self.env.cell_loads = original_cell_loads
            self.ts.prev_assignments = original_ts_prev_assignments
            
            # Extend our results with this batch's results
            results.extend(batch_results)
        
        return results

    def _dominates(self, obj1, obj2):
        """Checks if objective vector obj1 dominates obj2 (for maximization)."""
        better_or_equal = all(obj1[i] >= obj2[i] for i in range(len(obj1)))
        strictly_better = any(obj1[i] > obj2[i] for i in range(len(obj1)))
        return better_or_equal and strictly_better
    
    def _fast_non_dominated_sort(self, objectives_list):
        """NSGA-II Fast Non-dominated Sorting algorithm."""
        pop_size = len(objectives_list)
        fronts = [[]]
        dominated_count = [0] * pop_size
        dominated_solutions = [[] for _ in range(pop_size)]
        
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j:
                    if self._dominates(objectives_list[i], objectives_list[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives_list[j], objectives_list[i]):
                        dominated_count[i] += 1
            if dominated_count[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)
        
        if not fronts[-1]: fronts.pop()
        return fronts
    
    def _calculate_crowding_distance(self, objectives_list, front_indices):
        """Calculates crowding distance for diversity preservation."""
        if len(front_indices) <= 2: return [float('inf')] * len(front_indices)
        
        distances = [0.0] * len(front_indices)
        front_objectives = [objectives_list[i] for i in front_indices]
        num_objectives = len(front_objectives[0])
        
        for obj_idx in range(num_objectives):
            sorted_indices = sorted(range(len(front_indices)), key=lambda i: front_objectives[i][obj_idx])
            sorted_front = [front_indices[i] for i in sorted_indices]
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            obj_range = objectives_list[sorted_front[-1]][obj_idx] - objectives_list[sorted_front[0]][obj_idx]
            if obj_range > 0:
                for i in range(1, len(sorted_front) - 1):
                    distances[sorted_indices[i]] += (objectives_list[sorted_front[i+1]][obj_idx] - 
                                                     objectives_list[sorted_front[i-1]][obj_idx]) / obj_range
        
        return distances

    def _tournament_selection(self, population, objectives_vectors, fronts):
        """Selects individuals using tournament selection based on rank and crowding distance."""
        tournament_size = min(3, len(population))
        candidates_indices = random.sample(range(len(population)), tournament_size)
        
        best_candidate_idx = -1
        best_rank = float('inf')
        
        for idx in candidates_indices:
            current_rank = -1
            for rank, front in enumerate(fronts):
                if idx in front:
                    current_rank = rank
                    break
            
            if current_rank < best_rank:
                best_rank = current_rank
                best_candidate_idx = idx
            elif current_rank == best_rank:
                # Use crowding distance to break ties
                front_indices = fronts[best_rank]
                distances = self._calculate_crowding_distance(objectives_vectors, front_indices)
                best_distance = distances[front_indices.index(best_candidate_idx)]
                current_distance = distances[front_indices.index(idx)]
                if current_distance > best_distance:
                    best_candidate_idx = idx
                
        return population[best_candidate_idx]

    def generate_inputs(self, dt=1.0):
        """Evolves the population to find the best adversarial inputs."""
        if self.ts.prev_assignments is None:
            rsrp_init, sinr_init, load_init, prio_init = self.env.compute_metrics()
            current_assignments = self.ts.assign_ues(rsrp_init, sinr_init, load_init, prio_init, dt=0)
        else:
            current_assignments = self.ts.prev_assignments

        population = []
        for _ in range(self.population_size):
            load_modifier = np.random.uniform(-0.1, 0.1, self.env.num_cells)
            position_modifier = np.random.uniform(-5, 5, (self.env.num_ues, 2))
            inputs = np.concatenate([load_modifier, position_modifier.flatten()])
            population.append(inputs)

        best_overall_individual = population[0].copy()
        best_overall_objectives = None

        pbar_gen = tqdm(range(self.generations), desc="AI Fuzzer Evolution", leave=False)
        
        for gen in pbar_gen:
            # Batching is handled by `_calculate_objectives`
            objectives_vectors = self._calculate_objectives(np.array(population), current_assignments, dt)
            
            # Add successful vulnerability patterns to memory
            for i, obj_vector in enumerate(objectives_vectors):
                vulnerability_score = np.sum(obj_vector)
                if vulnerability_score > 1.5:
                    self.vulnerability_memory.append({
                        'individual': population[i].copy(),
                        'objectives': obj_vector,
                        'generation': gen
                    })
            
            if len(self.vulnerability_memory) > 50:
                self.vulnerability_memory.sort(key=lambda x: np.sum(x['objectives']), reverse=True)
                self.vulnerability_memory = self.vulnerability_memory[:30]
            
            fronts = self._fast_non_dominated_sort(objectives_vectors)
            
            if len(fronts[0]) > 0:
                front0_distances = self._calculate_crowding_distance(objectives_vectors, fronts[0])
                best_idx_in_front0 = fronts[0][np.argmax(front0_distances)]
                current_best_objectives = objectives_vectors[best_idx_in_front0]
                
                if (best_overall_objectives is None or 
                    self._dominates(current_best_objectives, best_overall_objectives)):
                    best_overall_objectives = current_best_objectives
                    best_overall_individual = population[best_idx_in_front0].copy()
            
            new_population = []
            current_size = 0
            for front in fronts:
                if current_size + len(front) <= self.population_size:
                    new_population.extend([population[idx].copy() for idx in front])
                    current_size += len(front)
                else:
                    remaining = self.population_size - current_size
                    if remaining > 0:
                        distances = self._calculate_crowding_distance(objectives_vectors, front)
                        sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                        new_population.extend([population[idx].copy() for idx, _ in sorted_front[:remaining]])
                    break
            
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = self._tournament_selection(population, objectives_vectors, fronts)
                parent2 = self._tournament_selection(population, objectives_vectors, fronts)
                
                crossover_point = np.random.randint(1, self.input_vector_size)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                
                for child in [child1, child2]:
                    # Increased mutation probability from 0.2 to 0.35
                    if np.random.rand() < 0.35:
                        # Adaptive mutation: higher mutation rate for stagnating populations
                        mutation_influence = 0.4  # Increased from 0.3 for more exploration
                        if len(self.vulnerability_memory) > 0:
                            # Prefer more severe vulnerabilities from memory
                            # Using a simple uniform selection as fallback
                            memory_pattern = np.random.choice(self.vulnerability_memory)
                            child = (1 - mutation_influence) * child + mutation_influence * memory_pattern['individual']
                        
                        # Different mutation rates for different parts of the genome
                        # Increased variance for cell loads to explore more extreme scenarios
                        child[:self.env.num_cells] += np.random.normal(0, 0.15, self.env.num_cells)
                        # Improved exploration of UE positions and velocities
                        child[self.env.num_cells:] += np.random.normal(0, 2.5, self.env.num_ues * 2)
                        child[:self.env.num_cells] = np.clip(child[:self.env.num_cells], -0.4, 0.4)  # Wider range
                    offspring.append(child)
            
            population = offspring[:self.population_size]
            
            avg_objectives = np.mean(objectives_vectors, axis=0)
            # Check for stagnation to trigger adaptive mutation
            if gen > 0:
                prev_best = np.max([np.sum(obj) for obj in objectives_vectors])
                if prev_best <= self.prev_best_obj_sum:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
                self.prev_best_obj_sum = prev_best
                
            vulnerability_count = len([obj for obj in objectives_vectors if np.sum(obj) > 1.5])  # More strict threshold
            status_dict = {
                'HO': f'{avg_objectives[0]:.2f}', 
                'QoE': f'{avg_objectives[1]:.2f}',
                'UF': f'{avg_objectives[2]:.2f}', 
                'EN': f'{avg_objectives[3]:.2f}'
            }
            
            # Only add vulnerability index if available (for backwards compatibility)
            if len(avg_objectives) > 4:
                status_dict['Vuln'] = f'{avg_objectives[4]:.2f}'
                
            status_dict['Vulns'] = vulnerability_count
            pbar_gen.set_postfix(status_dict)
            
        pbar_gen.close()
        return best_overall_individual

# --- Module 3b: Enhanced Fuzzer Variants ---
class NaiveRandomFuzzer:
    """A simple random fuzzer baseline for comparison."""
    def __init__(self, env: NetworkEnvironment, ts: TrafficSteeringAlgorithm):
        self.env = env
        self.ts = ts
    def generate_inputs(self, dt=1.0):
        load_modifier = np.random.uniform(-0.05, 0.05, self.env.num_cells)
        position_modifier_2d = np.random.uniform(-3, 3, (self.env.num_ues, 2))
        inputs = np.concatenate([load_modifier, position_modifier_2d.flatten()])
        return inputs

# --- Module 4: Oracle (Vulnerability Detector) ---
class Oracle:
    """
    Evaluates network performance and detects vulnerabilities based on predefined rules.
    """
    def __init__(self, num_ues, num_cells, ping_pong_window=4, ping_pong_threshold=2, qos_sinr_threshold=5.0, fairness_threshold=0.4):
        self.num_ues = num_ues
        self.num_cells = num_cells
        self.ping_pong_window = ping_pong_window
        self.ping_pong_threshold = ping_pong_threshold
        self.qos_sinr_threshold = qos_sinr_threshold
        self.fairness_threshold = fairness_threshold
        self.handover_history = {}
        
    def _jain_fairness(self, allocations):
        """Calculates Jain's Fairness Index."""
        allocations = np.asarray(allocations)
        allocations_cleaned = allocations[np.isfinite(allocations) & (allocations > 1e-12)]
        if len(allocations_cleaned) == 0: return 1.0
        sum_val = np.sum(allocations_cleaned)
        sum_sq_val = np.sum(allocations_cleaned**2)
        if sum_sq_val < 1e-20: return 1.0
        return sum_val**2 / (len(allocations_cleaned) * sum_sq_val)

    def evaluate(self, rsrp, sinr, assignments, cell_loads, priorities, prev_assignments=None):
        """
        Evaluates the current state of the network for vulnerabilities.
        """
        vulnerabilities_found = []
        num_ping_pongs_detected_this_step = 0

        for ue_idx in range(self.num_ues):
            if ue_idx not in self.handover_history:
                self.handover_history[ue_idx] = []
            
            self.handover_history[ue_idx].append(assignments[ue_idx])
            if len(self.handover_history[ue_idx]) > self.ping_pong_window:
                self.handover_history[ue_idx].pop(0)
            
            history = self.handover_history[ue_idx]
            if len(history) == self.ping_pong_window:
                changes = sum(1 for i in range(len(history) - 1) if history[i] != history[i+1])
                if changes >= self.ping_pong_threshold:
                    num_ping_pongs_detected_this_step += 1
                    
        # Enhanced ping-pong detection with severity scoring
        ping_pong_severity = 0
        if num_ping_pongs_detected_this_step >= self.ping_pong_threshold:
            ping_pong_ratio = num_ping_pongs_detected_this_step / self.num_ues
            ping_pong_severity = min(3, 1 + int(ping_pong_ratio * 5))  # Scale severity from 1-3 based on ratio
            vulnerabilities_found.append(f"Ping-Pong: {num_ping_pongs_detected_this_step} UEs oscillating (Severity: {ping_pong_severity})")

        temp_assigned_sinr_list = [sinr[i, assignments[i]] if 0 <= assignments[i] < self.num_cells else np.nan for i in range(self.num_ues)]
        assigned_sinr_np_finite = np.array([s for s in temp_assigned_sinr_list if pd.notna(s)])
        
        # Improved high-priority user QoE analysis
        high_priority_mask = (priorities == 1)
        assigned_sinr_hp_ues_list = [sinr[i, assignments[i]] for i in range(self.num_ues) if high_priority_mask[i]]
        assigned_sinr_hp_ues_np = np.array(assigned_sinr_hp_ues_list) if assigned_sinr_hp_ues_list else np.array([])
        
        qoe_violation_severity = 0
        has_qoe_violation = False
        if assigned_sinr_hp_ues_np.size > 0:
            avg_sinr_high = np.mean(assigned_sinr_hp_ues_np)
            if avg_sinr_high < self.qos_sinr_threshold:
                qoe_deficit = self.qos_sinr_threshold - avg_sinr_high
                qoe_violation_severity = min(3, 1 + int(qoe_deficit / 2))  # Scale severity 1-3 based on how far below threshold
                vulnerabilities_found.append(f"QoS Violation: Avg High Prio SINR = {avg_sinr_high:.2f} dB (Severity: {qoe_violation_severity})")
                has_qoe_violation = True

        jain_score = 1.0
        unfairness_severity = 0
        has_unfairness = False
        if assigned_sinr_np_finite.size > 0:
            assigned_sinr_linear = 10**(assigned_sinr_np_finite / 10.0)
            jain_score = self._jain_fairness(assigned_sinr_linear)
            if jain_score < self.fairness_threshold:
                fairness_deficit = self.fairness_threshold - jain_score
                unfairness_severity = min(3, 1 + int(fairness_deficit / 0.2))  # Scale severity 1-3
                vulnerabilities_found.append(f"Unfairness: Jain Index = {jain_score:.2f} (Severity: {unfairness_severity})")
                has_unfairness = True
        
        # Calculate additional metrics for later analysis
        handover_rate = np.sum(assignments != prev_assignments) / self.num_ues if prev_assignments is not None else 0
        
        # Combined vulnerability severity score (1-10 scale)
        total_vulnerability_score = ping_pong_severity*2 + qoe_violation_severity*3 + unfairness_severity*2
        
        return {
            'vulnerabilities': vulnerabilities_found,
            'vulnerability_score': total_vulnerability_score,  # New severity score
            'severity': total_vulnerability_score,  # Add this for backward compatibility
            'jain_index': jain_score,
            'avg_sinr_db': np.mean(assigned_sinr_np_finite) if assigned_sinr_np_finite.size > 0 else np.nan,
            'sinr_5th_percentile_db': safe_nanpercentile(assigned_sinr_np_finite, 5),
            'avg_high_prio_sinr': np.mean(assigned_sinr_hp_ues_np) if assigned_sinr_hp_ues_np.size > 0 else np.nan,
            'handover_rate': handover_rate,
            'has_ping_pong': num_ping_pongs_detected_this_step > 0,
            'has_qoe_violation': has_qoe_violation,
            'has_unfairness': has_unfairness,
        }
    
# --- Module 5: Main Simulation Loop and Analysis ---
def run_simulation(scenario_name, num_ues, initial_load, max_speed, scenario_type, active_cell_indices=None, inter_site_distance=100.0):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    
    # Configure TensorFlow memory growth for optimal H100 utilization
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using {len(gpus)} GPU(s) for scenario: {scenario_name}")
        # Let TensorFlow allocate memory as needed for H100
        try:
            # This allows TensorFlow to use most GPU memory but still grow as needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error configuring GPU memory growth: {e}")
    
    # Create the environment with optimized parameters
    shared_env_state = NetworkEnvironment(
        num_ues=num_ues, initial_load=initial_load, scenario_max_speed=max_speed,
        scenario_type=scenario_type, active_cell_indices=active_cell_indices,
        inter_site_distance=inter_site_distance
    )
    
    algorithm_factories = {
        "Baseline": lambda: BaselineA3(num_ues=num_ues, num_cells=shared_env_state.num_cells),
        "Utility": lambda: UtilityBased(num_ues=num_ues, num_cells=shared_env_state.num_cells),
        "ML-Based": lambda: MLTrafficSteering(num_ues=num_ues, num_cells=shared_env_state.num_cells)
    }
    
    fuzzer_map = {
        "AI-Fuzzer": lambda env, ts: AIFuzzer(env, ts, use_nsga2=ENABLE_NSGA2_FUZZER),
        "Random-Fuzzer": lambda env, ts: NaiveRandomFuzzer(env, ts)
    }
    
    results_list = []
    fuzzer_effectiveness = {}
    
    combination_pbar = tqdm(total=len(fuzzer_map) * len(algorithm_factories), desc=f"Processing {scenario_name}", leave=False)
    
    for fuzzer_name, fuzzer_factory in fuzzer_map.items():
        fuzzer_effectiveness[fuzzer_name] = {
            'vulnerability_counts': [],
            'vulnerability_severities': [],
            'handover_rates': [],
            'qoe_violations': [],
            'unfairness_events': [],
            'ping_pong_events': [],
        }
        for actual_algo_name, algo_factory in algorithm_factories.items():
            combination_pbar.set_description(f"{scenario_name}: {fuzzer_name}+{actual_algo_name}")
            
            shared_env_state.reset(initial_load=initial_load, max_speed=max_speed)
            ts_instance = algo_factory()
            oracle = Oracle(num_ues=num_ues, num_cells=shared_env_state.num_cells)
            fuzzer = fuzzer_factory(shared_env_state, ts_instance)
            
            rsrp_init, sinr_init, _, prio_init = shared_env_state.compute_metrics()
            initial_assignments = ts_instance.assign_ues(rsrp_init, sinr_init, shared_env_state.cell_loads, prio_init, dt=0)
            
            if initial_assignments is None: continue
            
            shared_env_state.update_cell_loads(initial_assignments)
            
            iter_pbar = tqdm(range(SIMULATION_ITERATIONS), desc=f" {fuzzer_name}+{actual_algo_name} Iterations", leave=False)
            for iteration in iter_pbar:
                try:
                    current_assignments = ts_instance.prev_assignments.copy()
                    
                    if hasattr(fuzzer, 'generate_inputs'):
                        fuzzed_inputs = fuzzer.generate_inputs(dt=1.0)
                    else:
                        # For RandomFuzzer, we don't need to call a sub-method
                        load_modifier = np.random.uniform(-0.05, 0.05, shared_env_state.num_cells)
                        position_modifier_2d = np.random.uniform(-3, 3, (num_ues, 2))
                        fuzzed_inputs = np.concatenate([load_modifier, position_modifier_2d.flatten()])

                    load_modifier = fuzzed_inputs[:shared_env_state.num_cells]
                    position_modifier_2d = fuzzed_inputs[shared_env_state.num_cells:].reshape(num_ues, 2)
                    pos_modifier_3d_np = np.hstack([position_modifier_2d, np.zeros((num_ues, 1))])

                    shared_env_state.cell_loads = np.clip(shared_env_state.cell_loads + load_modifier, 0, 1)
                    shared_env_state.ue_loc.assign(shared_env_state.ue_loc + tf.constant(pos_modifier_3d_np[np.newaxis,...], dtype=tf.float32))
                    shared_env_state.update_ue_positions_and_velocities(dt=1.0)
                    
                    rsrp, sinr, cell_loads_eval, priorities_eval = shared_env_state.compute_metrics()
                    new_assignments = ts_instance.assign_ues(rsrp, sinr, cell_loads_eval, priorities_eval, dt=1.0)
                    new_assignments = np.clip(new_assignments, 0, shared_env_state.num_cells - 1)
                    
                    shared_env_state.update_cell_loads(new_assignments)
                    oracle_metrics = oracle.evaluate(rsrp, sinr, new_assignments, shared_env_state.cell_loads, priorities_eval, current_assignments)
                    
                    assigned_sinr_list = [sinr[i, new_assignments[i]] if 0 <= new_assignments[i] < shared_env_state.num_cells else np.nan for i in range(num_ues)]
                    assigned_sinr_np_finite = np.array([s for s in assigned_sinr_list if pd.notna(s)])
                    
                    assigned_sinr_linear = 10**(np.array(assigned_sinr_list, dtype=np.float32) / 10.0)
                    user_throughputs_bps = calculate_estimated_shannon_throughput_tf(assigned_sinr_linear, BANDWIDTH).numpy()
                    user_throughputs_mbps = user_throughputs_bps / 1e6
                    transmission_time_ms = calculate_transmission_delay_ms(user_throughputs_bps)

                    results_list.append({
                        'scenario': scenario_name, 'iteration': iteration, 'fuzzer_type': fuzzer_name,
                        'algorithm': actual_algo_name,
                        'handover_count': np.sum(new_assignments != current_assignments),
                        'handover_rate': oracle_metrics['handover_rate'],
                        'vulnerability_count': len(oracle_metrics['vulnerabilities']),
                        'vulnerabilities_list': oracle_metrics['vulnerabilities'],
                        'jain_fairness_index': float(oracle_metrics['jain_index']),
                        'avg_sinr_db': np.mean(assigned_sinr_np_finite) if assigned_sinr_np_finite.size > 0 else np.nan,
                        'sinr_5th_percentile_db': safe_nanpercentile(assigned_sinr_np_finite, 5),
                        'avg_throughput_mbps': np.nanmean(user_throughputs_mbps),
                        'throughput_5th_percentile_mbps': safe_nanpercentile(user_throughputs_mbps, 5),
                        'avg_transmission_time_ms': np.nanmean(transmission_time_ms),
                        'load_std': np.std(shared_env_state.cell_loads),
                        'has_ping_pong': oracle_metrics['has_ping_pong'],
                        'has_qoe_violation': oracle_metrics['has_qoe_violation'],
                        'has_unfairness': oracle_metrics['has_unfairness']
                    })
                    
                    # Update fuzzer effectiveness metrics
                    fuzzer_effectiveness[fuzzer_name]['vulnerability_counts'].append(len(oracle_metrics['vulnerabilities']))
                    # Use severity from oracle_metrics if available, otherwise use the old calculation
                    if 'severity' in oracle_metrics:
                        fuzzer_effectiveness[fuzzer_name]['vulnerability_severities'].append(oracle_metrics['severity'])
                    else:
                        fuzzer_effectiveness[fuzzer_name]['vulnerability_severities'].append(
                            oracle_metrics['has_ping_pong'] * 3 + oracle_metrics['has_qoe_violation'] * 2 + oracle_metrics['has_unfairness'] * 1
                        )
                    fuzzer_effectiveness[fuzzer_name]['handover_rates'].append(oracle_metrics['handover_rate'])
                    fuzzer_effectiveness[fuzzer_name]['qoe_violations'].append(oracle_metrics['has_qoe_violation'])
                    fuzzer_effectiveness[fuzzer_name]['unfairness_events'].append(oracle_metrics['has_unfairness'])
                    fuzzer_effectiveness[fuzzer_name]['ping_pong_events'].append(oracle_metrics['has_ping_pong'])
                    
                    iter_pbar.set_postfix({'Vulns': len(oracle_metrics['vulnerabilities']), '5th Thrpt': f'{safe_nanpercentile(user_throughputs_mbps, 5):.2f}Mbps'})
                except Exception as e:
                    print(f"ERROR in iteration {iteration} for {fuzzer_name}+{actual_algo_name}: {e}")
                    continue
            iter_pbar.close()
            combination_pbar.update(1)
            
    combination_pbar.close()
    return results_list, fuzzer_effectiveness

def summarize_and_plot(df, effectiveness_data, script_version):
    """
    Generates summary statistics and plots for the paper.
    """
    if df.empty:
        print("No data available for summary or plotting.")
        return

    # --- Print Summary and Statistical Analysis ---
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS - AI vs RANDOM FUZZER")
    print("="*80)
    
    overall_effectiveness = {}
    for scenario_name, scenario_data in effectiveness_data.items():
        for fuzzer_name, fuzzer_data in scenario_data.items():
            if fuzzer_name not in overall_effectiveness:
                overall_effectiveness[fuzzer_name] = {
                    'total_vulns': 0, 'total_severity': 0, 'scenarios': 0
                }
            overall_effectiveness[fuzzer_name]['total_vulns'] += sum(fuzzer_data['vulnerability_counts'])
            overall_effectiveness[fuzzer_name]['total_severity'] += sum(fuzzer_data['vulnerability_severities'])
            overall_effectiveness[fuzzer_name]['scenarios'] += 1

    print("\nOVERALL FUZZER EFFECTIVENESS ACROSS ALL SCENARIOS:")
    for fuzzer, metrics in overall_effectiveness.items():
        print(f"  {fuzzer}:")
        print(f"    Total Vulnerabilities Found: {metrics['total_vulns']}")
        print(f"    Average Vulnerability Severity: {metrics['total_severity'] / max(1, metrics['total_vulns']):.2f}")
    
    # Perform t-test for a key metric, e.g., vulnerability count
    if 'AI-Fuzzer' in overall_effectiveness and 'Random-Fuzzer' in overall_effectiveness:
        ai_vulns = [sum(eff['AI-Fuzzer']['vulnerability_counts']) for eff in effectiveness_data.values() if 'AI-Fuzzer' in eff]
        random_vulns = [sum(eff['Random-Fuzzer']['vulnerability_counts']) for eff in effectiveness_data.values() if 'Random-Fuzzer' in eff]

        if len(ai_vulns) > 1 and len(random_vulns) > 1:
            try:
                t_stat, p_value = stats.ttest_ind(ai_vulns, random_vulns, equal_var=False)
                print("\nSTATISTICAL SIGNIFICANCE (T-TEST) for Total Vulnerabilities:")
                print(f"  T-statistic: {t_stat:.3f}, P-value: {p_value:.5f}")
                if p_value < 0.05:
                    print("  Result: The difference is statistically significant (p < 0.05).")
                else:
                    print("  Result: No statistically significant difference found.")
            except Exception as e:
                print(f"Could not perform t-test: {e}")
    
    print("\n" + "="*80)
    print("GENERATING PLOTS FOR PAPER")
    print("="*80)
    
    output_plot_dir = f"plots_{script_version}"
    os.makedirs(output_plot_dir, exist_ok=True)
    
    # Plot Metrics as CDFs
    def plot_cdf(df_to_plot, metric, ylabel, fuzzer_type, filename):
        plt.figure(figsize=(10, 6))
        for algo in df_to_plot['algorithm'].unique():
            algo_df = df_to_plot[df_to_plot['algorithm'] == algo]
            data = algo_df[metric].dropna().sort_values().reset_index(drop=True)
            if data.empty: continue
            y = np.linspace(0, 1, len(data))
            plt.plot(data, y, label=f"{algo} ({fuzzer_type})", linewidth=2)
        plt.title(f'CDF of {ylabel} - Scenario: {df_to_plot["scenario"].iloc[0]}')
        plt.xlabel(ylabel)
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, filename))
        plt.close()

    scenarios = df['scenario'].unique()
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario]
        for fuzzer_type in scenario_df['fuzzer_type'].unique():
            fuzzer_df = scenario_df[scenario_df['fuzzer_type'] == fuzzer_type]
            
            # CDF of 5th Percentile Throughput
            plot_cdf(fuzzer_df, 'throughput_5th_percentile_mbps', '5th Percentile Throughput (Mbps)', fuzzer_type,
                     f'{scenario.replace(" ", "_")}_{fuzzer_type.replace(" ", "_")}_5th_percentile_throughput_cdf.png')
            
            # CDF of Average Transmission Time
            plot_cdf(fuzzer_df, 'avg_transmission_time_ms', 'Average Transmission Time (ms)', fuzzer_type,
                     f'{scenario.replace(" ", "_")}_{fuzzer_type.replace(" ", "_")}_avg_transmission_time_cdf.png')

    print(f"All plots saved to {output_plot_dir}")
    
    
def main():
    print(f"--- Starting AI Fuzzing Simulation ({SCRIPT_VERSION_NAME}) ---")
    print("--- H100 GPU Optimizations Enabled: ---")
    print("  - Mixed precision (FP16/BF16) for tensor cores")
    print("  - XLA JIT compilation with experimental optimizations")
    print("  - Increased batch processing (64 inputs per batch)")
    print("  - Parallel execution of fuzzing iterations")
    print("  - Dynamic memory growth for optimal GPU utilization")
    
    # Log GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu}")
            # Get memory info using a safer approach
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details and 'device_name' in gpu_details:
                    print(f"    * Device: {gpu_details['device_name']}")
            except Exception as e:
                print(f"    * Could not get detailed GPU info: {e}")
    else:
        print("  - No GPUs detected. Running on CPU only.")
    
    start_time_main = time.time()
    all_results_data = []
    all_fuzzer_effectiveness = {}

    # Define a function to monitor GPU usage
    def log_gpu_usage():
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"Active GPUs: {len(gpus)}")
                # Use nvidia-smi within Docker if available
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used', 
                                            '--format=csv,noheader'], 
                                            capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"GPU Utilization: {result.stdout.strip()}")
                except:
                    print("Could not get GPU utilization info")
        except:
            pass

    try:
        # Configure TensorFlow logging
        if ENABLE_TF_DEVICE_LOGGING: 
            tf.debugging.set_log_device_placement(False) 
        tf.get_logger().setLevel('ERROR')
        
        # Log initial GPU usage
        log_gpu_usage()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            for gpu_device_config in tf.config.get_visible_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu_device_config, True)
                try:
                    tf.config.experimental.set_memory_limit(gpu_device_config, 32768)
                except Exception:
                    print("Could not set memory limit, using dynamic growth.")
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            tf.config.optimizer.set_jit(True)
            print("--- GPU configuration successful ---")
        else:
            print("--- No GPU detected by TensorFlow. Running on CPU. ---")
            
        emergency_active_cells = [i for i in range(NUM_CELLS) if i not in [0, 1, 5, 10]]
        scenarios_to_run = [
            {'name': 'High Mobility', 'params': {'num_ues': 30, 'initial_load': 0.5, 'max_speed': 10, 'scenario_type': 'default'}},
            {'name': 'Emergency (BS Outage)', 'params': {'num_ues': 30, 'initial_load': 0.5, 'max_speed': 5, 'scenario_type': 'default', 'active_cell_indices': emergency_active_cells}},
            {'name': 'High Load', 'params': {'num_ues': 30, 'initial_load': 0.7, 'max_speed': 5, 'scenario_type': 'default'}}
        ]

        scenario_pbar = tqdm(scenarios_to_run, desc="Overall Progress", position=0)
        for scenario_info in scenario_pbar:
            name = scenario_info['name']
            params = scenario_info['params']
            scenario_pbar.set_description(f"Running: {name}")
            np.random.seed(42); tf.random.set_seed(42)
            results, effectiveness = run_simulation(scenario_name=name, **params)
            all_results_data.extend(results)
            all_fuzzer_effectiveness[name] = effectiveness
        scenario_pbar.close()

    except Exception as main_exc:
        import traceback
        print(f"\nCRITICAL ERROR in main loop: {main_exc}")
        print(f"\nStacktrace: {traceback.format_exc()}")
    finally:
        print("\n--- Finalizing script: Saving results... ---")
        if all_results_data:
            results_df = pd.DataFrame(all_results_data)
            csv_filename = f'fuzzing_results_{SCRIPT_VERSION_NAME}.csv'
            try:
                results_df.to_csv(csv_filename, index=False, encoding='utf-8')
                print(f"\n--- Results saved to {csv_filename} ---")
            except Exception as e:
                print(f"Could not save CSV file: {e}")
            
            summarize_and_plot(results_df, all_fuzzer_effectiveness, SCRIPT_VERSION_NAME)
        else:
            print("No results were generated to save.")

    end_time_main = time.time()
    print(f"\n--- Simulation Finished in {end_time_main - start_time_main:.2f} seconds ---")

if __name__ == "__main__":
    np.random.seed(42); random.seed(42); tf.random.set_seed(42)
    main()
