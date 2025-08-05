# -*- coding: utf-8 -*-
# Combined AI Fuzzing Script for O-RAN Traffic Steering Vulnerability Analysis
# Version 25.2: Reverted to Single-GPU, kept Mixed-Mobility scenario and bug fixes.

# --- Imports ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sionna
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from collections import Counter
from tqdm import tqdm

# --- Sionna specific imports for channel modeling ---
from sionna.phy.channel.tr38901 import UMi, PanelArray, Antenna
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import GenerateOFDMChannel


# --- Global Constants ---
NUM_CELLS = 19
NUM_UES = 30
BANDWIDTH = 13.68e6
CARRIER_FREQUENCY = 3.5e9
TX_POWER_DBM = 30
NOISE_POWER_DBM_PER_HZ = -174

SIMULATION_ITERATIONS = 200
FUZZER_GENERATIONS = 100
FUZZER_POPULATION = 10

ENABLE_DETAILED_METRIC_PRINT = False
ENABLE_TF_DEVICE_LOGGING = False

SCRIPT_VERSION_NAME = "v25_2_single_gpu_fix" # For output files

def calculate_throughput(sinr_linear_arr, bandwidth_hz):
    """
    Calculates the theoretical maximum throughput using the Shannon-Hartley theorem.
    """
    # Ensure SINR values are positive
    positive_sinr = np.maximum(sinr_linear_arr, 1e-9)
    # C = B * log2(1 + SINR)
    throughput_bps = bandwidth_hz * np.log2(1 + positive_sinr)
    return throughput_bps

# --- Module 1: Network Simulation Environment ---
class NetworkEnvironment:
    def __init__(self, initial_load=0.3, scenario_max_speed=5, scenario_type='default'):
        self.batch_size = 1
        self.num_cells = NUM_CELLS  # Use the global constant
        self.initial_load_param = initial_load
        self.max_speed_param = scenario_max_speed
        self.scenario_type = scenario_type
        self.ue_mobility_types = np.full(NUM_UES, 'mobile', dtype=object)

        self.ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                                   polarization='single', polarization_type='V',
                                   antenna_pattern='omni', carrier_frequency=CARRIER_FREQUENCY,
                                   precision="single")
        self.bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                                   polarization='single', polarization_type='V',
                                   antenna_pattern='omni', carrier_frequency=CARRIER_FREQUENCY,
                                   precision="single")

        try:
            self.channel_model_3gpp = UMi(
                carrier_frequency=CARRIER_FREQUENCY, o2i_model='low',
                ut_array=self.ut_array, bs_array=self.bs_array,
                direction='downlink',
                enable_pathloss=True, enable_shadow_fading=True,
                always_generate_lsp=True,
                precision="single"
            )
        except Exception as e:
            print(f"CRITICAL ERROR instantiating Sionna UMi model: {e}")
            raise

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=512,
            subcarrier_spacing=30e3,
            num_tx=self.num_cells,  # Use the new class variable
            num_streams_per_tx=1,
            cyclic_prefix_length=20,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
            num_guard_carriers=(28, 27),
            dc_null=True
        )

        self.generate_h_freq_layer = GenerateOFDMChannel(
            channel_model=self.channel_model_3gpp,
            resource_grid=self.resource_grid,
            precision="single"
        )

        # --- Modified Section ---
        # Instead of fixed coordinates, we use the new function to generate the layout
        inter_site_distance = 100.0  # Distance between BS sites in meters
        self.bs_pos_2d = self._generate_hexagonal_layout(self.num_cells, inter_site_distance)
        self.bs_loc = tf.constant(np.hstack([self.bs_pos_2d, np.ones((self.num_cells, 1)) * 10.0])[np.newaxis,...], dtype=tf.float32)
        # --- End of Modified Section ---

        self.ue_loc = tf.Variable(tf.zeros([self.batch_size, NUM_UES, 3], dtype=tf.float32), name="ue_loc")
        self.ue_velocities = tf.Variable(tf.zeros([self.batch_size, NUM_UES, 3], dtype=tf.float32), name="ue_velocities")

        self.ut_orientations = tf.zeros([self.batch_size, NUM_UES, 3], dtype=tf.float32)
        self.bs_orientations = tf.zeros([self.batch_size, self.num_cells, 3], dtype=tf.float32)
        self.in_state = tf.zeros([self.batch_size, NUM_UES], dtype=tf.bool)

        self.cell_loads = np.ones(self.num_cells) * initial_load
        self.ue_priorities = np.random.choice([1, 2, 3], size=NUM_UES, p=[0.3, 0.4, 0.3]).astype(np.float32)

        self.noise_power_watts = 10**((NOISE_POWER_DBM_PER_HZ - 30) / 10) * BANDWIDTH
        self.tx_power_watts_total = 10**((TX_POWER_DBM - 30) / 10)

        self.reset(initial_load, scenario_max_speed)

    def _generate_hexagonal_layout(self, num_cells, distance):
        """
        This function generates the x and y coordinates for a hexagonal grid layout.
        """
        if num_cells == 1:
            return np.array([[0.0, 0.0]])

        coords = [(0.0, 0.0)]
        # Axial directions for movement in a hexagonal grid.
        # These are vectors used to move from one cell to its neighbor.
        axial_directions = [
            (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)
        ]
        
        # Using axial coordinates (q, r) for simpler hex grid logic
        axial_coords = [(0, 0)]
        seen_coords = set([(0, 0)])

        ring = 1
        while len(axial_coords) < num_cells:
            # Start at a corner of the new ring
            current_axial = (ring, -ring)
            if current_axial not in seen_coords:
                 axial_coords.append(current_axial)
                 seen_coords.add(current_axial)

            # Traverse the 6 sides of the hexagon ring
            for i in range(6):
                for _ in range(ring):
                    if len(axial_coords) >= num_cells: break
                    # Move one step in the current side's direction
                    current_axial = (current_axial[0] + axial_directions[i][0], current_axial[1] + axial_directions[i][1])
                    if current_axial not in seen_coords:
                        axial_coords.append(current_axial)
                        seen_coords.add(current_axial)
                if len(axial_coords) >= num_cells: break
            ring += 1
            
        # Convert axial coordinates to cartesian (x, y) coordinates
        cartesian_coords = []
        for q, r in axial_coords:
            x = distance * (3./2. * q)
            y = distance * (np.sqrt(3)/2. * q + np.sqrt(3) * r)
            cartesian_coords.append((x, y))

        return np.array(cartesian_coords[:num_cells])

    def reset(self, initial_load, max_speed):
        self.initial_load_param = initial_load
        self.max_speed_param = max_speed
        
        # Center the random UE positions around the mean of the BS locations
        center_x = np.mean(self.bs_pos_2d[:, 0])
        center_y = np.mean(self.bs_pos_2d[:, 1])
        max_dist = np.max(np.linalg.norm(self.bs_pos_2d, axis=1)) + 50 # Area slightly larger than the cell layout

        ue_pos_2d_np = np.random.uniform(-max_dist, max_dist, size=(NUM_UES, 2)) + np.array([center_x, center_y])
        
        self.ue_loc.assign(np.hstack([ue_pos_2d_np, np.ones((NUM_UES, 1)) * 1.5])[np.newaxis,...])
        ue_vel_2d_np = np.random.uniform(-max_speed, max_speed, size=(NUM_UES, 2))
        self.ue_velocities.assign(np.hstack([ue_vel_2d_np, np.zeros((NUM_UES, 1))])[np.newaxis,...])
        self.cell_loads = np.ones(self.num_cells) * initial_load

        if self.scenario_type == 'mixed':
            mobile_mask = np.random.rand(NUM_UES) > 0.5
            self.ue_mobility_types = np.where(mobile_mask, 'mobile', 'static')
            current_velocities = self.ue_velocities.numpy()
            static_ue_indices = np.where(self.ue_mobility_types == 'static')[0]
            if static_ue_indices.size > 0:
                current_velocities[0, static_ue_indices, :] = 0.0
                self.ue_velocities.assign(current_velocities)
            print(f"Mixed Mobility: {np.sum(self.ue_mobility_types == 'static')} static UEs, {np.sum(self.ue_mobility_types == 'mobile')} mobile UEs.")
        else:
            self.ue_mobility_types.fill('mobile')

    def update_ue_positions_and_velocities(self, dt=1.0, max_speed=None):
        if max_speed is None: max_speed = self.max_speed_param
        mobile_mask = tf.constant(self.ue_mobility_types == 'mobile', dtype=tf.float32)
        mobile_mask_3d = tf.reshape(mobile_mask, (1, NUM_UES, 1))

        velocity_updates = tf.random.normal(shape=self.ue_velocities.shape, stddev=1.0, dtype=tf.float32) * dt
        new_velocities = self.ue_velocities + (velocity_updates * mobile_mask_3d)

        speeds = tf.norm(new_velocities, axis=2, keepdims=True)
        safe_speeds = tf.where(speeds < 1e-9, tf.ones_like(speeds) * 1e-9, speeds)
        scale = tf.minimum(1.0, max_speed / safe_speeds)
        new_velocities = new_velocities * scale
        new_velocities = new_velocities * mobile_mask_3d

        new_velocities = tf.where(tf.math.is_finite(new_velocities), new_velocities, tf.zeros_like(new_velocities))
        self.ue_velocities.assign(new_velocities)

        new_loc = self.ue_loc + new_velocities * dt
        new_loc = tf.where(tf.math.is_finite(new_loc), new_loc, self.ue_loc)
        self.ue_loc.assign(new_loc)
    
    @tf.function(jit_compile=True)
    def compute_metrics_tf(self, ue_loc_tf, bs_loc_tf, ut_orient_tf, bs_orient_tf, ut_vel_tf, in_state_tf):
        self.channel_model_3gpp.set_topology(ut_loc=ue_loc_tf, bs_loc=bs_loc_tf, ut_orientations=ut_orient_tf, bs_orientations=bs_orient_tf, ut_velocities=ut_vel_tf, in_state=in_state_tf)
        h_freq = self.generate_h_freq_layer(batch_size=self.batch_size)
        
        h_freq_squeezed = tf.squeeze(h_freq, axis=[2, 4])
        avg_channel_power_gain = tf.reduce_mean(tf.abs(h_freq_squeezed)**2, axis=[-2, -1])
        received_power_watts_tf = self.tx_power_watts_total * avg_channel_power_gain

        MIN_POWER_WATTS = 10**((-140 - 30) / 10)
        MAX_POWER_WATTS = 10**((-40 - 30) / 10)
        received_power_watts_tf = tf.clip_by_value(received_power_watts_tf, MIN_POWER_WATTS, MAX_POWER_WATTS)
        received_power_watts_tf = tf.where(tf.math.is_finite(received_power_watts_tf), received_power_watts_tf, tf.zeros_like(received_power_watts_tf))

        rp_ue_cell = received_power_watts_tf[0]
        rsrp_db_tf = 10.0 * (tf.math.log(tf.maximum(rp_ue_cell / 1e-3, 1e-20)) / tf.math.log(10.0))

        signal_power_ue_cell = rp_ue_cell
        total_power_at_ue_u = tf.reduce_sum(rp_ue_cell, axis=1, keepdims=True)
        interference_ue_cell = total_power_at_ue_u - signal_power_ue_cell
        noise_ue_cell = self.noise_power_watts * tf.ones_like(signal_power_ue_cell)

        sinr_linear_tf = tf.math.divide_no_nan(signal_power_ue_cell, interference_ue_cell + noise_ue_cell)
        sinr_db_tf = 10.0 * (tf.math.log(tf.maximum(sinr_linear_tf, 1e-20)) / tf.math.log(10.0))

        rsrp_db_tf = tf.where(tf.math.is_finite(rsrp_db_tf), rsrp_db_tf, -200.0 * tf.ones_like(rsrp_db_tf))
        sinr_db_tf = tf.where(tf.math.is_finite(sinr_db_tf), sinr_db_tf, -30.0 * tf.ones_like(sinr_db_tf))

        return rsrp_db_tf, sinr_db_tf

    def compute_metrics(self):
        if self.channel_model_3gpp is None or self.resource_grid is None or not hasattr(self, 'generate_h_freq_layer'):
            return np.full((NUM_UES, self.num_cells), -200.0), np.full((NUM_UES, self.num_cells), -30.0), self.cell_loads.copy(), self.ue_priorities.copy()
        try:
            if not (tf.reduce_all(tf.math.is_finite(self.ue_loc)) and \
                    tf.reduce_all(tf.math.is_finite(self.bs_loc)) and \
                    tf.reduce_all(tf.math.is_finite(self.ue_velocities))):
                return np.full((NUM_UES, self.num_cells), -200.0), np.full((NUM_UES, self.num_cells), -30.0), self.cell_loads.copy(), self.ue_priorities.copy()
            
            rsrp_db_tf, sinr_db_tf = self.compute_metrics_tf(self.ue_loc, self.bs_loc, self.ut_orientations, self.bs_orientations, self.ue_velocities, self.in_state)
            return rsrp_db_tf.numpy(), sinr_db_tf.numpy(), self.cell_loads.copy(), self.ue_priorities.copy()
        except Exception as e:
            print(f"General Uncaught Error during Sionna UMi metric computation: {e}")
        return np.full((NUM_UES, self.num_cells), -200.0), np.full((NUM_UES, self.num_cells), -30.0), self.cell_loads.copy(), self.ue_priorities.copy()

    def update_cell_loads(self, assignments):
        self.cell_loads = np.zeros(self.num_cells)
        unique_cells, counts = np.unique(assignments, return_counts=True)
        load_per_ue = 1.0 / NUM_UES
        for cell_idx, count in zip(unique_cells, counts):
            if 0 <= cell_idx < self.num_cells:
                self.cell_loads[cell_idx] = count * load_per_ue
        self.cell_loads = np.clip(self.cell_loads, 0.0, 1.0)

# --- Module 2: Traffic Steering Algorithms ---
class TrafficSteering:
    def __init__(self, algorithm="baseline", rsrp_threshold=-100, hysteresis=3, ttt=0.1, load_threshold=0.8):
        self.algorithm = algorithm; self.rsrp_threshold = rsrp_threshold
        self.hysteresis = hysteresis; self.ttt = ttt; self.load_threshold = load_threshold
        self.prev_assignments = None; self.ttt_targets = {}
    def assign_initial(self, rsrp):
        self.prev_assignments = np.argmax(rsrp, axis=1); self.ttt_targets = {}
        return self.prev_assignments.copy()
    def baseline_a3(self, rsrp, sinr, cell_loads, priorities, dt):
        if self.prev_assignments is None: return self.assign_initial(rsrp)
        assignments = self.prev_assignments.copy()
        current_ttt_targets_state = {k: v.copy() for k, v in self.ttt_targets.items()}
        for ue_idx in range(NUM_UES):
            current_cell = assignments[ue_idx]; best_neighbor_quality = -np.inf; potential_target = -1
            if ue_idx not in current_ttt_targets_state: current_ttt_targets_state[ue_idx] = {}
            active_targets_for_ue = set()
            for cell_idx in range(NUM_CELLS):
                if cell_idx == current_cell: continue
                a3_cond = rsrp[ue_idx, cell_idx] > rsrp[ue_idx, current_cell] + self.hysteresis
                load_cond = cell_loads[cell_idx] < self.load_threshold
                rsrp_cond = rsrp[ue_idx, cell_idx] > self.rsrp_threshold
                if a3_cond and load_cond and rsrp_cond:
                    active_targets_for_ue.add(cell_idx)
                    current_ttt_targets_state[ue_idx][cell_idx] = current_ttt_targets_state[ue_idx].get(cell_idx, 0) + dt
                    if current_ttt_targets_state[ue_idx][cell_idx] >= self.ttt:
                        if rsrp[ue_idx, cell_idx] > best_neighbor_quality:
                            best_neighbor_quality = rsrp[ue_idx, cell_idx]; potential_target = cell_idx
            targets_to_reset = set(current_ttt_targets_state[ue_idx].keys()) - active_targets_for_ue
            for target in targets_to_reset: current_ttt_targets_state[ue_idx].pop(target, None)
            if potential_target != -1:
                assignments[ue_idx] = potential_target; current_ttt_targets_state[ue_idx] = {}
        self.prev_assignments = assignments; self.ttt_targets = current_ttt_targets_state
        return assignments
    def utility_based(self, rsrp, sinr, cell_loads, priorities):
        assignments = np.zeros(NUM_UES, dtype=int)
        for ue_idx in range(NUM_UES):
            utilities = np.zeros(NUM_CELLS)
            for cell_idx in range(NUM_CELLS):
                sinr_w = 0.4
                load_w = 0.4
                prio_w = 0.2
                sinr_c = sinr_w * np.clip(sinr[ue_idx, cell_idx], -20, 30)
                load_c = load_w * (1.0 - cell_loads[cell_idx]) * 20
                prio_c = prio_w * (4.0 - float(priorities[ue_idx])) * 10
                utilities[cell_idx] = sinr_c + load_c + prio_c
            assignments[ue_idx] = np.argmax(utilities)
        self.prev_assignments = assignments; self.ttt_targets = {}
        return assignments
    def assign_ues(self, rsrp, sinr, cell_loads, priorities, dt=1.0):
        if self.algorithm == "baseline": return self.baseline_a3(rsrp, sinr, cell_loads, priorities, dt)
        if self.algorithm == "utility": return self.utility_based(rsrp, sinr, cell_loads, priorities)
        if self.prev_assignments is None: self.assign_initial(rsrp)
        return self.prev_assignments.copy()

# --- Module 3: AI Fuzzer ---
class AIFuzzer:
    def __init__(self, env: NetworkEnvironment, ts: TrafficSteering,
                 population_size=FUZZER_POPULATION,
                 generations=FUZZER_GENERATIONS):
        self.env = env; self.ts = ts
        self.population_size = population_size; self.generations = generations
        self.input_vector_size = NUM_CELLS + NUM_UES * 2
        self.objective_call_count = 0
    def _objective_function(self, inputs, current_assignments, dt_fitness=1.0):
        self.objective_call_count += 1
        original_loads = self.env.cell_loads.copy(); original_positions_tf = tf.identity(self.env.ue_loc)
        original_ts_prev_assignments = self.ts.prev_assignments.copy() if self.ts.prev_assignments is not None else None
        original_ts_ttt_targets = {k: v.copy() for k, v in self.ts.ttt_targets.items()}
        try:
            load_modifier = inputs[:NUM_CELLS]
            position_modifier_2d_np = inputs[NUM_CELLS:].reshape(NUM_UES, 2)
            position_modifier_3d_np = np.hstack([position_modifier_2d_np, np.zeros((NUM_UES, 1))])
            position_modifier_tf = tf.constant(position_modifier_3d_np[np.newaxis,...], dtype=tf.float32)
            temp_loads = np.clip(original_loads + load_modifier, 0, 1)
            self.env.ue_loc.assign(original_positions_tf + position_modifier_tf)
            self.env.cell_loads = temp_loads
            rsrp, sinr, _, priorities = self.env.compute_metrics()
            self.ts.prev_assignments = current_assignments
            new_assignments = self.ts.assign_ues(rsrp, sinr, temp_loads, priorities, dt=dt_fitness)
            num_handovers = np.sum(new_assignments != current_assignments)
            fitness_score = -float(num_handovers)
        finally:
            self.env.cell_loads = original_loads; self.env.ue_loc.assign(original_positions_tf)
            self.ts.prev_assignments = original_ts_prev_assignments; self.ts.ttt_targets = original_ts_ttt_targets
        return fitness_score
    def generate_inputs(self, dt=1.0):
        if self.ts.prev_assignments is None:
            rsrp_init, sinr_init, load_init, prio_init = self.env.compute_metrics()
            current_assignments = self.ts.assign_ues(rsrp_init, sinr_init, load_init, prio_init, dt=0)
        else: current_assignments = self.ts.prev_assignments
        population = []; best_overall_fitness = np.inf
        for _ in range(self.population_size):
            load_modifier = np.random.uniform(-0.1, 0.1, NUM_CELLS)
            position_modifier = np.random.uniform(-5, 5, (NUM_UES, 2))
            inputs = np.concatenate([load_modifier, position_modifier.flatten()])
            population.append(inputs)
        if not population: return np.concatenate([np.random.uniform(-0.1,0.1,NUM_CELLS), np.random.uniform(-5,5,(NUM_UES,2)).flatten()])
        best_overall_individual = population[0].copy()
        self.objective_call_count = 0
        
        # Add progress bar for fuzzer generations
        pbar_gen = tqdm(range(self.generations), desc="AI Fuzzer Evolution", leave=False, disable=not hasattr(tqdm, '_instances'))
        for gen in pbar_gen:
            fitness = [self._objective_function(ind, current_assignments, dt) for ind in population]
            sorted_indices = np.argsort(fitness); current_best_fitness = fitness[sorted_indices[0]]
            if current_best_fitness < best_overall_fitness:
                best_overall_fitness = current_best_fitness; best_overall_individual = population[sorted_indices[0]].copy()
            
            # Update progress bar description with current best fitness
            pbar_gen.set_postfix({'Best Fitness': f'{best_overall_fitness:.2f}'})
            
            new_population = [best_overall_individual.copy()]
            num_elites = max(1, int(self.population_size * 0.2)); parent_pool_indices = sorted_indices[:num_elites]
            for _ in range(self.population_size - 1):
                idx1, idx2 = np.random.choice(parent_pool_indices, 2, replace=True)
                parent1 = population[idx1]; parent2 = population[idx2]
                crossover_point = np.random.randint(1, self.input_vector_size)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                if np.random.rand() < 0.1:
                    child[:NUM_CELLS] += np.random.normal(0, 0.05, NUM_CELLS)
                    child[NUM_CELLS:] += np.random.normal(0, 1.0, NUM_UES * 2)
                    child[:NUM_CELLS] = np.clip(child[:NUM_CELLS], -0.2, 0.2)
                new_population.append(child)
            population = new_population
        
        pbar_gen.close()  # Close the progress bar
        return best_overall_individual

# --- Module 3b: Random Fuzzer ---
class RandomFuzzer:
    def __init__(self, env: NetworkEnvironment, ts: TrafficSteering): self.env = env
    def generate_inputs(self, dt=1.0):
        load_modifier = np.random.uniform(-0.1, 0.1, NUM_CELLS)
        position_modifier_2d = np.random.uniform(-5, 5, (NUM_UES, 2))
        inputs = np.concatenate([load_modifier, position_modifier_2d.flatten()])
        return inputs

# --- Module 4: Oracle ---
class Oracle:
    def __init__(self, ping_pong_window=4, ping_pong_threshold=2, qos_sinr_threshold=5.0, fairness_threshold=0.4):
        self.ping_pong_window = ping_pong_window
        self.ping_pong_threshold = ping_pong_threshold
        self.qos_sinr_threshold = qos_sinr_threshold
        self.fairness_threshold = fairness_threshold
        self.handover_history = {}

    def _jain_fairness(self, allocations):
        allocations = np.asarray(allocations)
        allocations_cleaned = allocations[np.isfinite(allocations) & (allocations > 1e-12)]
        if len(allocations_cleaned) == 0: return 1.0
        sum_val = np.sum(allocations_cleaned)
        sum_sq_val = np.sum(allocations_cleaned**2)
        if sum_sq_val < 1e-20: return 1.0
        return sum_val**2 / (len(allocations_cleaned) * sum_sq_val)

    # --- NEW METHOD ADDED HERE ---
    def _alpha_fairness(self, allocations, alpha):
        """
        Calculates the alpha-fairness utility.
        Allocations are typically user throughputs or a proxy like linear SINR.
        """
        # Clean allocations to prevent math errors with log(0) or division by zero.
        allocations_cleaned = allocations[np.isfinite(allocations) & (allocations > 1e-9)]
        if len(allocations_cleaned) == 0:
            return 0.0

        if alpha == 1:
            # Special case for Proportional Fairness (alpha=1)
            return np.sum(np.log(allocations_cleaned))
        else:
            # General formula for alpha-fairness
            return np.sum(allocations_cleaned**(1 - alpha)) / (1 - alpha)

    # --- EVALUATE METHOD MODIFIED ---
    def evaluate(self, rsrp, sinr, assignments, cell_loads, priorities):
        # This method now returns a dictionary of metrics instead of just a list.
        results = {
            'vulnerabilities': [],
            'jain_index': 1.0,
            'alpha_fairness_a1': 0.0, # Proportional Fairness (alpha=1)
            'alpha_fairness_a2': 0.0  # More balanced fairness (alpha=2)
        }
        vulnerabilities_found = []
        num_ping_pongs_detected_this_step = 0

        # --- Ping-Pong Detection (No Change) ---
        for ue_idx in range(NUM_UES):
            if ue_idx not in self.handover_history: self.handover_history[ue_idx] = []
            self.handover_history[ue_idx].append(assignments[ue_idx])
            while len(self.handover_history[ue_idx]) > self.ping_pong_window: self.handover_history[ue_idx].pop(0)
            history = self.handover_history[ue_idx]
            if len(history) == self.ping_pong_window:
                changes = sum(1 for i in range(len(history) - 1) if history[i] != history[i+1])
                if changes >= self.ping_pong_window - 1:
                    num_ping_pongs_detected_this_step +=1
        if num_ping_pongs_detected_this_step >= self.ping_pong_threshold:
            vulnerabilities_found.append(f"Ping-Pong: {num_ping_pongs_detected_this_step} UEs oscillating")

        # --- QoS Violation Detection (No Change) ---
        temp_assigned_sinr_list = []
        for ue_idx in range(NUM_UES):
            assigned_cell_idx = assignments[ue_idx]
            if 0 <= assigned_cell_idx < NUM_CELLS:
                 temp_assigned_sinr_list.append(sinr[ue_idx, assigned_cell_idx])
        assigned_sinr_np = np.array(temp_assigned_sinr_list) if temp_assigned_sinr_list else np.array([])
        
        high_priority_mask = (priorities == 1)
        assigned_sinr_hp_ues_list = [sinr[i, assignments[i]] for i in range(NUM_UES) if high_priority_mask[i]]
        assigned_sinr_hp_ues_np = np.array(assigned_sinr_hp_ues_list) if assigned_sinr_hp_ues_list else np.array([])
        
        if assigned_sinr_hp_ues_np.size > 0:
            avg_sinr_high = np.mean(assigned_sinr_hp_ues_np)
            if avg_sinr_high < self.qos_sinr_threshold:
                vulnerabilities_found.append(f"QoS Violation: Avg High Prio SINR = {avg_sinr_high:.2f} dB (Threshold: {self.qos_sinr_threshold} dB)")

        results['vulnerabilities'] = vulnerabilities_found

        # --- Fairness Calculation Section (Modified) ---
        if assigned_sinr_np.size > 0:
            # Use linear SINR as a proxy for user allocations/throughput
            assigned_sinr_linear = 10**(assigned_sinr_np / 10.0)

            # 1. Calculate Jain's Fairness Index
            jain_score = self._jain_fairness(assigned_sinr_linear)
            results['jain_index'] = jain_score
            if jain_score < self.fairness_threshold:
                results['vulnerabilities'].append(f"Unfairness: Jain Index = {jain_score:.2f}")

            # 2. Calculate Alpha-Fairness for different alpha values
            results['alpha_fairness_a1'] = self._alpha_fairness(assigned_sinr_linear, alpha=1.0)
            results['alpha_fairness_a2'] = self._alpha_fairness(assigned_sinr_linear, alpha=2.0)
        
        return results
    
# --- Module 5: Main Simulation Loop and Analysis ---
#برای توان عملیاتی، از عبارت «توان عملیاتی تخمینی شانون» (Estimated Shannon Throughput) استفاده کنید.

#برای تأخیر، حتماً ذکر کنید که این «تأخیر ارسال برای یک بسته ۱۵۰۰ بایتی» (Transmission Time Delay for a 1500-byte packet) است و نه تأخیر کامل شبکه.
def run_simulation(scenario_name, initial_load=0.3, max_speed=5, scenario_type='default'):
    print(f"\n--- Running Scenario: {scenario_name} (Load: {initial_load}, Speed: {max_speed}, Type: {scenario_type}) ---")
    start_time_scenario = time.time()

    shared_env_state = NetworkEnvironment(initial_load=initial_load, scenario_max_speed=max_speed, scenario_type=scenario_type)
    ts_baseline_proto = TrafficSteering(algorithm="baseline")
    ts_utility_proto = TrafficSteering(algorithm="utility")
    oracle = Oracle(qos_sinr_threshold=5.0, fairness_threshold=0.4, ping_pong_window=4, ping_pong_threshold=3)
    results_list = []
    dt = 1.0
    fuzzer_map = {"AI": AIFuzzer, "Random": RandomFuzzer}
    ts_prototypes = {"baseline": ts_baseline_proto, "utility": ts_utility_proto}
    
    total_combinations = len(fuzzer_map) * len(ts_prototypes)
    combination_pbar = tqdm(total=total_combinations, desc=f"Processing {scenario_name}", leave=False)

    for fuzzer_name, FuzzerClass in fuzzer_map.items():
        for actual_algo_name, ts_instance_proto in ts_prototypes.items():
            combination_pbar.set_description(f"{scenario_name}: {fuzzer_name}+{actual_algo_name}")
            
            shared_env_state.reset(initial_load=initial_load, max_speed=max_speed)
            ts_instance = TrafficSteering(algorithm=actual_algo_name, rsrp_threshold=ts_instance_proto.rsrp_threshold, hysteresis=ts_instance_proto.hysteresis, ttt=ts_instance_proto.ttt, load_threshold=ts_instance_proto.load_threshold)
            oracle.handover_history = {}
            fuzzer = FuzzerClass(shared_env_state, ts_instance)
            
            rsrp_init, sinr_init, _, prio_init = shared_env_state.compute_metrics()
            _ = ts_instance.assign_ues(rsrp_init, sinr_init, shared_env_state.cell_loads, prio_init, dt=0)

            if ts_instance.prev_assignments is None: 
                print(f"CRITICAL ERROR: Initial assignment failed for {actual_algo_name}. Skipping.")
                combination_pbar.update(1)
                continue
            
            shared_env_state.update_cell_loads(ts_instance.prev_assignments)
            
            iter_pbar = tqdm(range(SIMULATION_ITERATIONS), desc=f"  {fuzzer_name}+{actual_algo_name} Iterations", leave=False)
            for iteration in iter_pbar:
                assignments_at_start_of_iter = ts_instance.prev_assignments.copy()
                fuzzed_inputs = fuzzer.generate_inputs(dt)

                load_modifier = fuzzed_inputs[:NUM_CELLS]
                position_modifier_2d = fuzzed_inputs[NUM_CELLS:].reshape(NUM_UES, 2)
                pos_modifier_3d_np = np.hstack([position_modifier_2d, np.zeros((NUM_UES, 1))])

                shared_env_state.cell_loads = np.clip(shared_env_state.cell_loads + load_modifier, 0, 1)
                shared_env_state.ue_loc.assign(shared_env_state.ue_loc.numpy() + pos_modifier_3d_np[np.newaxis,...])
                shared_env_state.update_ue_positions_and_velocities(dt)
                
                rsrp, sinr, cell_loads_eval, priorities_eval = shared_env_state.compute_metrics()
                new_assignments = ts_instance.assign_ues(rsrp, sinr, cell_loads_eval, priorities_eval, dt)
                handovers_this_step = np.sum(new_assignments != assignments_at_start_of_iter)
                shared_env_state.update_cell_loads(new_assignments)

                oracle_metrics = oracle.evaluate(rsrp, sinr, new_assignments, shared_env_state.cell_loads, priorities_eval)
                
                assigned_sinr_list = [sinr[i, new_assignments[i]] if 0 <= new_assignments[i] < NUM_CELLS else np.nan for i in range(NUM_UES)]
                assigned_sinr_np_finite = np.array([s for s in assigned_sinr_list if pd.notna(s)])
                
                # --- NEW: THROUGHPUT & LATENCY PROXY CALCULATION ---
                assigned_sinr_linear = 10**(np.array(assigned_sinr_list) / 10.0)
                
                # Calculate throughput for all UEs in Mbps
                user_throughputs_bps = calculate_throughput(assigned_sinr_linear, BANDWIDTH)
                user_throughputs_mbps = user_throughputs_bps / 1e6
                
                # Calculate Transmission Time Delay for a standard 1500-byte packet
                packet_size_bits = 1500 * 8
                # Add a small epsilon to throughput to avoid division by zero
                transmission_time_s = packet_size_bits / (user_throughputs_bps + 1e-9) 
                # --- END OF NEW SECTION ---

                # --- MODIFIED RESULTS DICTIONARY ---
                results_list.append({
                    'scenario': scenario_name, 'iteration': iteration, 'fuzzer_type': fuzzer_name,
                    'algorithm': actual_algo_name,
                    'handover_count_iter': int(handovers_this_step),
                    'vulnerabilities': oracle_metrics['vulnerabilities'],
                    'jain_fairness_index': float(oracle_metrics['jain_index']),
                    'alpha_fairness_a1': float(oracle_metrics['alpha_fairness_a1']),
                    'alpha_fairness_a2': float(oracle_metrics['alpha_fairness_a2']),
                    'avg_sinr_db': np.mean(assigned_sinr_np_finite) if assigned_sinr_np_finite.size > 0 else np.nan,
                    'sinr_5th_percentile_db': np.percentile(assigned_sinr_np_finite, 5) if assigned_sinr_np_finite.size > 0 else np.nan,
                    'avg_throughput_mbps': np.nanmean(user_throughputs_mbps),
                    'throughput_5th_percentile_mbps': np.nanpercentile(user_throughputs_mbps, 5),
                    'avg_transmission_time_ms': np.nanmean(transmission_time_s) * 1000
                })
                # --- END OF MODIFIED DICTIONARY ---

                iter_pbar.set_postfix({'HOs': handovers_this_step, 'Thrpt_5th': f'{np.nanpercentile(user_throughputs_mbps, 5):.2f}Mbps'})
            iter_pbar.close()
            combination_pbar.update(1)
            
    combination_pbar.close()
    end_time_scenario = time.time()
    print(f"--- Scenario {scenario_name} finished in {end_time_scenario - start_time_scenario:.2f} seconds ---")
    return results_list

def plot_results(df, output_plot_dir="plots_default"):
    print("\n--- Generating Plots ---")
    if df.empty: 
        print("No data to plot.")
        return
    os.makedirs(output_plot_dir, exist_ok=True)
    
    # --- MODIFIED SECTION: UPDATED LIST OF METRICS TO PLOT ---
    # The old list was referencing column names that no longer exist.
    # This new list matches the data now being saved by the run_simulation function.
    metrics_to_plot = [
        'vulnerability_count', 
        'handover_count_iter', 
        'jain_fairness_index', 
        'alpha_fairness_a1',
        'avg_sinr_db', 
        'sinr_5th_percentile_db', 
        'avg_throughput_mbps', 
        'throughput_5th_percentile_mbps', 
        'avg_transmission_time_ms'
    ]
    # --- END OF MODIFIED SECTION ---

    # Calculate total plots to generate
    scenarios = df['scenario'].unique()
    total_plots = len(scenarios) * len(metrics_to_plot)
    plot_pbar = tqdm(total=total_plots, desc="Generating Plots")
    
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario].copy()
        if 'vulnerabilities' in scenario_df.columns: 
            # Ensure vulnerability data is a list before applying len
            scenario_df['vulnerability_count'] = scenario_df['vulnerabilities'].apply(lambda v: len(v) if isinstance(v, list) else 0)
        else: 
            scenario_df['vulnerability_count'] = 0

        for metric in metrics_to_plot:
            plot_pbar.set_description(f"Plotting {scenario}: {metric}")
            if metric not in scenario_df.columns: 
                print(f"Metric '{metric}' not found, skipping plot for {scenario}.")
                plot_pbar.update(1)
                continue

            plt.figure(figsize=(14, 8))
            plot_title = f'{metric.replace("_", " ").title()} over Iterations - Scenario: {scenario}'
            plt.title(plot_title)

            for fuzzer_type in scenario_df['fuzzer_type'].unique():
                fuzzer_df = scenario_df[fuzzer_type == scenario_df['fuzzer_type']]
                for algo in fuzzer_df['algorithm'].unique():
                    algo_fuzzer_df = fuzzer_df[fuzzer_df['algorithm'] == algo]
                    if algo_fuzzer_df.empty or metric not in algo_fuzzer_df.columns: continue

                    if 'iteration' in algo_fuzzer_df.columns:
                        plot_data = algo_fuzzer_df.set_index('iteration')[metric].dropna()
                        if not plot_data.empty:
                            plt.plot(plot_data.index, plot_data.values, marker='o', linestyle='-', markersize=3, alpha=0.7, label=f"{algo} ({fuzzer_type})")
                    else:
                        plot_data_grouped = algo_fuzzer_df.groupby(np.arange(len(algo_fuzzer_df)))[metric].mean()
                        if not plot_data_grouped.empty:
                            plt.plot(plot_data_grouped.index, plot_data_grouped.values, marker='o', linestyle='-', markersize=3, alpha=0.7, label=f"{algo} ({fuzzer_type}) (by row index)")

            plt.xlabel('Iteration')
            plt.ylabel(metric.replace('_', " ").title())
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            plt.grid(True)
            plt.xticks(np.arange(0, SIMULATION_ITERATIONS + 1, step=max(1, SIMULATION_ITERATIONS // 10)))
            
            # --- MODIFIED Y-LIMITS CONDITION TO MATCH NEW METRIC NAMES ---
            if metric not in ['avg_sinr_db', 'sinr_5th_percentile_db', 'alpha_fairness_a1', 'alpha_fairness_a2']: 
                plt.ylim(bottom=0)
            
            safe_scenario_name="".join(c for c in scenario if c.isalnum() or c in (' ','_')).rstrip()
            plot_filename = os.path.join(output_plot_dir, f'{safe_scenario_name.replace(" ","_")}_{metric}.png')
            try:
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plt.savefig(plot_filename)
            except Exception as e:
                print(f"Error saving plot {plot_filename}: {e}")
            plt.close()
            plot_pbar.update(1)
    
    plot_pbar.close()
    print(f"All plots saved to {output_plot_dir}")

def summarize_results(df):
    print("\n--- Results Summary ---")
    if df.empty: print("No results to summarize."); return
    if 'vulnerabilities' in df.columns:
        all_vuln_strings = [v for sl in df['vulnerabilities'].dropna() for v in sl if isinstance(v, str)]
        overall_counts = Counter(all_vuln_strings); print("Overall Vulnerability Counts (All Fuzzers & Algorithms):");
        if not overall_counts: print("   No vulnerabilities detected overall.")
        else:
            for vuln_str, count in overall_counts.most_common(): print(f"   '{vuln_str}': {count}")
    else: print("   'vulnerabilities' column not found for overall summary.")
    print("-" * 40)
    for scenario in df['scenario'].unique():
        print(f"\nScenario: {scenario}"); scenario_data = df[df['scenario'] == scenario]
        for fuzzer_type in scenario_data['fuzzer_type'].unique():
            print(f"   Fuzzer Type: {fuzzer_type}"); fuzzer_data = scenario_data[scenario_data['fuzzer_type'] == fuzzer_type]
            for algo in fuzzer_data['algorithm'].unique():
                print(f"      Algorithm: {algo}"); algo_fuzzer_df = fuzzer_data[fuzzer_data['algorithm'] == algo]
                if 'vulnerabilities' in algo_fuzzer_df.columns:
                    algo_vuln_strings = [v for sl in algo_fuzzer_df['vulnerabilities'].dropna() for v in sl if isinstance(v,str)]
                    if not algo_vuln_strings: print("         No vulnerabilities detected.")
                    else:
                        algo_counts = Counter(algo_vuln_strings)
                        for vuln_str, count in algo_counts.most_common(5): print(f"         '{vuln_str}': {count} occurrences")
                else: print("         'vulnerabilities' column not found for this group.")
                new_metrics_to_summarize = ['avg_overall_sinr', 'avg_high_prio_sinr', 'fairness_index', 'handover_count_iter', 'num_ues_below_qos', 'sinr_5th_percentile', 'sinr_50th_percentile', 'sinr_95th_percentile', 'load_min', 'load_max', 'load_std']
                for metric in new_metrics_to_summarize:
                    if metric in algo_fuzzer_df.columns:
                        try:
                            mean_metric = algo_fuzzer_df[metric].mean(); std_metric = algo_fuzzer_df[metric].std()
                            min_metric = algo_fuzzer_df[metric].min(); max_metric = algo_fuzzer_df[metric].max()
                            print(f"         Avg {metric.replace('_', ' ')}: {mean_metric:.2f} (Std: {std_metric:.2f}, Min: {min_metric:.2f}, Max: {max_metric:.2f})")
                        except pd.errors.DataError: print(f"         Metric '{metric}' contains non-numeric data.")
    print("\n--- End Summary ---")

def main():
    print(f"--- Starting AI Fuzzing Simulation ({SCRIPT_VERSION_NAME}) ---")
    start_time_main = time.time()

    all_results_data = []
    try:
        # Reverted to single-GPU configuration
        if ENABLE_TF_DEVICE_LOGGING: tf.debugging.set_log_device_placement(True)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                print(f"--- Physical GPUs detected: {[gpu.name for gpu in gpus]}. Configuring first GPU. ---")
                tf.config.set_visible_devices(gpus[0], 'GPU')
                for gpu_device_config in tf.config.get_visible_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu_device_config, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                if logical_gpus: print(f"--- Configured to use {len(logical_gpus)} Logical GPU(s): {[lg.name for lg in logical_gpus]} ---")
            except Exception as e:
                print(f"Error during GPU setup: {e}. Using default CPU strategy.")
        else:
            print("--- No GPU detected by TensorFlow. Running on CPU. ---")
        
        # All operations run on the configured device (default or single GPU)
        scenarios_to_run = [
            ('Low Load', 0.3, 5, 'default'),
            ('High Load', 0.7, 5, 'default'),
            ('High Mobility', 0.5, 10, 'default'),
            ('Mixed Mobility', 0.5, 7, 'mixed')
        ]

        # Add main progress bar for scenarios
        scenario_pbar = tqdm(scenarios_to_run, desc="Overall Progress", position=0)
        for name, load, speed, scenario_type in scenario_pbar:
            scenario_pbar.set_description(f"Running: {name}")
            np.random.seed(42); tf.random.set_seed(42)
            results = run_simulation(scenario_name=name, initial_load=load, max_speed=speed, scenario_type=scenario_type)
            all_results_data.extend(results)

        scenario_pbar.close()

    except Exception as main_exc:
        print(f"\nCRITICAL ERROR in main loop: {main_exc}")

    finally:
        print("\n--- Finalizing script: Saving results... ---")
        if not all_results_data:
            print("No results were generated to save.")
        else:
            results_df = pd.DataFrame(all_results_data)
            csv_filename = f'fuzzing_results_{SCRIPT_VERSION_NAME}.csv'
            plot_dir = f"plots_{SCRIPT_VERSION_NAME}"

            try:
                results_df.to_csv(csv_filename, index=False, encoding='utf-8')
                print(f"\n--- Results successfully saved to {csv_filename} ---")
            except Exception as e:
                print(f"Could not save results to CSV {csv_filename}: {e}")

            summarize_results(results_df)
            plot_results(results_df, plot_dir)

        end_time_main = time.time()
        print(f"\n--- Simulation Finished in {end_time_main - start_time_main:.2f} seconds ---")

if __name__ == "__main__":
    np.random.seed(42); tf.random.set_seed(42)
    main()
