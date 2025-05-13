# -*- coding: utf-8 -*-
# Combined AI Fuzzing Script for O-RAN Traffic Steering Vulnerability Analysis
# Version 21: Incorporating friend's feedback: Tuned Fuzzer/Oracle params,
#             added more metrics, and shape assertions. Based on v20 (retracing fix).

# --- Imports ---
import sionna
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from collections import Counter

# --- Sionna specific imports for channel modeling ---
from sionna.phy.channel.tr38901 import UMi, PanelArray, Antenna
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel import GenerateOFDMChannel


# --- Global Constants ---
NUM_CELLS = 3
NUM_UES = 30
BANDWIDTH = 20e6
CARRIER_FREQUENCY = 3.5e9
TX_POWER_DBM = 30
NOISE_POWER_DBM_PER_HZ = -174

SIMULATION_ITERATIONS = 100
FUZZER_GENERATIONS = 100
FUZZER_POPULATION = 10

ENABLE_DETAILED_METRIC_PRINT = False 
ENABLE_TF_DEVICE_LOGGING = False


# --- Module 1: Network Simulation Environment ---
class NetworkEnvironment:
    def __init__(self, initial_load=0.3, scenario_max_speed=5):
        self.batch_size = 1 
        self.initial_load_param = initial_load
        self.max_speed_param = scenario_max_speed

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
            num_ofdm_symbols=14, fft_size=128,
            subcarrier_spacing=30e3,
            num_tx=NUM_CELLS, 
            num_streams_per_tx=1, 
            cyclic_prefix_length=20, 
            pilot_pattern="kronecker", 
            pilot_ofdm_symbol_indices=[2, 11],
            num_guard_carriers=(8, 8), 
            dc_null=True
        )
        
        self.generate_h_freq_layer = GenerateOFDMChannel(
            channel_model=self.channel_model_3gpp,
            resource_grid=self.resource_grid,
            precision="single" 
        )
        
        self.bs_pos_2d = np.array([[0,0], [100,0], [50, 86.6]]) * 1.0
        self.bs_loc = tf.constant(np.hstack([self.bs_pos_2d, np.ones((NUM_CELLS, 1)) * 10.0])[np.newaxis,...], dtype=tf.float32)
        
        self.ue_loc = tf.Variable(tf.zeros([self.batch_size, NUM_UES, 3], dtype=tf.float32), name="ue_loc")
        self.ue_velocities = tf.Variable(tf.zeros([self.batch_size, NUM_UES, 3], dtype=tf.float32), name="ue_velocities")

        self.ut_orientations = tf.zeros([self.batch_size, NUM_UES, 3], dtype=tf.float32)
        self.bs_orientations = tf.zeros([self.batch_size, NUM_CELLS, 3], dtype=tf.float32)
        self.in_state = tf.zeros([self.batch_size, NUM_UES], dtype=tf.bool)

        self.cell_loads = np.ones(NUM_CELLS) * initial_load
        self.ue_priorities = np.random.choice([1, 2, 3], size=NUM_UES, p=[0.3, 0.4, 0.3]).astype(np.float32)

        self.noise_power_watts = 10**((NOISE_POWER_DBM_PER_HZ - 30) / 10) * BANDWIDTH
        self.tx_power_watts_total = 10**((TX_POWER_DBM - 30) / 10)
        
        num_effective_sc = self.resource_grid.num_effective_subcarriers
        self.tx_power_watts_per_subcarrier = self.tx_power_watts_total / tf.cast(num_effective_sc, tf.float32)
        
        self.reset(initial_load, scenario_max_speed)
        # print("NetworkEnvironment Initialized (Full OFDM Channel v21).")

    def reset(self, initial_load, max_speed):
        self.initial_load_param = initial_load
        self.max_speed_param = max_speed

        ue_pos_2d_np = np.concatenate([
            np.random.uniform(-150, 250, size=(int(NUM_UES * 0.7), 2)),
            np.random.normal([np.mean(self.bs_pos_2d[:,0]), np.mean(self.bs_pos_2d[:,1])], 50, size=(int(NUM_UES * 0.3), 2))
        ])
        ue_pos_2d_np = np.clip(ue_pos_2d_np, -1e6, 1e6)
        self.ue_loc.assign(np.hstack([ue_pos_2d_np, np.ones((NUM_UES, 1)) * 1.5])[np.newaxis,...])

        ue_vel_2d_np = np.random.uniform(-max_speed, max_speed, size=(NUM_UES, 2))
        self.ue_velocities.assign(np.hstack([ue_vel_2d_np, np.zeros((NUM_UES, 1))])[np.newaxis,...])
        
        self.cell_loads = np.ones(NUM_CELLS) * initial_load
        # self.ue_priorities = np.random.choice([1, 2, 3], size=NUM_UES, p=[0.3, 0.4, 0.3]).astype(np.float32) # Reset if needed

    def update_ue_positions_and_velocities(self, dt=1.0, max_speed=None):
        if max_speed is None: max_speed = self.max_speed_param
        new_velocities = self.ue_velocities + tf.random.normal(shape=self.ue_velocities.shape, stddev=1.0, dtype=tf.float32) * dt
        speeds = tf.norm(new_velocities, axis=2, keepdims=True)
        safe_speeds = tf.where(speeds < 1e-9, tf.ones_like(speeds) * 1e-9, speeds)
        scale = tf.minimum(1.0, max_speed / safe_speeds)
        new_velocities = new_velocities * scale
        new_velocities = tf.where(tf.math.is_finite(new_velocities), new_velocities, tf.zeros_like(new_velocities))
        self.ue_velocities.assign(new_velocities)
        new_loc = self.ue_loc + new_velocities * dt
        new_loc = tf.where(tf.math.is_finite(new_loc), new_loc, self.ue_loc)
        self.ue_loc.assign(new_loc)
        ue_loc_np = self.ue_loc.numpy()[0]
        ue_vel_np = self.ue_velocities.numpy()[0]
        min_x, max_x = -200.0, 300.0; min_y, max_y = -200.0, 250.0
        hit_x_min = ue_loc_np[:, 0] <= min_x; hit_x_max = ue_loc_np[:, 0] >= max_x
        hit_y_min = ue_loc_np[:, 1] <= min_y; hit_y_max = ue_loc_np[:, 1] >= max_y
        ue_loc_np[hit_x_min, 0] = min_x; ue_loc_np[hit_x_max, 0] = max_x
        ue_loc_np[hit_y_min, 1] = min_y; ue_loc_np[hit_y_max, 1] = max_y
        ue_vel_np[hit_x_min | hit_x_max, 0] *= -1
        ue_vel_np[hit_y_min | hit_y_max, 1] *= -1
        ue_loc_np = np.where(np.isfinite(ue_loc_np), ue_loc_np, 0.0) 
        ue_vel_np = np.where(np.isfinite(ue_vel_np), ue_vel_np, 0.0)
        self.ue_loc.assign(ue_loc_np[np.newaxis,...])
        self.ue_velocities.assign(ue_vel_np[np.newaxis,...])

    @tf.function(jit_compile=True)
    def compute_metrics_tf(self, ue_loc_tf, bs_loc_tf, ut_orient_tf, bs_orient_tf, ut_vel_tf, in_state_tf):
        self.channel_model_3gpp.set_topology(
            ut_loc=ue_loc_tf, bs_loc=bs_loc_tf,
            ut_orientations=ut_orient_tf, bs_orientations=bs_orient_tf,
            ut_velocities=ut_vel_tf, in_state=in_state_tf
        )
        h_freq = self.generate_h_freq_layer(batch_size=self.batch_size)
        
        if ENABLE_DETAILED_METRIC_PRINT: # and not tf.config.functions_run_eagerly():
             # For JIT, tf.print can be problematic. Use with caution or when JIT is off.
             # tf.debugging.assert_shapes is better for JIT if checks are needed.
            tf.debugging.assert_shapes([
                (h_freq, (self.batch_size, NUM_UES, 1, NUM_CELLS, 1, self.resource_grid.num_ofdm_symbols, self.resource_grid.fft_size))
            ])
            # tf.print(f"h_freq shape from GenerateOFDMChannel: {tf.shape(h_freq)}")

        h_freq_squeezed = tf.squeeze(h_freq, axis=[2, 4])
        avg_channel_power_gain = tf.reduce_mean(tf.abs(h_freq_squeezed)**2, axis=[-2, -1])
        received_power_watts_tf = self.tx_power_watts_total * avg_channel_power_gain
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

        # The LSPs are implicitly included in h_freq. Logging _lsp.sf is for deeper debugging/comparison.
        # if ENABLE_DETAILED_METRIC_PRINT and not tf.config.functions_run_eagerly():
        #     # ... (SF logging as before) ...
        return rsrp_db_tf, sinr_db_tf

    def compute_metrics(self):
        if self.channel_model_3gpp is None or self.resource_grid is None or not hasattr(self, 'generate_h_freq_layer'):
            print("CRITICAL ERROR: Sionna components not initialized for OFDM-based metrics.")
            return np.full((NUM_UES, NUM_CELLS), -200.0), np.full((NUM_UES, NUM_CELLS), -30.0), \
                   self.cell_loads.copy(), self.ue_priorities.copy()
        try:
            if not (tf.reduce_all(tf.math.is_finite(self.ue_loc)) and \
                    tf.reduce_all(tf.math.is_finite(self.bs_loc)) and \
                    tf.reduce_all(tf.math.is_finite(self.ue_velocities))):
                print("Warning: Invalid UE/BS locations or velocities detected (NaN or Inf). Returning defaults.")
                return np.full((NUM_UES, NUM_CELLS), -200.0), np.full((NUM_UES, NUM_CELLS), -30.0), \
                       self.cell_loads.copy(), self.ue_priorities.copy()

            rsrp_db_tf, sinr_db_tf = self.compute_metrics_tf(
                self.ue_loc, self.bs_loc, self.ut_orientations,
                self.bs_orientations, self.ue_velocities, self.in_state
            )
            return rsrp_db_tf.numpy(), sinr_db_tf.numpy(), self.cell_loads.copy(), self.ue_priorities.copy()

        except tf.errors.InvalidArgumentError as e: print(f"TensorFlow InvalidArgumentError in compute_metrics: {e}")
        except tf.errors.ResourceExhaustedError as e: print(f"GPU memory EXCEEDED in compute_metrics: {e}")
        except AttributeError as ae: print(f"AttributeError during Sionna UMi metric computation: {ae}")
        except Exception as e:
            print(f"General Uncaught Error during Sionna UMi metric computation: {e}")
            if hasattr(self, 'ue_loc') and hasattr(self, 'bs_loc'):
                 print(f"Variables at error: ue_loc shape: {self.ue_loc.shape}, bs_loc shape: {self.bs_loc.shape}")
        
        return np.full((NUM_UES, NUM_CELLS), -200.0), np.full((NUM_UES, NUM_CELLS), -30.0), \
               self.cell_loads.copy(), self.ue_priorities.copy()

    def update_cell_loads(self, assignments):
        self.cell_loads = np.zeros(NUM_CELLS)
        unique_cells, counts = np.unique(assignments, return_counts=True)
        load_per_ue = 1.0 / NUM_UES
        for cell_idx, count in zip(unique_cells, counts):
             if 0 <= cell_idx < NUM_CELLS: self.cell_loads[cell_idx] = count * load_per_ue
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
                sinr_c = 0.5 * np.clip(sinr[ue_idx, cell_idx], -20, 30)
                load_c = 0.3 * (1.0 - cell_loads[cell_idx]) * 20
                prio_c = 0.2 * (4.0 - float(priorities[ue_idx])) * 10
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
    def __init__(self, env: NetworkEnvironment, ts: TrafficSteering, population_size=FUZZER_POPULATION, generations=FUZZER_GENERATIONS):
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
        else:
            current_assignments = self.ts.prev_assignments
        population = []
        for _ in range(self.population_size):
            # Tuned fuzzer parameter ranges (more conservative)
            load_modifier = np.random.uniform(-0.1, 0.1, NUM_CELLS) 
            position_modifier = np.random.uniform(-5, 5, (NUM_UES, 2)) 
            inputs = np.concatenate([load_modifier, position_modifier.flatten()])
            population.append(inputs)
        best_overall_fitness = np.inf; best_overall_individual = population[0].copy()
        self.objective_call_count = 0
        for gen in range(self.generations):
            fitness = [self._objective_function(ind, current_assignments, dt) for ind in population]
            sorted_indices = np.argsort(fitness); current_best_fitness = fitness[sorted_indices[0]]
            if current_best_fitness < best_overall_fitness:
                 best_overall_fitness = current_best_fitness; best_overall_individual = population[sorted_indices[0]].copy()
            new_population = [best_overall_individual.copy()]
            num_elites = max(1, int(self.population_size * 0.2)); parent_pool_indices = sorted_indices[:num_elites]
            for _ in range(self.population_size - 1):
                idx1, idx2 = np.random.choice(parent_pool_indices, 2, replace=True)
                parent1 = population[idx1]; parent2 = population[idx2]
                crossover_point = np.random.randint(1, self.input_vector_size)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                if np.random.rand() < 0.1: # Mutation
                    child[:NUM_CELLS] += np.random.normal(0, 0.05, NUM_CELLS) # Smaller mutation for load
                    child[NUM_CELLS:] += np.random.normal(0, 1.0, NUM_UES * 2) # Smaller mutation for position
                    child[:NUM_CELLS] = np.clip(child[:NUM_CELLS], -0.2, 0.2) # Clipped mutation range
                new_population.append(child)
            population = new_population
        return best_overall_individual

# --- Module 3b: Random Fuzzer ---
class RandomFuzzer:
    def __init__(self, env: NetworkEnvironment, ts: TrafficSteering):
        self.env = env
    def generate_inputs(self, dt=1.0):
        # Tuned fuzzer parameter ranges (more conservative)
        load_modifier = np.random.uniform(-0.1, 0.1, NUM_CELLS)
        position_modifier_2d = np.random.uniform(-5, 5, (NUM_UES, 2))
        inputs = np.concatenate([load_modifier, position_modifier_2d.flatten()])
        return inputs

# --- Module 4: Oracle ---
class Oracle:
    def __init__(self, ping_pong_window=4, ping_pong_threshold=1, # Increased ping_pong_window
                       qos_sinr_threshold=0.0, fairness_threshold=0.4): # Adjusted thresholds
        self.ping_pong_window = ping_pong_window
        self.ping_pong_threshold = ping_pong_threshold
        self.qos_sinr_threshold = qos_sinr_threshold
        self.fairness_threshold = fairness_threshold
        self.handover_history = {}
    def _jain_fairness(self, allocations):
        allocations = np.asarray(allocations)
        # Filter out non-finite and very small values before calculation to avoid NaNs in fairness
        allocations_cleaned = allocations[np.isfinite(allocations) & (allocations > 1e-12)] # Increased floor slightly
        if len(allocations_cleaned) == 0: return 1.0 # Perfect fairness if no one gets resources
        sum_val = np.sum(allocations_cleaned)
        sum_sq_val = np.sum(allocations_cleaned**2)
        if sum_sq_val < 1e-20: return 1.0 # Avoid division by zero if all allocations are extremely small
        return sum_val**2 / (len(allocations_cleaned) * sum_sq_val)
    def evaluate(self, rsrp, sinr, assignments, cell_loads, priorities):
        vulnerabilities_found = []; num_ping_pongs_detected_this_step = 0
        for ue_idx in range(NUM_UES):
            if ue_idx not in self.handover_history: self.handover_history[ue_idx] = []
            self.handover_history[ue_idx].append(assignments[ue_idx])
            while len(self.handover_history[ue_idx]) > self.ping_pong_window:
                self.handover_history[ue_idx].pop(0)
            history = self.handover_history[ue_idx]
            if len(history) == self.ping_pong_window:
                # Classic Ping-Pong: A -> B -> A
                if history[0] == history[2] and history[0] != history[1] and \
                   history[1] == history[3] and history[1] != history[2]: # A-B-A-B pattern (more robust for window 4)
                     num_ping_pongs_detected_this_step += 1
                # Simpler A-B-A for shorter windows if needed
                # elif self.ping_pong_window == 3 and history[0] == history[2] and history[0] != history[1]:
                #      num_ping_pongs_detected_this_step += 1


        if num_ping_pongs_detected_this_step >= self.ping_pong_threshold:
            vulnerabilities_found.append(f"Ping-Pong: {num_ping_pongs_detected_this_step} UEs")
        
        assigned_sinr = np.array([sinr[ue_idx, assignments[ue_idx]] for ue_idx in range(NUM_UES) if assignments[ue_idx] < NUM_CELLS and assignments[ue_idx] >=0])
        # Ensure priorities array aligns with assigned_sinr if some UEs had invalid assignments filtered out
        # For simplicity, assuming assignments are always valid for now for priority mask
        
        high_priority_mask = (priorities == 1) # Assuming priorities is [NUM_UES]
        assigned_sinr_for_hp_ues = np.array([sinr[i, assignments[i]] for i in range(NUM_UES) if high_priority_mask[i]])


        if assigned_sinr_for_hp_ues.size > 0: # Check if there are any high priority UEs
            avg_sinr_high = np.mean(assigned_sinr_for_hp_ues)
            if avg_sinr_high < self.qos_sinr_threshold:
                vulnerabilities_found.append(f"QoS Violation: Avg High Prio SINR = {avg_sinr_high:.2f} dB (Threshold: {self.qos_sinr_threshold} dB)")
        
        # Use assigned_sinr (already filtered for valid assignments implicitly by list comprehension)
        # Ensure sinr values are not excessively negative before converting to linear, which can cause overflow for 10**(large_negative/10)
        # Clip SINR values before converting to linear for fairness calculation
        clipped_sinr_for_fairness = np.clip(assigned_sinr, -30, 30) # Clip to avoid extreme values in fairness
        assigned_sinr_linear = 10**(clipped_sinr_for_fairness / 10.0)
        fairness = self._jain_fairness(assigned_sinr_linear)
        if fairness < self.fairness_threshold:
            vulnerabilities_found.append(f"Unfairness: Jain Index = {fairness:.2f}")
        return vulnerabilities_found

# --- Module 5: Main Simulation Loop and Analysis ---
def run_simulation(scenario_name, initial_load=0.3, max_speed=5):
    print(f"\n--- Running Scenario: {scenario_name} (Load: {initial_load}, Speed: {max_speed}) ---")
    start_time_scenario = time.time()
    
    # Create one NetworkEnvironment instance per scenario run
    shared_env_state = NetworkEnvironment(initial_load=initial_load, scenario_max_speed=max_speed)

    # Adjusted Oracle thresholds as per friend's suggestion (example values)
    oracle = Oracle(qos_sinr_threshold=0.0, fairness_threshold=0.4, ping_pong_window=4, ping_pong_threshold=2)
    results_list = []; dt = 1.0

    fuzzer_map = {"AI": AIFuzzer, "Random": RandomFuzzer}
    ts_prototypes = {"baseline": TrafficSteering(algorithm="baseline"), 
                     "utility": TrafficSteering(algorithm="utility")}

    for fuzzer_name, FuzzerClass in fuzzer_map.items():
        print(f"=== Fuzzer Type: {fuzzer_name} ===")
        for actual_algo_name, ts_instance_proto in ts_prototypes.items():
            print(f"--- Algorithm: {actual_algo_name} (with {fuzzer_name} Fuzzer) ---")
            start_time_algo = time.time()

            shared_env_state.reset(initial_load=initial_load, max_speed=max_speed)
            
            ts_instance = TrafficSteering(algorithm=actual_algo_name, 
                                          rsrp_threshold=ts_instance_proto.rsrp_threshold,
                                          hysteresis=ts_instance_proto.hysteresis,
                                          ttt=ts_instance_proto.ttt,
                                          load_threshold=ts_instance_proto.load_threshold)
            oracle.handover_history = {}

            fuzzer = FuzzerClass(shared_env_state, ts_instance)

            # Store assignments from before TS makes a decision in this iteration
            # This helps in calculating handovers *due to TS decisions in the current iteration*
            # assignments_before_ts_decision_this_iter = shared_env_state.ue_assignments if hasattr(shared_env_state, 'ue_assignments') else np.full(NUM_UES, -1, dtype=int)
            
            # Initial metric calculation and UE assignment
            rsrp_init, sinr_init, _, prio_init = shared_env_state.compute_metrics()
            initial_assignments = ts_instance.assign_ues(rsrp_init, sinr_init, shared_env_state.cell_loads, prio_init, dt=0)
            # ts_instance.prev_assignments is now set by assign_ues
            
            if ts_instance.prev_assignments is None: 
                print(f"CRITICAL ERROR: Initial assignment for {actual_algo_name} with {fuzzer_name}. Skipping.")
                continue
            shared_env_state.update_cell_loads(ts_instance.prev_assignments)
            
            total_handovers_in_run = 0 # For this algorithm run

            for iteration in range(SIMULATION_ITERATIONS):
                assignments_at_start_of_iter = ts_instance.prev_assignments.copy()

                fuzzed_inputs = fuzzer.generate_inputs(dt)
                load_modifier = fuzzed_inputs[:NUM_CELLS]
                position_modifier_2d = fuzzed_inputs[NUM_CELLS:].reshape(NUM_UES, 2)
                position_modifier_3d = np.hstack([position_modifier_2d, np.zeros((NUM_UES,1))])
                
                shared_env_state.cell_loads = np.clip(shared_env_state.cell_loads + load_modifier, 0, 1)
                shared_env_state.ue_loc.assign_add(tf.constant(position_modifier_3d[np.newaxis,...], dtype=tf.float32))
                shared_env_state.update_ue_positions_and_velocities(dt, max_speed) 
                
                rsrp, sinr, cell_loads_eval, priorities_eval = shared_env_state.compute_metrics()
                new_assignments = ts_instance.assign_ues(rsrp, sinr, cell_loads_eval, priorities_eval, dt)
                
                # Calculate handovers for this iteration step
                handovers_this_step = np.sum(new_assignments != assignments_at_start_of_iter)
                total_handovers_in_run += handovers_this_step

                shared_env_state.update_cell_loads(new_assignments)
                vulnerabilities = oracle.evaluate(rsrp, sinr, new_assignments, shared_env_state.cell_loads, priorities_eval)
                
                # Calculate additional metrics for this iteration
                assigned_sinr_for_metrics = np.array([sinr[ue_idx, new_assignments[ue_idx]] for ue_idx in range(NUM_UES) if new_assignments[ue_idx] < NUM_CELLS and new_assignments[ue_idx] >=0])
                avg_overall_sinr_iter = np.mean(assigned_sinr_for_metrics) if assigned_sinr_for_metrics.size > 0 else -999
                
                priorities_iter = shared_env_state.ue_priorities # Assuming this is current
                high_priority_mask_iter = (priorities_iter == 1)
                assigned_sinr_hp_iter = np.array([sinr[i, new_assignments[i]] for i in range(NUM_UES) if high_priority_mask_iter[i]])
                avg_high_prio_sinr_iter = np.mean(assigned_sinr_hp_iter) if assigned_sinr_hp_iter.size > 0 else -999

                clipped_sinr_for_fairness = np.clip(assigned_sinr_for_metrics, -30, 30)
                fairness_index_iter = oracle._jain_fairness(10**(clipped_sinr_for_fairness / 10.0))

                results_list.append({
                    'scenario': scenario_name, 
                    'iteration': iteration, 
                    'fuzzer_type': fuzzer_name,
                    'algorithm': actual_algo_name, 
                    # 'assignments': new_assignments.tolist(), # Can be very large for CSV
                    'cell_loads': shared_env_state.cell_loads.tolist(), 
                    'avg_overall_sinr': avg_overall_sinr_iter,
                    'avg_high_prio_sinr': avg_high_prio_sinr_iter,
                    'fairness_index': fairness_index_iter,
                    'handover_count_iter': handovers_this_step,
                    'vulnerabilities': vulnerabilities
                })
                if (iteration + 1) % (SIMULATION_ITERATIONS // 10 or 1) == 0 :
                     print(f"    Fuzzer: {fuzzer_name}, Algo: {actual_algo_name}, Scenario: {scenario_name}, Iter: {iteration + 1}/{SIMULATION_ITERATIONS} done.")
            
            avg_ho_rate_per_iter = total_handovers_in_run / SIMULATION_ITERATIONS if SIMULATION_ITERATIONS > 0 else 0
            print(f"    Avg HO rate for {actual_algo_name} with {fuzzer_name}: {avg_ho_rate_per_iter:.2f} HOs/iteration")
            end_time_algo = time.time()
            print(f"--- Algorithm {actual_algo_name} with {fuzzer_name} Fuzzer finished in {end_time_algo - start_time_algo:.2f} seconds ---")
    
    end_time_scenario = time.time()
    print(f"--- Scenario {scenario_name} finished in {end_time_scenario - start_time_scenario:.2f} seconds ---")
    return results_list

# --- plot_results and summarize_results need to be updated to handle new metrics ---
def plot_results(df, output_plot_dir="plots_default"):
    print("\n--- Generating Plots ---")
    if df.empty: print("No data to plot."); return
    os.makedirs(output_plot_dir, exist_ok=True)
    
    metrics_to_plot = ['vulnerability_count', 'avg_overall_sinr', 'avg_high_prio_sinr', 'fairness_index', 'handover_count_iter']
    
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario].copy()
        if 'vulnerabilities' in scenario_df.columns:
             scenario_df['vulnerability_count'] = scenario_df['vulnerabilities'].apply(len)
        else:
            scenario_df['vulnerability_count'] = 0


        for metric in metrics_to_plot:
            if metric not in scenario_df.columns:
                print(f"Metric {metric} not found in results, skipping plot for {scenario}.")
                continue

            plt.figure(figsize=(14, 8))
            for fuzzer_type in scenario_df['fuzzer_type'].unique():
                fuzzer_df = scenario_df[scenario_df['fuzzer_type'] == fuzzer_type]
                for algo in fuzzer_df['algorithm'].unique():
                    algo_fuzzer_df = fuzzer_df[fuzzer_df['algorithm'] == algo]
                    # Group by iteration and calculate mean for the metric
                    plot_data = algo_fuzzer_df.groupby('iteration')[metric].mean()
                    plt.plot(plot_data.index, plot_data.values, marker='o', linestyle='-', markersize=4, label=f"{algo} ({fuzzer_type})")
            
            plt.xlabel('Iteration')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} - Scenario: {scenario}')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Place legend outside
            plt.grid(True)
            plt.xticks(np.arange(0, SIMULATION_ITERATIONS + 1, step=max(1, SIMULATION_ITERATIONS // 10)))
            if metric != 'vulnerability_count': # Allow auto-scaling for SINR etc.
                 pass # plt.ylim(bottom=0 if metric != 'avg_overall_sinr' and metric != 'avg_high_prio_sinr' else None)
            else:
                plt.ylim(bottom=0)

            safe_scenario_name="".join(c for c in scenario if c.isalnum() or c in (' ','_')).rstrip()
            plot_filename = os.path.join(output_plot_dir, f'{metric}_{safe_scenario_name.replace(" ","_")}.png')
            try:
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
                plt.savefig(plot_filename); print(f"Saved plot: {plot_filename}")
            except Exception as e: print(f"Error saving plot {plot_filename}: {e}")
            plt.close()

def summarize_results(df):
    print("\n--- Results Summary ---")
    if df.empty: print("No results to summarize."); return
    
    # Overall Summary (Aggregated across all runs)
    if 'vulnerabilities' in df.columns:
        all_vuln_strings = [v for sl in df['vulnerabilities'].dropna() for v in sl if isinstance(v, str)]
        overall_counts = Counter(all_vuln_strings)
        print("Overall Vulnerability Counts (All Fuzzers & Algorithms):");
        if not overall_counts: print("  No vulnerabilities detected overall.")
        else:
            for vuln_str, count in overall_counts.most_common(): print(f"  '{vuln_str}': {count}")
    else:
        print("  'vulnerabilities' column not found in results for overall summary.")
    print("-" * 40)

    # Per Scenario / Fuzzer Type / Algorithm Summary
    for scenario in df['scenario'].unique():
        print(f"\nScenario: {scenario}")
        scenario_data = df[df['scenario'] == scenario]
        for fuzzer_type in scenario_data['fuzzer_type'].unique():
            print(f"  Fuzzer Type: {fuzzer_type}")
            fuzzer_data = scenario_data[scenario_data['fuzzer_type'] == fuzzer_type]
            for algo in fuzzer_data['algorithm'].unique():
                print(f"    Algorithm: {algo}")
                algo_fuzzer_df = fuzzer_data[fuzzer_data['algorithm'] == algo]
                
                if 'vulnerabilities' in algo_fuzzer_df.columns:
                    algo_vuln_strings = [v for sl in algo_fuzzer_df['vulnerabilities'].dropna() for v in sl if isinstance(v,str)]
                    if not algo_vuln_strings: print("      No vulnerabilities detected.")
                    else:
                        algo_counts = Counter(algo_vuln_strings)
                        for vuln_str, count in algo_counts.most_common(5): print(f"      '{vuln_str}': {count} occurrences")
                else:
                     print("      'vulnerabilities' column not found for this group.")

                # Summarize new metrics
                for metric in ['avg_overall_sinr', 'avg_high_prio_sinr', 'fairness_index', 'handover_count_iter']:
                    if metric in algo_fuzzer_df.columns:
                        mean_metric = algo_fuzzer_df[metric].mean()
                        std_metric = algo_fuzzer_df[metric].std()
                        print(f"      Avg {metric.replace('_', ' ')}: {mean_metric:.2f} (Std: {std_metric:.2f})")
    print("\n--- End Summary ---")


def main():
    script_version_name = "v21_tuned_OFDM" # For output files
    print(f"--- Starting AI Fuzzing Simulation ({script_version_name}) ---")
    start_time_main = time.time()

    if ENABLE_TF_DEVICE_LOGGING:
        tf.debugging.set_log_device_placement(True)

    gpus = tf.config.list_physical_devices('GPU')
    strategy = tf.distribute.get_strategy() 

    if gpus:
        try:
            print(f"--- Physical GPUs detected: {[gpu.name for gpu in gpus]}. Configuring first GPU. ---")
            tf.config.set_visible_devices(gpus[0], 'GPU') 
            for gpu_device_config in tf.config.get_visible_devices('GPU'): 
                tf.config.experimental.set_memory_growth(gpu_device_config, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            if logical_gpus:
                print(f"--- Configured to use {len(logical_gpus)} Logical GPU(s): {[lg.name for lg in logical_gpus]} ---")
            else:
                print("--- WARNING: No logical GPUs configured. ---")
        except Exception as e: 
            print(f"Error during GPU setup: {e}. Using default CPU strategy.")
    else:
        print("--- No GPU detected by TensorFlow. Running on CPU. ---")

    with strategy.scope(): 
        scenarios_to_run = [('Low Load', 0.3, 5), ('High Load', 0.7, 5), ('High Mobility', 0.5, 10)]
        all_results_data = []
        for name, load, speed in scenarios_to_run:
            np.random.seed(42); tf.random.set_seed(42)
            results = run_simulation(scenario_name=name, initial_load=load, max_speed=speed)
            all_results_data.extend(results)
            
    if not all_results_data: print("Simulation produced no results."); return
    
    results_df = pd.DataFrame(all_results_data)
    csv_filename = f'fuzzing_results_{script_version_name}.csv'
    plot_dir = f"plots_{script_version_name}"

    try: results_df.to_csv(csv_filename, index=False, encoding='utf-8'); print(f"\n--- Results saved to {csv_filename} ---")
    except Exception as e: print(f"Error saving results to CSV {csv_filename}: {e}")
    
    summarize_results(results_df)
    plot_results(results_df, plot_dir)
    
    end_time_main = time.time()
    print(f"\n--- Simulation Finished in {end_time_main - start_time_main:.2f} seconds ---")

if __name__ == "__main__":
    np.random.seed(42); tf.random.set_seed(42)
    main()