# -*- coding: utf-8 -*-
# Combined AI Fuzzing Script for O-RAN Traffic Steering Vulnerability Analysis
# Version with increased iterations and stricter QoS threshold

# --- Imports ---
import sionna as sn
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time # Added for timing info
from collections import Counter # For improved summary
import sionna.phy.channel.tr38901 # Might be needed for UMi
from sionna.channel import OFDMChannel
from sionna.ofdm import ResourceGrid
from sionna.channel.antenna import PanelArray, Antenna # For antenna definitions

# --- Global Constants ---
NUM_CELLS = 3
NUM_UES = 30
BANDWIDTH = 20e6  # 20 MHz
CARRIER_FREQUENCY = 3.5e9  # 3.5 GHz
TX_POWER = 30  # 30 dBm (Watts conversion happens inside class)
NOISE_POWER = -174  # dBm/Hz (Watts conversion happens inside class)
# --- User Modified Parameters ---
SIMULATION_ITERATIONS = 100 # Number of iterations per scenario/algorithm (Increased)
FUZZER_GENERATIONS = 100 # Fuzzer generations per iteration (Increased)
FUZZER_POPULATION = 10   # Fuzzer population size (Increased)
# --- End User Modified Parameters ---


# --- Module 1: Network Simulation Environment ---
class NetworkEnvironment:
    """
    Simulates the basic network environment with cells, UEs, channel, and mobility.
    Uses simplified models for initial speed. Sionna channel object initialization is
    commented out temporarily due to import issues; direct formulas are used below.
    """
    def __init__(self, initial_load=0.3):
        # print(f"Initializing NetworkEnvironment with initial load: {initial_load}") # Less verbose
        # Cell layout (hexagonal cluster)
        side_len = 100 / np.sqrt(3) * 2 / 2 # Approximate side length for 100m distance
        self.cell_positions = np.array([
            [0, 0],
            [side_len * np.sqrt(3), 0],
            [side_len * np.sqrt(3) / 2, side_len * 3 / 2]
        ]) * 1.0 # Ensure float

        # Initial UE positions (mix of uniform and clustered)
        center_x = np.mean(self.cell_positions[:, 0])
        center_y = np.mean(self.cell_positions[:, 1])
        cluster_center = [center_x, center_y / 2]

        self.ue_positions = np.concatenate([
            np.random.uniform(self.cell_positions[:,0].min()-50, self.cell_positions[:,0].max()+50, size=(int(NUM_UES * 0.7), 2)), # Wider uniform
            np.random.normal(cluster_center, 25, size=(int(NUM_UES * 0.3), 2)) # Clustered
        ])

        # Mobility model (Random Waypoint - simplified velocity)
        self.ue_velocities = np.random.uniform(-5, 5, size=(NUM_UES, 2))

        # Initial cell loads
        self.cell_loads = np.ones(NUM_CELLS) * initial_load
        self.initial_load = initial_load

        # Sionna UMi channel model - Temporarily Commented Out due to AttributeError
        # print("Note: sn.channel.UMi initialization commented out.") # Less verbose

        # UE Priorities (1: High - VoNR, 2: Medium - Video, 3: Low - Web)
        self.ue_priorities = np.random.choice([1, 2, 3], size=NUM_UES, p=[0.3, 0.4, 0.3])

        # Convert dBm/dB to linear Watts once
        self.noise_power_watts = 10**((NOISE_POWER - 30) / 10) * BANDWIDTH # Noise power in Watts
        self.tx_power_watts = 10**((TX_POWER - 30) / 10) # TX power in Watts
        # print("NetworkEnvironment Initialized.") # Less verbose


    def update_ue_positions_and_velocities(self, dt=1.0, max_speed=5):
        """Updates UE positions based on velocity and applies simple boundary reflection."""
        self.ue_velocities += np.random.normal(0, 1, size=(NUM_UES, 2)) * dt
        speeds = np.linalg.norm(self.ue_velocities, axis=1)
        speeds[speeds < 1e-9] = 1e-9
        scale = np.minimum(1.0, max_speed / speeds)
        self.ue_velocities *= scale[:, np.newaxis]
        self.ue_positions += self.ue_velocities * dt

        min_x = self.cell_positions[:,0].min() - 100
        max_x = self.cell_positions[:,0].max() + 100
        min_y = self.cell_positions[:,1].min() - 100
        max_y = self.cell_positions[:,1].max() + 100

        hit_x_min = self.ue_positions[:, 0] <= min_x
        hit_x_max = self.ue_positions[:, 0] >= max_x
        hit_y_min = self.ue_positions[:, 1] <= min_y
        hit_y_max = self.ue_positions[:, 1] >= max_y

        self.ue_positions[hit_x_min, 0] = min_x
        self.ue_positions[hit_x_max, 0] = max_x
        self.ue_positions[hit_y_min, 1] = min_y
        self.ue_positions[hit_y_max, 1] = max_y

        self.ue_velocities[hit_x_min | hit_x_max, 0] *= -1
        self.ue_velocities[hit_y_min | hit_y_max, 1] *= -1


    def compute_metrics(self):
        """Computes RSRP, SINR based on positions and simplified channel model."""
        ue_pos_np = self.ue_positions
        cell_pos_np = self.cell_positions

        rsrp_db = np.zeros((NUM_UES, NUM_CELLS))
        sinr_db = np.zeros((NUM_UES, NUM_CELLS))

        distances = np.linalg.norm(ue_pos_np[:, np.newaxis, :] - cell_pos_np[np.newaxis, :, :], axis=2)
        distances = np.maximum(distances, 1.0)

        path_loss_db = 32.4 + 20 * np.log10(CARRIER_FREQUENCY / 1e9) + 31.0 * np.log10(distances / 1000)
        shadowing_db = np.random.normal(0, 7.0, size=(NUM_UES, NUM_CELLS))
        rsrp_db = TX_POWER - path_loss_db - shadowing_db

        linear_gain = 10**(-(path_loss_db + shadowing_db) / 10)
        received_power_watts = self.tx_power_watts * linear_gain

        interference_watts = np.zeros((NUM_UES, NUM_CELLS))
        for c_idx in range(NUM_CELLS):
            interfering_indices = [o_idx for o_idx in range(NUM_CELLS) if o_idx != c_idx]
            if interfering_indices:
                interference_watts[:, c_idx] = np.sum(received_power_watts[:, interfering_indices], axis=1)

        signal_watts = received_power_watts
        noise_plus_interference_watts = self.noise_power_watts + interference_watts
        sinr_linear = signal_watts / noise_plus_interference_watts
        sinr_linear = np.maximum(sinr_linear, 1e-20)
        sinr_db = 10 * np.log10(sinr_linear)

        return rsrp_db, sinr_db, self.cell_loads.copy(), self.ue_priorities.copy()

    def update_cell_loads(self, assignments):
        """Updates cell loads based on current UE assignments."""
        self.cell_loads = np.zeros(NUM_CELLS)
        unique_cells, counts = np.unique(assignments, return_counts=True)
        load_per_ue = 1.0 / NUM_UES
        for cell_idx, count in zip(unique_cells, counts):
             if 0 <= cell_idx < NUM_CELLS:
                 self.cell_loads[cell_idx] = count * load_per_ue
        self.cell_loads = np.clip(self.cell_loads, 0.0, 1.0)


# --- Module 2: Traffic Steering Algorithms ---
class TrafficSteering:
    """Implements different Traffic Steering algorithms."""
    def __init__(self, algorithm="baseline", rsrp_threshold=-100, hysteresis=3, ttt=0.1, load_threshold=0.8):
        self.algorithm = algorithm
        self.rsrp_threshold = rsrp_threshold
        self.hysteresis = hysteresis
        self.ttt = ttt
        self.load_threshold = load_threshold
        self.prev_assignments = None
        self.ttt_targets = {}
        # print(f"Initializing TrafficSteering with algorithm: {self.algorithm}") # Less verbose

    def assign_initial(self, rsrp):
        """Assigns UEs initially based on best RSRP."""
        self.prev_assignments = np.argmax(rsrp, axis=1)
        self.ttt_targets = {}
        # print("Initial UE assignments based on RSRP complete.") # Less verbose
        return self.prev_assignments.copy()

    def baseline_a3(self, rsrp, sinr, cell_loads, priorities, dt):
        """Implements A3 event based handover with load check."""
        if self.prev_assignments is None:
            return self.assign_initial(rsrp)

        assignments = self.prev_assignments.copy()
        current_ttt_targets_state = {k: v.copy() for k, v in self.ttt_targets.items()}

        for ue_idx in range(NUM_UES):
            current_cell = assignments[ue_idx]
            best_neighbor_quality = -np.inf
            potential_target = -1

            if ue_idx not in current_ttt_targets_state:
                 current_ttt_targets_state[ue_idx] = {}

            active_targets_for_ue = set()

            for cell_idx in range(NUM_CELLS):
                if cell_idx == current_cell:
                    continue

                a3_condition = rsrp[ue_idx, cell_idx] > rsrp[ue_idx, current_cell] + self.hysteresis
                load_condition = cell_loads[cell_idx] < self.load_threshold
                rsrp_condition = rsrp[ue_idx, cell_idx] > self.rsrp_threshold

                if a3_condition and load_condition and rsrp_condition:
                    active_targets_for_ue.add(cell_idx)
                    current_ttt_targets_state[ue_idx][cell_idx] = current_ttt_targets_state[ue_idx].get(cell_idx, 0) + dt
                    if current_ttt_targets_state[ue_idx][cell_idx] >= self.ttt:
                         if rsrp[ue_idx, cell_idx] > best_neighbor_quality:
                             best_neighbor_quality = rsrp[ue_idx, cell_idx]
                             potential_target = cell_idx

            targets_to_reset = set(current_ttt_targets_state[ue_idx].keys()) - active_targets_for_ue
            for target in targets_to_reset:
                 current_ttt_targets_state[ue_idx].pop(target, None)

            if potential_target != -1:
                assignments[ue_idx] = potential_target
                current_ttt_targets_state[ue_idx] = {}

        self.prev_assignments = assignments
        self.ttt_targets = current_ttt_targets_state
        return assignments

    def utility_based(self, rsrp, sinr, cell_loads, priorities):
        """Implements utility-based assignment."""
        assignments = np.zeros(NUM_UES, dtype=int)
        for ue_idx in range(NUM_UES):
            utilities = np.zeros(NUM_CELLS)
            for cell_idx in range(NUM_CELLS):
                w_sinr = 0.5
                w_load = 0.3
                w_prio = 0.2
                sinr_component = w_sinr * np.clip(sinr[ue_idx, cell_idx], -20, 30)
                load_component = w_load * (1.0 - cell_loads[cell_idx]) * 20
                priority_component = w_prio * (4.0 - priorities[ue_idx]) * 10
                utilities[cell_idx] = sinr_component + load_component + priority_component
            assignments[ue_idx] = np.argmax(utilities)

        self.prev_assignments = assignments
        self.ttt_targets = {}
        return assignments

    def assign_ues(self, rsrp, sinr, cell_loads, priorities, dt=1.0):
        """Assigns UEs based on the selected algorithm."""
        if self.algorithm == "baseline":
            return self.baseline_a3(rsrp, sinr, cell_loads, priorities, dt)
        elif self.algorithm == "utility":
            return self.utility_based(rsrp, sinr, cell_loads, priorities)
        else:
            # print(f"Warning: Unknown TS algorithm '{self.algorithm}'. Defaulting.") # Less verbose
            if self.prev_assignments is None:
                 self.assign_initial(rsrp)
            return self.prev_assignments.copy()

# --- Module 3: AI Fuzzer ---
class AIFuzzer:
    """
    Uses a Genetic Algorithm (GA) to find inputs (network state modifications)
    that maximize an objective function (e.g., number of handovers).
    """
    def __init__(self, env: NetworkEnvironment, ts: TrafficSteering, population_size=FUZZER_POPULATION, generations=FUZZER_GENERATIONS):
        self.env = env
        self.ts = ts
        self.population_size = population_size
        self.generations = generations
        self.input_vector_size = NUM_CELLS + NUM_UES * 2
        self.objective_call_count = 0
        # print(f"Initializing AIFuzzer with Pop: {population_size}, Gen: {generations}") # Less verbose

    def _objective_function(self, inputs, current_assignments, dt_fitness=1.0):
        """Internal helper: Applies inputs, runs TS, returns fitness (neg handovers)."""
        self.objective_call_count += 1
        original_loads = self.env.cell_loads.copy()
        original_positions = self.env.ue_positions.copy()
        original_ts_prev_assignments = self.ts.prev_assignments.copy() if self.ts.prev_assignments is not None else None
        original_ts_ttt_targets = {k: v.copy() for k, v in self.ts.ttt_targets.items()}

        try:
            load_modifier = inputs[:NUM_CELLS]
            position_modifier = inputs[NUM_CELLS:].reshape(NUM_UES, 2)
            temp_loads = np.clip(original_loads + load_modifier, 0, 1)
            temp_positions = original_positions + position_modifier

            self.env.cell_loads = temp_loads
            self.env.ue_positions = temp_positions
            rsrp, sinr, _, priorities = self.env.compute_metrics()

            self.ts.prev_assignments = current_assignments
            new_assignments = self.ts.assign_ues(rsrp, sinr, temp_loads, priorities, dt=dt_fitness)
            num_handovers = np.sum(new_assignments != current_assignments)
            fitness_score = -float(num_handovers)

        finally:
            self.env.cell_loads = original_loads
            self.env.ue_positions = original_positions
            self.ts.prev_assignments = original_ts_prev_assignments
            self.ts.ttt_targets = original_ts_ttt_targets

        return fitness_score

    def generate_inputs(self, dt=1.0):
        """Generates challenging inputs using GA."""
        if self.ts.prev_assignments is None:
            # print("Warning: Fuzzer called before initial TS assignment. Performing initial assignment.") # Less verbose
             rsrp_init, sinr_init, load_init, prio_init = self.env.compute_metrics()
             current_assignments = self.ts.assign_ues(rsrp_init, sinr_init, load_init, prio_init, dt=0)
        else:
            current_assignments = self.ts.prev_assignments

        population = []
        for _ in range(self.population_size):
            load_modifier = np.random.uniform(-0.3, 0.3, NUM_CELLS)
            position_modifier = np.random.uniform(-15, 15, (NUM_UES, 2))
            inputs = np.concatenate([load_modifier, position_modifier.flatten()])
            population.append(inputs)

        best_overall_fitness = np.inf
        best_overall_individual = population[0]
        self.objective_call_count = 0

        for gen in range(self.generations):
            fitness = [self._objective_function(ind, current_assignments, dt) for ind in population]
            sorted_indices = np.argsort(fitness)
            current_best_fitness = fitness[sorted_indices[0]]
            if current_best_fitness < best_overall_fitness:
                 best_overall_fitness = current_best_fitness
                 best_overall_individual = population[sorted_indices[0]].copy()

            new_population = [best_overall_individual.copy()] # Elitism
            num_elites = max(1, int(self.population_size * 0.2))
            parent_pool_indices = sorted_indices[:num_elites]

            for _ in range(self.population_size - 1):
                idx1, idx2 = np.random.choice(parent_pool_indices, 2, replace=True)
                parent1 = population[idx1]
                parent2 = population[idx2]
                crossover_point = np.random.randint(1, self.input_vector_size)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])

                mutation_rate = 0.1
                if np.random.rand() < mutation_rate:
                    mutation_strength_load = 0.1
                    mutation_strength_pos = 2.0
                    child[:NUM_CELLS] += np.random.normal(0, mutation_strength_load, NUM_CELLS)
                    child[NUM_CELLS:] += np.random.normal(0, mutation_strength_pos, NUM_UES * 2)
                    child[:NUM_CELLS] = np.clip(child[:NUM_CELLS], -0.5, 0.5)
                new_population.append(child)
            population = new_population
        return best_overall_individual

# --- Module 4: Oracle for Vulnerability Detection ---
class Oracle:
    """Detects predefined vulnerabilities based on simulation state."""
    # --- Thresholds are now passed during instantiation ---
    def __init__(self, ping_pong_window=3, ping_pong_threshold=1, qos_sinr_threshold=0.0, fairness_threshold=0.5):
        self.ping_pong_window = ping_pong_window
        self.ping_pong_threshold = ping_pong_threshold
        self.qos_sinr_threshold = qos_sinr_threshold # This value will be overridden
        self.fairness_threshold = fairness_threshold
        self.handover_history = {}
        # print(f"Initializing Oracle with PP_Win:{ping_pong_window}, PP_Thr:{ping_pong_threshold}, QoS_SINR:{qos_sinr_threshold}, Fair_Thr:{fairness_threshold}") # Less verbose

    def _jain_fairness(self, allocations):
        """Calculates Jain's Fairness Index."""
        allocations = np.asarray(allocations)
        allocations_cleaned = allocations[np.isfinite(allocations) & (allocations > 1e-9)]
        if len(allocations_cleaned) == 0: return 1.0
        sum_val = np.sum(allocations_cleaned)
        sum_sq_val = np.sum(allocations_cleaned**2)
        if sum_sq_val < 1e-15: return 1.0
        return sum_val**2 / (len(allocations_cleaned) * sum_sq_val)

    def evaluate(self, rsrp, sinr, assignments, cell_loads, priorities):
        """Evaluates the current state for vulnerabilities."""
        vulnerabilities_found = []

        # Ping-Pong Detection
        num_ping_pongs_detected_this_step = 0
        for ue_idx in range(NUM_UES):
            if ue_idx not in self.handover_history: self.handover_history[ue_idx] = []
            self.handover_history[ue_idx].append(assignments[ue_idx])
            while len(self.handover_history[ue_idx]) > self.ping_pong_window: self.handover_history[ue_idx].pop(0)
            history = self.handover_history[ue_idx]
            if len(history) == self.ping_pong_window and history[0] == history[2] and history[0] != history[1]:
                num_ping_pongs_detected_this_step += 1
        if num_ping_pongs_detected_this_step >= self.ping_pong_threshold:
            vulnerabilities_found.append(f"Ping-Pong: {num_ping_pongs_detected_this_step} UEs")

        # QoS Violation Detection
        assigned_sinr = np.array([sinr[ue_idx, assignments[ue_idx]] for ue_idx in range(NUM_UES)])
        high_priority_mask = (priorities == 1)
        if np.any(high_priority_mask):
             high_priority_sinr = assigned_sinr[high_priority_mask]
             avg_sinr_high = np.mean(high_priority_sinr)
             # --- Use the stricter threshold passed during init ---
             if avg_sinr_high < self.qos_sinr_threshold:
                  vulnerabilities_found.append(f"QoS Violation: Avg High Prio SINR = {avg_sinr_high:.2f} dB (Threshold: {self.qos_sinr_threshold} dB)")

        # Unfairness Detection
        assigned_sinr_linear = 10**(assigned_sinr / 10)
        fairness = self._jain_fairness(assigned_sinr_linear)
        if fairness < self.fairness_threshold:
            vulnerabilities_found.append(f"Unfairness: Jain Index = {fairness:.2f}")

        return vulnerabilities_found

# --- Module 5: Main Simulation Loop and Analysis ---

def run_simulation(scenario_name, initial_load=0.3, max_speed=5):
    """Runs the full simulation for one scenario."""
    print(f"\n--- Running Scenario: {scenario_name} (Load: {initial_load}, Speed: {max_speed}) ---")
    start_time_scenario = time.time()

    env = NetworkEnvironment(initial_load=initial_load)
    ts_baseline = TrafficSteering(algorithm="baseline")
    ts_utility = TrafficSteering(algorithm="utility")
    algorithms = {"baseline": ts_baseline, "utility": ts_utility}

    # --- Instantiate Oracle with stricter QoS threshold ---
    # Note: Other thresholds like ping_pong_threshold or fairness_threshold could also be adjusted here if needed.
    oracle = Oracle(qos_sinr_threshold=5.0)
    # --- End Oracle Instantiation Change ---

    results_list = []
    dt = 1.0

    for algo_name, ts_instance in algorithms.items():
        print(f"--- Algorithm: {algo_name} ---")
        start_time_algo = time.time()
        env = NetworkEnvironment(initial_load=initial_load)
        env.ue_velocities = np.random.uniform(-max_speed, max_speed, size=(NUM_UES, 2))
        ts_instance.prev_assignments = None
        ts_instance.ttt_targets = {}
        oracle.handover_history = {} # Reset oracle history per algorithm run

        fuzzer = AIFuzzer(env, ts_instance, population_size=FUZZER_POPULATION, generations=FUZZER_GENERATIONS)

        rsrp_init, sinr_init, load_init, prio_init = env.compute_metrics()
        _ = ts_instance.assign_ues(rsrp_init, sinr_init, load_init, prio_init, dt=0)
        current_assignments = ts_instance.prev_assignments
        if current_assignments is None: # Handle case where assign_initial might fail or not run
             print(f"ERROR: Initial assignment failed for {algo_name}")
             continue # Skip this algorithm if initial state is bad
        env.update_cell_loads(current_assignments)

        for iteration in range(SIMULATION_ITERATIONS):
            iter_start_time = time.time()
            fuzzed_inputs = fuzzer.generate_inputs(dt)

            load_modifier = fuzzed_inputs[:NUM_CELLS]
            position_modifier = fuzzed_inputs[NUM_CELLS:].reshape(NUM_UES, 2)
            env.cell_loads = np.clip(env.cell_loads + load_modifier, 0, 1)
            env.ue_positions += position_modifier
            env.update_ue_positions_and_velocities(dt, max_speed)

            rsrp, sinr, cell_loads, priorities = env.compute_metrics()
            new_assignments = ts_instance.assign_ues(rsrp, sinr, cell_loads, priorities, dt)
            env.update_cell_loads(new_assignments)
            vulnerabilities = oracle.evaluate(rsrp, sinr, new_assignments, cell_loads, priorities)

            results_list.append({
                'scenario': scenario_name,
                'iteration': iteration,
                'algorithm': algo_name,
                # 'inputs': fuzzed_inputs.tolist(), # Exclude large input vector from basic results
                'assignments': new_assignments.tolist(),
                'cell_loads': env.cell_loads.tolist(),
                'vulnerabilities': vulnerabilities
            })
            iter_end_time = time.time()
            # Optional: Print progress per iteration
            # if (iteration + 1) % 10 == 0:
            #      print(f"   Algo: {algo_name}, Iter: {iteration + 1}/{SIMULATION_ITERATIONS} done.")

        end_time_algo = time.time()
        print(f"--- Algorithm {algo_name} finished in {end_time_algo - start_time_algo:.2f} seconds ---")

    end_time_scenario = time.time()
    print(f"--- Scenario {scenario_name} finished in {end_time_scenario - start_time_scenario:.2f} seconds ---")
    return results_list

def plot_results(df):
    """Generates plots for vulnerabilities found."""
    print("\n--- Generating Plots ---")
    if df.empty:
         print("No data to plot.")
         return

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    for scenario in df['scenario'].unique():
        plt.figure(figsize=(12, 7))
        scenario_df = df[df['scenario'] == scenario].copy()
        scenario_df['vulnerability_count'] = scenario_df['vulnerabilities'].apply(len)

        for algo in scenario_df['algorithm'].unique():
            algo_df = scenario_df[scenario_df['algorithm'] == algo]
            plot_data = algo_df.groupby('iteration')['vulnerability_count'].mean()
            plt.plot(plot_data.index, plot_data.values, marker='o', linestyle='-', label=f"{algo}")

        plt.xlabel('Iteration')
        plt.ylabel('Average Number of Vulnerabilities Detected')
        plt.title(f'Vulnerabilities Detected per Iteration - Scenario: {scenario}')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(0, SIMULATION_ITERATIONS, step=max(1, SIMULATION_ITERATIONS // 10)))
        plt.ylim(bottom=0)
        # Sanitize filename for different OS
        safe_scenario_name = "".join(c for c in scenario if c.isalnum() or c in (' ', '_')).rstrip()
        plot_filename = os.path.join(plot_dir, f'vulnerabilities_{safe_scenario_name.replace(" ", "_")}.png')
        try:
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close()


def summarize_results(df):
    """Prints a text summary of the results using collections.Counter."""
    print("\n--- Results Summary ---")
    if df.empty:
        print("No results to summarize.")
        return

    # --- Overall Summary ---
    all_vuln_strings = [
        vuln_str for sublist in df['vulnerabilities']
        for vuln_str in sublist if isinstance(vuln_str, str)
    ]
    overall_counts = Counter(all_vuln_strings)

    print("Overall Vulnerability Counts:")
    if not overall_counts:
        print("  No vulnerabilities detected overall.")
    else:
        for vuln_str, count in overall_counts.most_common():
            print(f"  '{vuln_str}': {count}")
    print("-" * 20)

    # --- Per Scenario / Algorithm Summary ---
    for scenario in df['scenario'].unique():
        print(f"\nScenario: {scenario}")
        for algo in df[df['scenario'] == scenario]['algorithm'].unique():
            print(f"  Algorithm: {algo}")
            algo_df = df[(df['scenario'] == scenario) & (df['algorithm'] == algo)]
            algo_vuln_strings = [
                vuln_str for sublist in algo_df['vulnerabilities']
                for vuln_str in sublist if isinstance(vuln_str, str)
            ]

            if not algo_vuln_strings:
                print("    No vulnerabilities detected.")
            else:
                algo_counts = Counter(algo_vuln_strings)
                for vuln_str, count in algo_counts.most_common():
                    print(f"    '{vuln_str}': {count} occurrences")
    print("\n--- End Summary ---")


def main():
    """Main function to run scenarios, collect data, and analyze."""
    print("--- Starting AI Fuzzing Simulation ---")
    start_time_main = time.time()

    # --- Configure TensorFlow to use only one GPU ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"--- Configured to use 1 Physical GPU: {gpus[0].name}, {len(logical_gpus)} Logical GPU(s) available. ---")
        except RuntimeError as e:
            print(f"Error setting visible devices (must be set at program start): {e}")
            print("Proceeding with default GPU configuration.")
    else:
        print("--- No GPU detected by TensorFlow. Running on CPU. ---")
    # --- End GPU Configuration ---

    # Define scenarios: (name, initial_load, max_speed)
    scenarios_to_run = [
        ('Low Load', 0.3, 5),
        ('High Load', 0.7, 5),
        ('High Mobility', 0.5, 10)
    ]

    all_results_data = []
    for name, load, speed in scenarios_to_run:
        np.random.seed(42) # Reset seed for each scenario run for comparability
        tf.random.set_seed(42)
        results = run_simulation(scenario_name=name, initial_load=load, max_speed=speed)
        all_results_data.extend(results)

    if not all_results_data:
         print("Simulation produced no results.")
         return

    results_df = pd.DataFrame(all_results_data)
    csv_filename = 'fuzzing_results_long.csv' # Changed filename for new run
    try:
        results_df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"\n--- Results saved to {csv_filename} ---")
    except Exception as e:
        print(f"Error saving results to CSV {csv_filename}: {e}")

    summarize_results(results_df)
    plot_results(results_df)

    end_time_main = time.time()
    print(f"\n--- Simulation Finished in {end_time_main - start_time_main:.2f} seconds ---")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()