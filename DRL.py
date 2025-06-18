import os
import numpy as np
import pandas as pd
import random
import networkx as nx
import collections
from typing import List
import gym
from gym import spaces
from stable_baselines3 import PPO


def load_flows(csv_file):
    df = pd.read_csv(csv_file)
    flows = []
    for idx, row in df.iterrows():
        flow = {
            "id": row["id"],
            "talker": row["talker"],
            "listener": row["listener"],
            "frame_size": float(row["frame_size"]),
            "period": float(row["period"]),
            "deadline": float(row["deadline"]),
            "release_time": float(row["release_time"]),
            "queue": int(row["queue"])
        }
        flows.append(flow)
    return flows


def compute_paths(flows, G):
    for flow in flows:
        try:
            path = nx.shortest_path(G, source=flow["talker"], target=flow["listener"])
        except nx.NetworkXNoPath:
            path = []
        flow["path"] = path
    return flows


def compute_metrics(flows, link_available, G):
    total_flows = len(flows)
    if total_flows == 0:
        return 100.0, 0.0, 100.0

    successful = [f for f in flows if f["finish_time"] <= f["deadline_time"]]
    success_rate = (len(successful) / total_flows) * 100.0

    lat_sum = 0.0
    for f in flows:
        if f.get("finish_time") > f.get("deadline_time"):
            lat_sum += (f["finish_time"] - f["arrival_time"])
        else:
            lat_sum += (f["finish_time"] - f["arrival_time"])

    avg_latency = lat_sum / total_flows

    trunk_links = []
    for u, v in G.edges():
        if G.nodes[u].get("type") == "switch" and \
           (G.nodes[v].get("type") == "switch" or "Central" in v):
            trunk_links.append((u, v))

    simulation_window = max(f["finish_time"] for f in flows)
    total_available = simulation_window * len(trunk_links)
    total_occupied = sum(link_available.get(edge, 0.0) for edge in trunk_links)

    if total_available <= 0.0:
        idle_percentage = 100.0
    else:
        used_pct = (total_occupied / total_available) * 100.0
        idle_percentage = max(0.0, 100.0 - used_pct)
    return success_rate, avg_latency, idle_percentage


def define_zonal_topology(num_zones=6):
    G = nx.DiGraph()
    central_switch = "Central_Switch"
    G.add_node(central_switch, type="switch", ports=7)
    central_computer = "Central_Computer"
    G.add_node(central_computer, type="endpoint")
    G.add_edge(central_switch, central_computer, capacity_mbps=100, portA=0, portB=0)
    G.add_edge(central_computer, central_switch, capacity_mbps=100, portA=0, portB=0)

    for z in range(num_zones):
        zsw = f"Zone_{z}_Switch"
        zc = f"Zone_{z}_Controller"
        s0 = f"Zone_{z}_Sensor0"
        s1 = f"Zone_{z}_Sensor1"
        s2 = f"Zone_{z}_Sensor2"

        G.add_node(zsw, type="switch", ports=5)
        G.add_node(zc, type="endpoint")
        G.add_node(s0, type="endpoint")
        G.add_node(s1, type="endpoint")
        G.add_node(s2, type="endpoint")
        G.add_edge(central_switch, zsw, capacity_mbps=100, portA=(z + 1), portB=0)
        G.add_edge(zsw, central_switch, capacity_mbps=100, portA=0, portB=(z + 1))
        G.add_edge(zsw, zc, capacity_mbps=100, portA=1, portB=0)
        G.add_edge(zc, zsw, capacity_mbps=100, portA=0, portB=1)
        G.add_edge(zsw, s0, capacity_mbps=100, portA=2, portB=0)
        G.add_edge(s0, zsw, capacity_mbps=100, portA=0, portB=2)
        G.add_edge(zsw, s1, capacity_mbps=100, portA=3, portB=0)
        G.add_edge(s1, zsw, capacity_mbps=100, portA=0, portB=3)
        G.add_edge(zsw, s2, capacity_mbps=100, portA=4, portB=0)
        G.add_edge(s2, zsw, capacity_mbps=100, portA=0, portB=4)

    return G


class TASEnv(gym.Env):
    BYTES_PER_MS = (10 * 1e6) / 8 / 10000.0 

    def __init__(self,
                 scenario_files: List[str],
                 G: nx.DiGraph,
                 max_flows=50,
                 alpha=0.01,
                 num_queues=8,
                 max_segments=10):
        super().__init__()
        self.scenario_files = scenario_files
        self.G = G
        self.max_flows = max_flows
        self.alpha = alpha
        self.num_queues = num_queues
        self.max_segments = max_segments

        self.num_features = 7

        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(self.num_queues + 1,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.max_flows * self.num_features,),
                                            dtype=np.float32)

        self.current_step_count = 0
        self.current_flows = []
        self.num_flows = 0
        self.link_available = {}
        self.sim_time = 0.0
        self.done = False
        self.link_intervals = collections.defaultdict(list)

        self.meet_deadline_reward = 0.1
        self.miss_deadline_penalty = -0.1
        self.invalid_action_penalty = -0.01
        self.w_idle = 0.1  


    def _get_observation(self):
        DEADLINE_DIV = 10000.0
        SIZE_DIV = 9000.0
        RELEASE_DIV = 10000.0
        PATHLEN_DIV = 20.0
        EARLIEST_DIV = 200000.0

        obs = np.zeros((self.max_flows, self.num_features), dtype=np.float32)

        for i in range(self.num_flows):
            f = self.current_flows[i]
            not_finished_flag = 1.0 if (f["finish_time"] is None) else 0.0
            d_norm = min(f["deadline"] / DEADLINE_DIV, 1.0)
            s_norm = min(f["frame_size"] / SIZE_DIV, 1.0) if not_finished_flag > 0.5 else 0.0
            r_norm = min(f["release_time"] / RELEASE_DIV, 1.0)
            path_len = max(0, len(f["path"]) - 1)
            p_norm = min(path_len / PATHLEN_DIV, 1.0)

            if not_finished_flag > 0.5:
                next_start_est = max(self.sim_time, f["arrival_time"])
                est_norm = min(next_start_est / EARLIEST_DIV, 1.0)
            else:
                est_norm = 0.0

            slack = (f["deadline_time"] - self.sim_time) - \
                    (f["frame_size"] / self.BYTES_PER_MS)
            urgency = max(0.0, min(slack / f["deadline"], 1.0))

            obs[i, 0] = not_finished_flag
            obs[i, 1] = d_norm
            obs[i, 2] = s_norm
            obs[i, 3] = r_norm
            obs[i, 4] = p_norm
            obs[i, 5] = est_norm
            obs[i, 6] = urgency

        return obs.flatten()

    def reset(self):
        self.current_step_count = 0
        self.sim_time = 0.0
        self.done = False
        self.link_available = {}
        self.link_intervals.clear()

        scenario_path = random.choice(self.scenario_files)
        flows = load_flows(scenario_path)
        flows = compute_paths(flows, self.G)
        flows = [f for f in flows if f["path"]]
        if len(flows) > self.max_flows:
            flows = flows[:self.max_flows]
        self.current_flows = flows
        self.num_flows = len(flows)

        for f in self.current_flows:
            f["arrival_time"] = f["release_time"]
            f["deadline_time"] = f["release_time"] + f["deadline"]
            f["finish_time"] = None

        return self._get_observation()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, {}

        gate_mask = action[:self.num_queues]
        seg_len_norm = action[-1]

        gates_open = [i for i in range(self.num_queues) if gate_mask[i] >= 0.5]
        segment_length = seg_len_norm * 20000.0
        if segment_length < 1.0:
            segment_length = 1.0

        reward = 0.0
        info = {}

        if len(gates_open) == 0:
            reward += self.invalid_action_penalty

        start_t = self.sim_time
        end_t = self.sim_time + segment_length

        for f in self.current_flows:
            if f["finish_time"] is not None or f["queue"] not in gates_open:
                continue

            path = f["path"]
            current_time = max(f["arrival_time"], start_t)
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge not in self.link_available:
                    self.link_available[edge] = 0.0
                capacity = self.G[edge[0]][edge[1]]["capacity_mbps"]
                bytes_per_ms = (capacity * 1e6) / 8 / 10000.0
                tx_time = f["frame_size"] / bytes_per_ms

                earliest_start = max(self.link_available[edge], current_time)
                finish_time = earliest_start + tx_time

                if finish_time > end_t:
                    portion = end_t - earliest_start
                    if portion > 0:
                        used_up = earliest_start + portion
                        self.link_available[edge] = used_up
                        self.link_intervals[edge].append((earliest_start, used_up))
                        leftover_time = (finish_time - end_t)
                        leftover_bytes = (leftover_time / tx_time) * f["frame_size"]
                        f["frame_size"] = leftover_bytes
                    break
                else:
                    self.link_available[edge] = finish_time
                    self.link_intervals[edge].append((earliest_start, finish_time))
                    current_time = finish_time
            else:
                f["finish_time"] = current_time

        for f in self.current_flows:
            if f["finish_time"] is not None:
                if start_t <= f["finish_time"] <= end_t:
                    if f["finish_time"] <= f["deadline_time"]:
                        reward += self.meet_deadline_reward
                    else:
                        reward += self.miss_deadline_penalty

        self.sim_time = end_t
        self.current_step_count += 1
        all_finished = all(f["finish_time"] is not None for f in self.current_flows)
        if self.current_step_count >= self.max_segments or all_finished:
            self.done = True

        if self.done:
            sr, avg_lat, old_idle = self._finalize_and_metrics()
            idle = self._compute_correct_idle()
            final_r = (sr / 100.0) - (self.alpha * avg_lat) + (self.w_idle * (idle / 100.0))
            reward += final_r
            info["sr"] = sr
            info["avg_lat"] = avg_lat
            info["idle"] = idle

        return self._get_observation(), reward, self.done, info


    def _finalize_and_metrics(self):
        for f in self.current_flows:
            if f["finish_time"] is None:
                f["finish_time"] = f["deadline_time"] + 999999
        sr, avg_lat, idle_ = compute_metrics(self.current_flows, self.link_available, self.G)
        return sr, avg_lat, idle_

    def _compute_correct_idle(self):
        if not self.current_flows:
            return 100.0
        real_end = max(f["finish_time"] for f in self.current_flows)
        if real_end <= 0.0:
            return 100.0
        all_intervals = []
        for edge, intervals in self.link_intervals.items():
            all_intervals.extend(intervals)
        if not all_intervals:
            return 100.0
        all_intervals.sort(key=lambda x: x[0])
        merged = []
        cur_start, cur_end = all_intervals[0]
        for st, ed in all_intervals[1:]:
            if st <= cur_end:
                cur_end = max(cur_end, ed)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = st, ed
        merged.append((cur_start, cur_end))
        total_busy = sum(interval[1] - interval[0] for interval in merged)
        idle_time = real_end - total_busy
        if idle_time < 0.0:
            idle_time = 0.0
        idle_pct = 100.0 * (idle_time / real_end)
        return idle_pct



def main():
    # PATH TO SAVED MODEL
    model_path = "ppo_final_model_tas.zip"
    if not os.path.isfile(model_path):
        print(f"Model not found at {model_path}")
        return
    model = PPO.load(model_path)
    print(f"Loaded PPO model from {model_path}")

    # PATH TO TEST DATA
    base_dir = "scenarios-test"
    classes = ["A", "B", "C", "D", "E"]
    scenario_files = []
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if os.path.isdir(cls_dir):
            csvs = [os.path.join(cls_dir, f)
                    for f in os.listdir(cls_dir)
                    if f.endswith(".csv")]
            scenario_files.extend(csvs)

    if not scenario_files:
        print("No scenario files found")
        return

    G = define_zonal_topology(num_zones=6)
    max_flows = 50
    results = []
    results_by_class = {c: [] for c in classes}

    for scenario_path in scenario_files:
        env = TASEnv([scenario_path], G, max_flows=max_flows,
                     alpha=0.01, num_queues=8, max_segments=10)

        obs = env.reset()
        done = False

        while not done:
            obs_tensor = np.array([obs], dtype=np.float32)
            action, _states = model.predict(obs_tensor, deterministic=True)
            chosen_action = action[0]
            new_obs, reward, done, info = env.step(chosen_action)
            obs = new_obs

        sr = info.get("sr", 0.0)
        avg_lat = info.get("avg_lat", 9999.0)
        idle = info.get("idle", 0.0)

        scenario_file = os.path.basename(scenario_path)
        c = next((cc for cc in classes if cc in scenario_path), "UNKNOWN")

        results_by_class[c].append((sr, avg_lat, idle))
        results.append((scenario_file, sr, avg_lat, idle))

    print("\n=== PPO Test Results (Time-Aware Shaper) ===")
    for (scenario_file, sr, lat, idle) in results:
        print(f"{scenario_file}: SR={sr:.2f}%, Lat={lat:.2f}ms, Idle={idle:.2f}%")

    print("\n--- Summary by class ---")
    df = pd.DataFrame(columns=[
        "Class", "SuccessRate", "AvgLatency", "IdleTime",
        "SuccessRate_min", "SuccessRate_max",
        "AvgLatency_min", "AvgLatency_max",
        "IdleTime_min", "IdleTime_max"
    ])

    for c in classes:
        data = results_by_class[c]
        if not data:
            continue
        sr_list = [d[0] for d in data]
        lat_list = [d[1] for d in data]
        idle_list = [d[2] for d in data]
        avg_sr = np.mean(sr_list)
        avg_lat = np.mean(lat_list)
        avg_idle = np.mean(idle_list)
        print(f"Class {c}: #Scenarios={len(data)}, "
              f"AvgSR={avg_sr:.2f}%, AvgLat={avg_lat:.2f}ms, AvgIdle={avg_idle:.2f}%")

        df.loc[len(df)] = [
            c,
            avg_sr,
            avg_lat,
            avg_idle,
            np.min(sr_list), np.max(sr_list),
            np.min(lat_list), np.max(lat_list),
            np.min(idle_list), np.max(idle_list)
        ]
    # TEST RESULT PATH
    df.to_excel("DRL.xlsx", index=False)
    print("\nSaved DRL results to DRL.xlsx")


if __name__ == "__main__":
    main()
