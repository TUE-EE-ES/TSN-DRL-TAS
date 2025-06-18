import os
import numpy as np
import pandas as pd
import networkx as nx

from zonal_topology import define_zonal_topology

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
            "queue": int(row["queue"]),
            "n_per_period": int(row["n_per_period"])
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

def static_edf_scheduler(flows, G):
    for flow in flows:
        flow["arrival_time"] = flow["release_time"]
        flow["deadline_time"] = flow["arrival_time"] + flow["deadline"]
    flows = [f for f in flows if f["path"]]

    for flow in flows:
        est_tx_time = 0.0
        for i in range(len(flow["path"]) - 1):
            edge = (flow["path"][i], flow["path"][i+1])
            capacity = G[edge[0]][edge[1]]["capacity_mbps"]
            bytes_per_ms = (capacity * 1e6) / 8 / 10000.0
            est_tx_time += flow["frame_size"] / bytes_per_ms
        flow["est_tx_time"] = est_tx_time
        flow["slack"] = (flow["deadline_time"] - flow["arrival_time"]) - est_tx_time

    flows = sorted(flows, key=lambda f: f["slack"], reverse=True)

    link_available = {}
    for flow in flows:
        for i in range(len(flow["path"]) - 1):
            edge = (flow["path"][i], flow["path"][i+1])
            if edge not in link_available:
                link_available[edge] = 0.0

    gcl_events = {edge: [] for edge in link_available.keys()}

    for flow in flows:
        current_time = flow["arrival_time"]
        for i in range(len(flow["path"]) - 1):
            edge = (flow["path"][i], flow["path"][i+1])
            capacity = G[edge[0]][edge[1]]["capacity_mbps"]
            bytes_per_ms = (capacity * 1e6) / 8 / 10000.0
            tx_time = flow["frame_size"] / bytes_per_ms

            start_time = max(current_time, link_available[edge])
            finish_time = start_time + tx_time

            link_available[edge] = finish_time
            gcl_events[edge].append({
                "start": start_time,
                "finish": finish_time,
                "queue": flow["queue"]
            })
            current_time = finish_time

        flow["finish_time"] = current_time

    return flows, link_available, gcl_events

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    merged = []
    current_start, current_end = intervals_sorted[0]

    for i in range(1, len(intervals_sorted)):
        start, end = intervals_sorted[i]
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged

def compute_all_links_post_compression_idle(gcl_events, flows):
    if not flows:
        return 100.0, 0.0

    real_end = max(f["finish_time"] for f in flows)
    if real_end <= 0.0:
        return 100.0, 0.0

    all_intervals = []
    for edge, events in gcl_events.items():
        for ev in events:
            all_intervals.append((ev["start"], ev["finish"]))

    if not all_intervals:
        return 100.0, 0.0

    merged = merge_intervals(all_intervals)

    total_busy = sum((m[1] - m[0]) for m in merged)
    if total_busy < 0:
        total_busy = 0
    idle_time = real_end - total_busy
    if idle_time < 0:
        idle_time = 0

    idle_pct = 100.0 * (idle_time / real_end)
    return idle_pct, total_busy

def compute_metrics(flows, link_available, G, gcl_events):
    total_flows = len(flows)
    if total_flows == 0:
        return 100.0, 0.0, 100.0

    successful = [f for f in flows if f["finish_time"] <= f["deadline_time"]]
    success_rate = (len(successful) / total_flows) * 100.0
    lat_sum = 0.0
    for f in flows:
        if f["finish_time"] > f["deadline_time"]:
            lat_sum += (f["finish_time"] - f["arrival_time"])
        else:
            lat_sum += (f["finish_time"] - f["arrival_time"])
    avg_latency = lat_sum / total_flows
    idle_percentage, _ = compute_all_links_post_compression_idle(gcl_events, flows)

    return success_rate, avg_latency, idle_percentage

def generate_gcls_for_trunk_ports(gcl_events, G, simulation_window):
    trunk_links = []
    for u, v in G.edges():
        if G.nodes[u].get("type") == "switch" and (G.nodes[v].get("type") == "switch" or "Central" in v):
            trunk_links.append((u, v))

    gcl_schedules = {}
    for edge in trunk_links:
        events = gcl_events.get(edge, [])
        events = sorted(events, key=lambda x: x["start"])
        gcl = []
        current_time = 0.0
        for ev in events:
            if current_time < ev["start"]:
                gcl.append({
                    "start_time": current_time,
                    "end_time": ev["start"],
                    "state": "closed",
                    "queue": None
                })
            gcl.append({
                "start_time": ev["start"],
                "end_time": ev["finish"],
                "state": "open",
                "queue": ev["queue"]
            })
            current_time = ev["finish"]
        if current_time < simulation_window:
            gcl.append({
                "start_time": current_time,
                "end_time": simulation_window,
                "state": "closed",
                "queue": None
            })
        gcl_schedules[edge] = gcl
    return gcl_schedules

def process_scenarios_for_class(class_dir):
    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".csv")]
    sr_list, lat_list, idle_list = [], [], []
    G = define_zonal_topology(num_zones=6)

    for csv_file in files:
        flows = load_flows(csv_file)
        flows = compute_paths(flows, G)
        flows, link_available, gcl_events = static_edf_scheduler(flows, G)
        sr, avg_lat, idle = compute_metrics(flows, link_available, G, gcl_events)

        if flows:
            simulation_window = max(f["finish_time"] for f in flows)
        else:
            simulation_window = 0.0
        _ = generate_gcls_for_trunk_ports(gcl_events, G, simulation_window)

        sr_list.append(sr)
        lat_list.append(avg_lat)
        idle_list.append(idle)

    return sr_list, lat_list, idle_list

def main():
    # PATH TO TEST DATA
    base_dir = "scenarios-test"
    classes = ["A", "B", "C", "D", "E"]
    results = {}

    df = pd.DataFrame(columns=[
        "Class", "SuccessRate", "AvgLatency", "IdleTime",
        "SuccessRate_min", "SuccessRate_max",
        "AvgLatency_min", "AvgLatency_max",
        "IdleTime_min", "IdleTime_max"
    ])

    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(class_dir):
            print(f"Directory for class {cls} not found in {base_dir}.")
            continue

        sr_list, lat_list, idle_list = process_scenarios_for_class(class_dir)
        if len(sr_list) == 0:
            continue
        sr_mean = np.mean(sr_list)
        lat_mean = np.mean(lat_list)
        idle_mean = np.mean(idle_list)

        sr_min, sr_max = np.min(sr_list), np.max(sr_list)
        lat_min, lat_max = np.min(lat_list), np.max(lat_list)
        idle_min, idle_max = np.min(idle_list), np.max(idle_list)

        results[cls] = (sr_mean, lat_mean, idle_mean)

        df.loc[len(df)] = [
            cls,
            sr_mean,
            lat_mean,
            idle_mean,
            sr_min,
            sr_max,
            lat_min,
            lat_max,
            idle_min,
            idle_max
        ]

    print("\nStatic EDF Scheduling (All-Link Post-Compression Idle) Results (per class):")
    for cls in classes:
        if cls in results:
            sr, avg_lat, idle = results[cls]
            print(f"  Class {cls}: Success Rate: {sr:.2f}%, Avg Latency: {avg_lat:.2f} ms, Idle: {idle:.2f}%")

    # RESULT PATH
    df.to_excel("edf.xlsx", index=False)
    print("\nSaved EDF results to edf.xlsx")

if __name__ == "__main__":
    main()
