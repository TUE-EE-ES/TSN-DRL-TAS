# TSN-DRL-TAS

This repository contains the source code for two related papers on
Deep-Reinforcement-Learning-based scheduling for the Time-Aware Shaper (TAS) in
in-vehicle Time-Sensitive Networks. Each paper has its own branch:

- **`main`** — *Fast Time-Aware Shaper Scheduling for In-Vehicle Networks via
  Deep Reinforcement Learning* (IEEE Internet of Things Journal, 2026).
- **`VTC-Version`** — *Deep-Reinforcement-Learning-based Scheduler for Time-Aware
  Shaper in In-Vehicle Networks* (IEEE VTC2025-Spring). The earlier conference
  work that the journal paper extends.

# 🔬 Reference

If you use this work, please cite:

M. Karimi, M. Nabi, A. Nelson, K. Goossens, and T. Basten, "Fast Time-Aware
Shaper Scheduling for In-Vehicle Networks via Deep Reinforcement Learning,"
*IEEE Internet of Things Journal*, early access, 2026,
doi: 10.1109/JIOT.2026.3695413.

M. Karimi, M. Nabi, A. Nelson, K. Goossens, and T. Basten,
"Deep-Reinforcement-Learning-based Scheduler for Time-Aware Shaper in In-Vehicle
Networks," in *Proc. 2025 IEEE 101st Vehicular Technology Conf.
(VTC2025-Spring)*, Oslo, Norway, Jun. 17–20, 2025.

# 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

# 📝 Abstract(s)

**Fast Time-Aware Shaper Scheduling for In-Vehicle Networks via Deep
Reinforcement Learning (IEEE Internet of Things Journal, 2026)**

Modern vehicles increasingly rely on distributed computing platforms that
exchange large volumes of sensor and control data with strict timing
requirements. Ensuring that this traffic meets its deadlines over Ethernet-based
in-vehicle networks requires Time-Sensitive Networking (TSN) and, in particular,
effective configuration of the Time-Aware Shaper (TAS). However, generating and
updating TAS schedules that remain valid as traffic patterns evolve is an
NP-hard problem that traditional optimization or heuristic methods address only
partially. This paper introduces a Deep Reinforcement Learning (DRL) scheduler
that learns to configure TAS schedules directly from network state while
preserving standard compliance through analytical validation. The proposed DRL
scheduler encodes the scenario (network topology and workload) of the in-vehicle
network using a Graph Neural Network (GNN) and learns scheduling policies that
balance deadline satisfaction, latency, and resource utilization. Evaluation on
a comprehensive benchmark shows that the proposed approach consistently
outperforms state-of-the-art heuristics and a topology-specific DRL baseline,
achieving higher success rate and lower delay while maintaining efficient
bandwidth use. Once trained, it can adapt to new traffic scenarios within
milliseconds, demonstrating the potential of the DRL-based scheduler as a
foundation for adaptive and reliable communication in next-generation
software-defined vehicles.

**Deep-Reinforcement-Learning-based Scheduler for Time-Aware Shaper in
In-Vehicle Networks (IEEE VTC2025-Spring)**

As vehicles develop into software-defined platforms with powerful automated
driving capabilities and driver support systems, their in-vehicle networks
become significantly more complicated. A key technique for ensuring
deterministic, low-latency connectivity for crucial data traffic in such
settings is Time-Sensitive Networking (TSN), and specifically the Time-Aware
Shaper (TAS). However, current TAS scheduling techniques have difficulty
adjusting schedules to dynamically shifting traffic patterns and changing
operating conditions. This paper presents an adaptive scheduler using Deep
Reinforcement Learning (DRL), which aims to meet strict deadlines, reducing
latency and providing near-ideal resource usage. Experimental results for
different vehicle scenarios show that our DRL-based scheduler performs better in
terms of success rate, low latency, and overall network performance than
state-of-the-art heuristic algorithms such as earliest deadline first (EDF)
scheduling.

# 📬 Contact Us

For questions, comments, or collaborations, feel free to contact us:
Mohammadparsa Karimi — m.karimi@tue.nl

# Acknowledgments

This work has received funding from the European Chips Joint Undertaking under
Framework Partnership Agreement No 101139789 (HAL4SDV).
