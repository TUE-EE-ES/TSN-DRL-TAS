from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple


class SimpleDiGraph(dict):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._node_attrs: Dict[str, Dict[str, object]] = {}

    def add_node(self, node: str, **attrs: object) -> None:
        if node not in self:
            super().__setitem__(node, {})
        self._node_attrs.setdefault(node, {}).update(attrs)

    def add_edge(self, u: str, v: str, **attrs: object) -> None:
        if u not in self:
            self[u] = {}
        if v not in self:
            self[v] = {}
        self[u][v] = attrs

    def nodes(self) -> Iterable[str]:
        return self.keys()

    def edges(self) -> Iterable[Tuple[str, str]]:
        for u in self:
            for v in self[u]:
                yield (u, v)

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        if source == target:
            return [source]
        visited = {source}
        queue: deque[Tuple[str, List[str]]] = deque([(source, [source])])
        while queue:
            current, path = queue.popleft()
            for neighbour in self[current]:
                if neighbour in visited:
                    continue
                if neighbour == target:
                    return path + [neighbour]
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))
        return None

    def __getitem__(self, item: str) -> Dict[str, object]:
        return super().__getitem__(item)

    def node_attributes(self, node: str) -> Dict[str, object]:
        return self._node_attrs.get(node, {})


def define_zonal_topology(num_zones: int = 6) -> SimpleDiGraph:
    G = SimpleDiGraph()

    central_switch = "Central_Switch"
    central_computer = "Central_Computer"
    G.add_node(central_switch, type="switch", ports=num_zones + 1)
    G.add_node(central_computer, type="endpoint", role="central")
    G.add_edge(central_switch, central_computer, capacity_mbps=100.0, portA=0, portB=0)
    G.add_edge(central_computer, central_switch, capacity_mbps=100.0, portA=0, portB=0)

    for z in range(num_zones):
        zsw = f"Zone_{z}_Switch"
        zc = f"Zone_{z}_Controller"
        sensors = [f"Zone_{z}_Sensor{i}" for i in range(3)]

        G.add_node(zsw, type="switch", ports=5)
        G.add_node(zc, type="endpoint", role="controller")
        for s in sensors:
            G.add_node(s, type="endpoint", role="sensor")

        G.add_edge(central_switch, zsw, capacity_mbps=100.0, portA=z + 1, portB=0)
        G.add_edge(zsw, central_switch, capacity_mbps=100.0, portA=0, portB=z + 1)
        G.add_edge(zsw, zc, capacity_mbps=100.0, portA=1, portB=0)
        G.add_edge(zc, zsw, capacity_mbps=100.0, portA=0, portB=1)
        for i, sensor in enumerate(sensors):
            port_offset = 2 + i
            G.add_edge(zsw, sensor, capacity_mbps=100.0, portA=port_offset, portB=0)
            G.add_edge(sensor, zsw, capacity_mbps=100.0, portA=0, portB=port_offset)

    return G
