import networkx as nx

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

        G.add_edge(central_switch, zsw, capacity_mbps=100, portA=(z+1), portB=0)
        G.add_edge(zsw, central_switch, capacity_mbps=100, portA=0, portB=(z+1))

        G.add_edge(zsw, zc, capacity_mbps=100, portA=1, portB=0)
        G.add_edge(zc, zsw, capacity_mbps=100, portA=0, portB=1)

        G.add_edge(zsw, s0, capacity_mbps=100, portA=2, portB=0)
        G.add_edge(s0, zsw, capacity_mbps=100, portA=0, portB=2)
        G.add_edge(zsw, s1, capacity_mbps=100, portA=3, portB=0)
        G.add_edge(s1, zsw, capacity_mbps=100, portA=0, portB=3)
        G.add_edge(zsw, s2, capacity_mbps=100, portA=4, portB=0)
        G.add_edge(s2, zsw, capacity_mbps=100, portA=0, portB=4)

    return G
