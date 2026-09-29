import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import math
import heapq

# 1. Hospital Coordinates & Graph
locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 6.0},
    "Laboratory": {"Emergency_Ward": 3.2},
    "Emergency_Ward": {}
}

# 2. Heuristic Function
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 3. Search Algorithms
def greedy_best_first_search(start, goal, graph):
    pq = []
    heapq.heappush(pq, (heuristic(start, goal), start))
    came_from = {start: None}
    visited = set()

    while pq:
        _, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == goal:
            break
            
        for neighbor in graph[current]:
            if neighbor not in visited:
                came_from[neighbor] = current
                heapq.heappush(pq, (heuristic(neighbor, goal), neighbor))

    # Reconstruct path
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()

    # Calculate actual path cost
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += graph[path[i]][path[i+1]]
        
    return path, total_cost

def a_star_search(start, goal, graph):
    pq = []
    heapq.heappush(pq, (heuristic(start, goal), start))
    came_from = {}
    g_cost = {start: 0}
    
    while pq:
        _, current = heapq.heappop(pq)
            
        if current == goal:
            break
            
        for neighbor, cost in graph[current].items():
            tentative_g_cost = g_cost[current] + cost
            
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic(neighbor, goal)
                heapq.heappush(pq, (f_cost, neighbor))
                
    # Reconstruct path
    path = []
    curr = goal
    while curr in came_from:
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    path.reverse()
    
    return path, g_cost.get(goal, 0)

# 4. Streamlit User Interface
st.set_page_config(page_title="Search Algorithm Visualization", layout="centered")
st.title("Interactive Search Visualization")
st.markdown("Compare Greedy Best-First Search (GBFS) and A* Search on the Emergency Supply Robot Network.")

col1, col2, col3 = st.columns(3)

with col1:
    start_node = st.selectbox("Select Initial Node", list(locations.keys()), index=0)
with col2:
    goal_node = st.selectbox("Select Goal Node", list(locations.keys()), index=5)
with col3:
    algorithm = st.selectbox("Select Algorithm", ["A* Search", "Greedy Best-First Search"])

if st.button("Run Search", type="primary"):
    
    # Run the selected algorithm
    if algorithm == "A* Search":
        path, cost = a_star_search(start_node, goal_node, hospital_graph)
    else:
        path, cost = greedy_best_first_search(start_node, goal_node, hospital_graph)
    
    # 5. NetworkX Visualization
    G = nx.DiGraph()
    for node, neighbors in hospital_graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = locations
    fig, ax = plt.subplots(figsize=(10, 6))

    # Base graph plotting
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=2000, font_size=9, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    # Highlight solution path
    if len(path) > 1 or start_node == goal_node:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightgreen', node_size=2000, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3, ax=ax)

    ax.set_title(f"{algorithm} Route", fontsize=14, fontweight="bold")
    plt.axis("off")
    
    # Display the network graph
    st.pyplot(fig)

    # Display results below the graph
    st.success(f"**Selected Algorithm:** {algorithm}")
    st.info(f"**Solution Path:** {' → '.join(path)}")
    st.warning(f"**Total Path Cost:** {cost:.2f}")