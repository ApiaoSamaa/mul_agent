import streamlit as st
import networkx as nx

# Get user input
N = st.number_input('Number of nodes', value=10)
K = st.number_input('Mean degree', value=4)
beta = st.slider('Beta', min_value=0.0, max_value=1.0, step=0.01)

# Generate WS graph
G = nx.watts_strogatz_graph(N, K, beta)

# Draw graph
st.write('WS Graph')
st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

# Calculate average shortest path length
avg_shortest_path_length = nx.average_shortest_path_length(G)
st.write(f'Average shortest path length: {avg_shortest_path_length}')

# Calculate average clustering coefficient
avg_clustering_coefficient = nx.average_clustering(G)
st.write(f'Average clustering coefficient: {avg_clustering_coefficient}')