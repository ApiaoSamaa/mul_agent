"""
task 1: 全局耦合网络
实现全局耦合网络，展示网络图，并计算平均路径长度和聚类系数。

网络描述
无向图，任意两个节点之间均有边相连
基本性质:
    K_{GC} = |A| - 1 (节点的度)
    L_{GC} = 1 (平均路径长度)
    C_{GC} = 1 (聚类系数)
    
平均路径长度(L):
任意两节点间最短路径长度d的平均值（最短路径长度可由广度优先搜索算法来确定）

task 2,3,4:

1.2 实现最近邻耦合网络，展示网络图，并计算平均路径长度和聚类系数。
1.3 实现星形网络，展示网络图，并计算平均路径长度和聚类系数。

"""
from utils import *
import streamlit as st
from draw_graphs import *
import matplotlib.pyplot as pl
from boid import *
from matplotlib.animation import FuncAnimation
from math import sqrt

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

OptionSequence = ["1.1: global_coupled_network",'1.2: nearest_neighbor_coupling_network', '1.3: star_shape_network', '2.1 Watts_and_Strogtz_network','2.2: Norman_and_Watts_network', '6.1 Boids_model']
option = st.sidebar.selectbox("选择问题类型",options=OptionSequence,index=0)


k = 2
center = 0
probability = 0.5
save = 'True'
speed = 200



if (option!='2.1 Watts_and_Strogtz_network' and option!='6.1 Boids_model' and option!='2.2: Norman_and_Watts_network'):
    nodes_num = st.sidebar.slider("网络中点的数量", 3, 20, 7, 1)

    st.title(option)
    st.subheader('交互结构的邻接矩阵表示')

    if option=="1.1: global_coupled_network":
        df = draw_mat(nodes_num, option)
        st.write(df)
        mat = np.array(df)
        
        # Set the labels for each node
        labels = {}
        nodes = list(range(nodes_num))
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for n in graph.nodes():
            labels[n] = str(n)
        edges = []
        # Generate a list of nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append([i, j])
        fig, ax = plt.subplots(figsize=(6, 4))
        graph.add_edges_from(edges)
        shortest_path = nx.average_shortest_path_length(graph)
        avg_clustering = nx.average_clustering(graph)
                
        if save == "True":
            nx.draw(graph, with_labels=True)
            plt.savefig("global_coupled.png")
        st.image('global_coupled.png')
        
        k, L, C = calcu(mat)
        st.write('全局耦合网络图的特征理论值')  
        st.latex(
            r'''
            K_{GC} = |A| - 1 (节点的度)\\
            
            L_{GC} = 1 (平均路径长度)\\
            
            C_{GC} = 1 (聚类系数)\\
            '''
        )
    if option=="1.2: nearest_neighbor_coupling_network":
        
        max_K = nodes_num if nodes_num%2==0 else nodes_num-1
        K_gcn = st.sidebar.slider("最近邻网络中K值", 2, max_K, 2, 2)
        df = draw_mat(nodes_num, option, K = K_gcn)
        st.write(df)
        mat = np.array(df)
        
               # Set the labels for each node
        labels = {}
        nodes = list(range(nodes_num))
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for n in graph.nodes():
            labels[n] = str(n)
        edges = []
        # Generate a list of nodes
        k = k
        if 2*k+1 > nodes_num:
            raise "Cannot generate the graph! "
        else:
            for i in range(len(nodes)):
                for j in range(k):
                    if i < (i+j+1) % nodes_num:
                        if [i, (i+j+1) % nodes_num] not in edges:
                            edges.append([i, (i+j+1) % nodes_num])
                    elif i > (i+j+1) % nodes_num:
                        if [(i+j+1) % nodes_num, i] not in edges:
                            edges.append([(i+j+1) % nodes_num, i])
        fig, ax = plt.subplots(figsize=(6, 4))
        graph.add_edges_from(edges)
        shortest_path = nx.average_shortest_path_length(graph)
        avg_clustering = nx.average_clustering(graph)
                
        if save == "True":
            nx.draw(graph, with_labels=True)
            plt.savefig("global_coupled.png")
        st.image('global_coupled.png')
        
        
        # k = K_gcn
        k, L, C = calcu(mat)
        st.write('最近邻耦合网络')
        st.latex(
            r'''
            K_{NC} = K \\
            
            L_{NC} = |A|/2K \\
            
            C_{NC} = 3/4 \\
            ''')

    if option=="1.3: star_shape_network":
        df = draw_mat(nodes_num, option)
        st.write(df)
        mat = np.array(df)
               # Set the labels for each node
        labels = {}
        nodes = list(range(nodes_num))
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for n in graph.nodes():
            labels[n] = str(n)
        edges = []
        # Generate a list of nodes
        # Check validity
        center = center
        if center >= nodes_num:
            raise "Cannot generate the graph! "
        else:
            # Connect all the nodes to the center
            for i in range(len(nodes)):
                if i != center:
                    edges.append([i, center])
        fig, ax = plt.subplots(figsize=(6, 4))
        graph.add_edges_from(edges)
        shortest_path = nx.average_shortest_path_length(graph)
        avg_clustering = nx.average_clustering(graph)
                
        if save == "True":
            nx.draw(graph, with_labels=True)
            plt.savefig("global_coupled.png")
        st.image('global_coupled.png')

        k, L, C = calcu(mat)
        
        # 因为会出现nan, 而我们规定如果只有一个邻居的时候就默认为1
        C = np.ones(shape=(nodes_num,))
        st.write('星形网络')
        st.latex(
            r'''
            k_{star} = 2 \\
            
            L_{star} = 2 \\
            
            C_{star} = 1 \\
            ''')
        
    st.subheader('该网络真实特征测量值')
    st.write(f'节点度：{k}')
    st.write(f'平均路径长度：{shortest_path}')
    st.write(f'聚类系数：{avg_clustering}')

# if option is not about the network
elif option=='6.1 Boids_model':
    st.title(option)
    st.write("The following is the presentation of the Boids model")
    
    video_file = open('boid.mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    
elif option=='2.1 Watts_and_Strogtz_network':
    def f(n):
        return 1 if n<1 else np.log(n)/n
    st.title(option)
    # define network parameters
    N = st.sidebar.slider("网络中点的数量", 8, 25, 10, 1)
    k_over_2 = st.sidebar.slider("K二倍值", 2, N//2, 2, 2)
    betas = [0, 0.15, 0.4]
    st.write(f"Here, we assign the porbability of reconnect = {betas}")
    labels = [ r'$p=0$', r'$p=0.15$', r'$p=0.4$']

    focal_node = 0
    fig, ax = pl.subplots(1,3,figsize=(9,3))

    # scan beta values
    for ib, beta in enumerate(betas):

        # generate small-world graphs and draw
        G = generate_small_world_graph(N, k_over_2, beta)
        draw_network(G,k_over_2,focal_node=focal_node,ax=ax[ib])

        ax[ib].set_title(labels[ib],fontsize=11)
    st.pyplot(fig=fig)
    st.write('WS小世界网络')
    st.latex(
        r'''
        k_{random} = K \\
        
        L_{WS} = 2N/K*f(NKp/2) \\
        
        C_{WS} = 3(K-2)/(4(K-1)) * (1=p)^3 \\
        ''')
    K = k_over_2//2
    L = 2*N/K * f(N * K * betas[2]/2)
# C = 3*(K-2)/(4*(K-1))*(1-betas[2])**3
    shortest_path = nx.average_shortest_path_length(G)
    avg_clustering = nx.average_clustering(G)
    st.subheader('该网络特征值')
    st.write(f'节点度：{K}')
    st.write(f'平均路径长度：{shortest_path}')
    st.write(f'聚类系数：{avg_clustering}')

    
elif option=='2.2: Norman_and_Watts_network': 
    def f(n) :
        return 1/(2*sqrt(n*n+2*n))*np.arctanh(n/(n+2))
    st.title(option)
    # define network parameters
    N = st.sidebar.slider("网络中点的数量", 8, 25, 10, 1)
    k_over_2 = st.sidebar.slider("K二倍值", 2, N//2, 2, 2)
    betas = [0, 0.025, 0.05]
    st.write(f"Here, we assign the porbability of reconnect = {betas}")
    labels = [ r'$p=0$', r'$p=0.025$', r'$p=0.075$']

    focal_node = 0
    fig, ax = pl.subplots(1,3,figsize=(9,3))

    # scan beta values
    for ib, beta in enumerate(betas):

        # generate small-world graphs and draw
        G = generate_small_world_graph(N, k_over_2, p=beta, use_slow_algorithm=True)
        
        draw_network(G,k_over_2,focal_node=focal_node,ax=ax[ib])

        ax[ib].set_title(labels[ib],fontsize=11)
    st.pyplot(fig=fig)
    # show
    st.write('NM小世界网络')
    st.latex(
        r'''
        k_{random} = K+p(|A|-1) \\
        
        L_{NM} =  2N/K*f(NKp/2)\\
        
        C_{NM} = 3(K-2)/ (4(K-1)+ 4Kp(p+2)) \\
        ''')
    K = int(k_over_2)//2 + betas[2]*(N-1)
    L = 2*N/K * f(N * K * betas[2]/2)
    C = 3*(K-2)/((4*(K-1))+4*K*betas[2]*(betas[2]+2))
    shortest_path = nx.average_shortest_path_length(G)
    avg_clustering = nx.average_clustering(G)
    st.subheader('该网络特征值')
    st.write(f'节点度：{K}')
    st.write(f'平均路径长度：{shortest_path}')
    st.write(f'聚类系数：{avg_clustering}')

    
    
    
    