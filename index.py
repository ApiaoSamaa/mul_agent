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


OptionSequence = ["1.1: global_coupled_network",'1.2: nearest_neighbor_coupling_network', '1.3: star_shape_network', '2.1 Watts_and_Strogtz_network','6.1 Boids_model']
option = st.sidebar.selectbox("选择问题类型",options=OptionSequence,index=0)
if (option!='1.5'):
    nodes_num = st.sidebar.slider("网络中点的数量", 2, 20, 7, 1)

    st.title(option)
    df = draw_mat(nodes_num, option)
    st.subheader('交互结构的邻接矩阵表示')
    st.write(df)

    mat = np.array(df)
    k, L, C = calcu(mat)

    if option=="1.1: global_coupled_network":
        st.write('全局耦合网络图的特征理论值')  
        st.latex(
            r'''
            K_{GC} = |A| - 1 (节点的度)\\
            
            L_{GC} = 1 (平均路径长度)\\
            
            C_{GC} = 1 (聚类系数)\\
            '''
        )
    if option=="1.2: global_coupled_network":
        st.write('')
        
    if option=="1.3: star_shape_network":
        st.write('')
        
    st.subheader('该网络真实特征测量值')
    st.write(f'节点度：{k}')
    st.write(f'平均路径长度：{C}')
    st.write(f'聚类系数：{C}')

# if option is not about the network
elif option=='6.1 Boids_model':
    0