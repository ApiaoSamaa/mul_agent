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


import streamlit as st
import numpy as np
import pandas as pd
from collections import deque

def draw_mat(nodes_num, type):
    
    if type == '1.1: global_coupled_network':
        df = pd.DataFrame(data=np.ones(shape=(nodes_num,nodes_num)),columns=(i+1 for i in range(nodes_num)))
        # change dataframe raw name, using data.index, directly change it.
        df.index = df.columns
        df = formalize(df)
        
        return df
    else:
        # 其他情况不能直接展示全1的网络了
        
        return
    
    
def calcu(mat):
    """_summary_

    Args:
        mat (pandas.dataframe(nodes_num, nodes_num)): 表示了相应交互结构的矩阵表示
    Returns:
        度(k) (lists(nodes_num)): 每一个节点的度
        路径长度(L) (float): 任意两节点间最短路径长度d的平均值（最短路径长度可由广度优先搜索算法来确定）
        聚类系数(C) (lists(nodes_num)): 表示了每个节点的聚类系数值
    """
    
    nodes_num = mat.shape[0]
    k = [mat[i].sum() for i in range(nodes_num)]
    L = 2 * 1/2 * shortest_path(mat).sum()/(nodes_num * (nodes_num - 1))
    C = [2 * findNeighbor(mat,i)/(k[i]* (k[i] -1 )) for i in range(nodes_num)]
    
    return k, L, C

def findNeighbor(mat, node_idx):
    # tuple 
    neighbor = np.where(mat[node_idx] == 1)
    # Equal: num = np.isin(np.where(mat[neighbor]==1)[0], neighbor).sum() / 2
    num = mat[neighbor][:,neighbor].sum() / 2
    return num.astype('int32')
    

def shortest_path(matrix):
    """_summary_
    使用广度优先算法搜索每一个点到另外一个的最短距离
    Args:
        mat (pandas.dataframe(nodes_num, nodes_num)): 表示了相应交互结构的矩阵表示
    Returns:
        leastLengthMat (np.array(nodes_num, nodes_num)): 一个满矩阵，每个值代表从i点到j点的最短路径长度
    """
    node_number = len(matrix)
    output_matrix = [[0 for _ in range(node_number)] for _ in range(node_number)]
    for i in range(node_number):
        visited = [False] * node_number
        queue = deque([(i, 0)])
        while queue:
            node, dist = queue.popleft()
            if visited[node]:
                continue
            visited[node] = True
            output_matrix[i][node] = dist
            for j in range(node_number):
                if matrix[node][j] == 1:
                    queue.append((j, dist + 1))
    return np.array(output_matrix)
    
    
def formalize(d):
    """_summary_

    Args:
        d (dataframe): 
    Returns: 规范化的矩阵，其中对角线为0，整数形式。
    """
    df_np = np.array(d)
    idx_diag = np.diag_indices(df_np.shape[0])
    df_np[idx_diag] = 0
    # df = pd.DataFrame(df_np.astype('int32'))
    return pd.DataFrame(df_np.astype('int32'))

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
    






















# iterations = st.sidebar.slider("Level of detail", 2, 20, 10, 1)
# separation = st.sidebar.slider("Separation", 0.7, 2.0, 0.7885)

# # Non-interactive elements return a placeholder to their location
# # in the app. Here we're storing progress_bar to update it later.
# progress_bar = st.sidebar.progress(0)

# # These two elements will be filled in later, so we create a placeholder
# # for them using st.empty()
# frame_text = st.sidebar.empty()
# image = st.empty()

# m, n, s = 960, 640, 400
# x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
# y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))

# for frame_num, a in enumerate(np.linspace(0.0, 4 * np.pi, 100)):
#     # Here were setting value for these two elements.
#     progress_bar.progress(frame_num)
#     frame_text.text("Frame %i/100" % (frame_num + 1))

#     # Performing some fractal wizardry.
#     c = separation * np.exp(1j * a)
#     Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
#     C = np.full((n, m), c)
#     M: Any = np.full((n, m), True, dtype=bool)
#     N = np.zeros((n, m))

#     for i in range(iterations):
#         Z[M] = Z[M] * Z[M] + C[M]
#         M[np.abs(Z) > 2] = False
#         N[M] = i

#     # Update the image placeholder by calling the image() function on it.
#     image.image(1.0 - (N / N.max()), use_column_width=True)

# # We clear elements by calling empty on them.
# progress_bar.empty()
# frame_text.empty()

# # Streamlit widgets automatically run the script from top to bottom. Since
# # this button is not connected to any other logic, it just causes a plain
# # rerun.
# st.button("Re-run")





