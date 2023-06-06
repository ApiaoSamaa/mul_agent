# %%
import numpy as np
import pandas as pd
from collections import deque
import networkx as nx
import random


# def draw_mat(nodes_num=10, type= '1.2: nearest_neighbor_coupling_network', K = 2):
#     """_summary_
#     画出选中的节点类型的邻接矩阵
#     Args:
#         nodes_num (_type_): _description_
#         type (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     if type == '1.1: global_coupled_network':
#         df = pd.DataFrame(data=np.ones(shape=(nodes_num,nodes_num)),columns=(i+1 for i in range(nodes_num)))
#         # change dataframe raw name, using data.index, directly change it.
#         df.index = df.columns
#         return formalize(df)
    
#     elif type == '1.2: nearest_neighbor_coupling_network':
#         df = pd.DataFrame(data=np.zeros(shape=(nodes_num,nodes_num)),columns=(i+1 for i in range(nodes_num)))
        
#         offset = K//2
#         df_np = np.array(df)
#         elements1, indices1 = get_diagonal_elements(df_np, offset)
#         elements11, indices11 = get_diagonal_elements(df_np, offset - nodes_num)
#         elements2, indices2 = get_diagonal_elements(df_np, - offset)
#         elements22, indices22 = get_diagonal_elements(df_np, - offset + nodes_num)
#         for i in (indices1+ indices11 + indices2 + indices22):
#             df_np[i] = 1
#         return formalize(pd.DataFrame(df_np))
                
#     elif type == '1.3: star_shape_network':
#         df = pd.DataFrame(data=np.zeros(shape=(nodes_num,nodes_num)),columns=(i+1 for i in range(nodes_num)))
        
#         df.iloc[nodes_num//2] = 1
#         df.iloc[:, nodes_num//2] = 1
#         return formalize(df)
        
        
# def get_diagonal_elements(arr, offset):
#     n, m = arr.shape
#     if offset >= 0:
#         i_start = 0
#         j_start = offset
#         i_step = 1
#         j_step = 1
#         length = min(n - offset, m)
#     else:
#         i_start = -offset
#         j_start = 0
#         i_step = 1
#         j_step = 1
#         length = min(n, m + offset)
#     indices = []
#     elements = []
#     for k in range(length):
#         i = i_start + k * i_step
#         j = j_start + k * j_step
#         indices.append((i, j))
#         elements.append(arr[i, j])
#     return elements, indices

# # arr = np.array([[0, 1, 2, 3,4],
# #                 [4, 5, 6, 7, 4],
# #                 [8, 9, 10, 11, 4],
# #                 [12, 13, 14, 15, 4],
# #                 [1,1,1,1,1]])d



def draw_mat(nodes_num, type, K = None):
    """_summary_
    画出选中的节点类型的邻接矩阵
    Args:
        nodes_num (_type_): _description_
        type (_type_): _description_

    Returns:
        _type_: _description_
    """
    if type == '1.1: global_coupled_network':
        df = pd.DataFrame(data=np.ones(shape=(nodes_num,nodes_num)),columns=(i+1 for i in range(nodes_num)))
        # change dataframe raw name, using data.index, directly change it.
        df.index = df.columns
        return formalize(df)
    
    elif type == '1.2: nearest_neighbor_coupling_network':
        df = pd.DataFrame(data=np.zeros(shape=(nodes_num,nodes_num)),columns=(i+1 for i in range(nodes_num)))
        
        offset = K//2
        df_np = np.array(df)
        elements1, indices1 = get_diagonal_elements(df_np, offset)
        elements11, indices11 = get_diagonal_elements(df_np, offset - nodes_num)
        elements2, indices2 = get_diagonal_elements(df_np, - offset)
        elements22, indices22 = get_diagonal_elements(df_np, - offset + nodes_num)
        for i in (indices1+ indices11 + indices2 + indices22):
            df_np[i] = 1
        return formalize(pd.DataFrame(df_np))
                
    elif type == '1.3: star_shape_network':
        df = pd.DataFrame(data=np.zeros(shape=(nodes_num,nodes_num)),columns=(i+1 for i in range(nodes_num)))
        
        df.iloc[nodes_num//2] = 1
        df.iloc[:, nodes_num//2] = 1
        return formalize(df)
        
        
def get_diagonal_elements(arr, offset):
    n, m = arr.shape
    if offset >= 0:
        i_start = 0
        j_start = offset
        i_step = 1
        j_step = 1
        length = min(n - offset, m)
    else:
        i_start = -offset
        j_start = 0
        i_step = 1
        j_step = 1
        length = min(n, m + offset)
    indices = []
    elements = []
    for k in range(length):
        i = i_start + k * i_step
        j = j_start + k * j_step
        indices.append((i, j))
        elements.append(arr[i, j])
    return elements, indices

# arr = np.array([[0, 1, 2, 3,4],
#                 [4, 5, 6, 7, 4],
#                 [8, 9, 10, 11, 4],
#                 [12, 13, 14, 15, 4],
#                 [1,1,1,1,1]])

def calcu(mat):
    """_summary_
    在有邻接矩阵的前提下，输出该网络的k L C
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
    """_summary_
    是计算 C 时候的 util函数，寻找每个idx节点的邻居数量信息
    Args:
        mat (_type_): _description_
        node_idx (_type_): _description_

    Returns:
        _type_: _description_
    """
    # tuple 
    neighbor = np.where(mat[node_idx] == 1)
    # Equal: num = np.isin(np.where(mat[neighbor]==1)[0], neighbor).sum() / 2
    num = mat[neighbor][:,neighbor].sum() / 2
    return num.astype('int32')
    

def find_length(mat):
    num_nodes = mat.shape[0]
    output = np.zeros(shape=(num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            backups = np.array([100 for i in range(num_nodes)])
            if mat[i, j] == 1:
                output[i, j] = 1
            else:
                backups = np.where(mat[i] == 1)
                
    
    
    
    
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
    1. 为了输出邻接矩阵美观性，将float显示值变成int
    2. 把邻接矩阵的对角线规范化为0
    Args:
        d (dataframe): 
    Returns: 规范化的矩阵，其中对角线为0，整数形式。
    """
    df_np = np.array(d)
    idx_diag = np.diag_indices(df_np.shape[0])
    df_np[idx_diag] = 0
    # df = pd.DataFrame(df_np.astype('int32'))
    return pd.DataFrame(df_np.astype('int32'))


def mat2G(mat):
    """_summary_
    make the mat into G in order to draw it
    Args:
        mat (_type_): _description_
    """
    return 0

def G2mat(G):
    """_summary_
    make the networkx Graph into mat, so that we can calculate the features.
    Args:
        G (_type_): _description_
    """
    return 0

def get_connection_probabilities(N, k_over_2, p):
    """_summary_
    返回连接可能性
    Return:
    the connection probabilities :math:`p_S` and :math:`p_L`.
    """
    assert_parameters(N, k_over_2, p)
    k = float(int(k_over_2*2))
    # pS = k / (k + p * (N-1.0-k))
    pS = 1
    # pL = k * p / (k + p*(N-1.0-k))
    pL = p
    return pS, pL

def get_edgelist_slow(N, k_over_2, p):
    """_summary_
    遍历所有的节点，计算距离并且根据短距离链接和长距离链接可能性*分别*加上一个边
    Args:
        N (_type_): _description_
        k_over_2 (_type_): _description_
        p (_type_): _description_
    """
    assert_parameters(N, k_over_2, p)
    # pS, pL = get_connection_probabilities(N, k_over_2, p)
    
    N = int(N)
    k_over_2 = int(k_over_2)
    
    E = []
    
    for i in range(N-1):
        for j in range(i+1,N):
            distance = j - i
            # if  (distance <= k_over_2) or ((N - distance) <= k_over_2):
            #     pC = pS
            # else:
            #     pC = pL
            if random.rand() < p:
                E.append((i,j))
                
    return E
    
    
def assert_parameters(N, k_over_2, p):
    """_summary_
    保证 N 和 k_over_2 是int，0<=p<=1
    Args:
        N (int): _description_
        k_over_2 (int): _description_
        p (float): _description_
    """
    assert(k_over_2 == int(k_over_2))
    assert(N == int(N))
    assert(p >= 0.0)
    assert(p <= 1.0)
    
def get_fast_smallworld_graph(N, k_over_2, p):
    """_summary_
    遍历所有的可能的短边并且以p_S 概率reconnet边
    
    Args:
        N (_type_): _description_
        k_over_2 (_type_): _description_
        p (_type_): _description_
    """
    assert_parameters(N,k_over_2,p)
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    N = int(N)
    k_over_2 = int(k_over_2)
    k = int(2*k_over_2)
    
    # add short range edges in order N* k/2
    for u in range(N):
        for v in range(u+1, u+k_over_2+1):
            # if random.rand() < 1:
            G.add_edge(u, v % N)

    mL_max = N*(N-1-k) // 2
    mL = np.random.binomial(mL_max, p)
    number_of_rejects = 0
    
    for m in range(mL):
        while True:
            u = random.randint(0,N)
            v = u + k_over_2 + random.randint(1, N - k)
            v %= N
            
            if not G.has_edge(u, v):
                G.add_edge(u,v)
                break
    return G
            
def get_22_smallworld_graph(N, k_over_2, p):
    """_summary_
    遍历所有的可能的短边并且以p_S 概率加上边
    
    Args:
        N (_type_): _description_
        k_over_2 (_type_): _description_
        p (_type_): _description_
    """
    assert_parameters(N,k_over_2,p)
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    N = int(N)
    k_over_2 = int(k_over_2)
    k = int(2*k_over_2)
    
    # add short range edges in order N* k/2
    for u in range(N):
        for v in range(u+1, u+k_over_2+1):
            if random.rand() < p:
                G.add_edge(u, v % N)

    mL_max = N*(N-1-k) // 2
    mL = np.random.binomial(mL_max, p)
    
    for m in range(mL):
        while True:
            u = random.randint(0,N)
            v = u + k_over_2 + random.randint(1, N - k)
            v %= N
            
            if not G.has_edge(u, v):
                G.add_edge(u,v)
                break
    return G

def generate_small_world_graph(N, k_over_2, p, use_slow_algorithm=False, verbose=False):
    if use_slow_algorithm:
            # add short range edges in order N* k/2
        G = get_fast_smallworld_graph(N, k_over_2, p)
        for u in range(N):
            # for v in range(u+1, u+2):
            G.add_edge(u, (u+1) % N)

    else:
        G = get_fast_smallworld_graph(N, k_over_2, p)
    
    return G






# %%
