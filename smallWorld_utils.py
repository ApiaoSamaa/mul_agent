
def get_edgelist_slow(N, k_over_2, p):
    """_summary_
    遍历所有的节点，计算距离并且根据短距离链接和长距离链接可能性分别加上一个边
    Args:
        N (_type_): _description_
        k_over_2 (_type_): _description_
        p (_type_): _description_
    """
    
def assert_parameters(N, k)

def generate_small_world_graph(N, k_over_2, p, use_slow_algorithm=Flase, verbose=False):
    if use_slow_algorithm:
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        G.add_edges_from(get_edgelist_slow(N,k_over_2,p))
    