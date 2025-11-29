
def compute_discounted_lightcone_scores_bitset(dag, H=8, gamma=0.8, alpha=1.0, beta=0.5):
    """
    Compute discounted reachability + cone-centrality scores for each DAG op-node.

    - H: horizon (max path length to consider)
    - gamma: discount factor (0 < gamma <= 1). Smaller => near-term emphasis.
    - alpha: weight for discounted forward reachability
    - beta:  weight for discounted cone centrality (past * future)

    Returns: dict { node._node_id : score }
    """
    # 1) Collect op nodes in topological order
    nodes = list(dag.topological_op_nodes())  # list of DAGNode
    N = len(nodes)
    if N == 0:
        return {}

    node_index = {node: idx for idx, node in enumerate(nodes)}

    # 2) Build forward adjacency (indices)
    forward_adj = [[] for _ in range(N)]
    backward_adj = [[] for _ in range(N)]
    last_on_qubit = dict()
    for idx, node in enumerate(nodes):
        for q in node.qargs:
            if q in last_on_qubit:
                prev_idx = last_on_qubit[q]
                forward_adj[prev_idx].append(idx)
                backward_adj[idx].append(prev_idx)
            last_on_qubit[q] = idx

    # 3) We will compute reach_exact[d][i] = bitset of nodes reachable from i in exactly d steps
    #    We'll store only up to H. Use Python ints as bitsets.
    reach_exact = [ [0]*N for _ in range(H+1) ]  # index 0 unused (distance 0 not needed)

    # distance 1: immediate children
    for i in range(N):
        bits = 0
        for child in forward_adj[i]:
            bits |= (1 << child)
        reach_exact[1][i] = bits

    # distances 2..H: use DP: nodes at exact d = union over children of their exact (d-1),
    # then remove nodes counted at smaller distances to keep "exact".
    for d in range(2, H+1):
        for i in range(N):
            bits = 0
            for child in forward_adj[i]:
                bits |= reach_exact[d-1][child]
                bits |= (1 << child)  # also include child at distance 1 -> contributes via child
            # Now remove nodes reachable in shorter distances to leave only exact-d nodes:
            # union_shorter = OR_{k=1..d-1} reach_exact[k][i]
            union_shorter = 0
            for k in range(1, d):
                union_shorter |= reach_exact[k][i]
            exact_d = bits & (~union_shorter)
            reach_exact[d][i] = exact_d

    # 4) Compute discounted forward reachability score: sum_{d=1..H} gamma^d * popcount(reach_exact[d][i])
    def popcount(x):
        return x.bit_count()
    disc_forward = [0.0]*N
    for i in range(N):
        s = 0.0
        for d in range(1, H+1):
            cnt = popcount(reach_exact[d][i])
            s += (gamma**d) * cnt
        disc_forward[i] = s

    # 5) Compute discounted backward (past) symmetrically if we want discounted cone centrality.
    #    We'll compute reach_exact_back similarly using backward_adj.
    reach_exact_back = [ [0]*N for _ in range(H+1) ]
    for i in range(N):
        bits = 0
        for parent in backward_adj[i]:
            bits |= (1 << parent)
        reach_exact_back[1][i] = bits

    for d in range(2, H+1):
        for i in range(N):
            bits = 0
            for parent in backward_adj[i]:
                bits |= reach_exact_back[d-1][parent]
                bits |= (1 << parent)
            union_shorter = 0
            for k in range(1, d):
                union_shorter |= reach_exact_back[k][i]
            exact_d = bits & (~union_shorter)
            reach_exact_back[d][i] = exact_d

    disc_backward = [0.0]*N
    for i in range(N):
        s = 0.0
        for d in range(1, H+1):
            cnt = popcount(reach_exact_back[d][i])
            s += (gamma**d) * cnt
        disc_backward[i] = s

    # 6) Compute cone centrality using discounted past * discounted future
    disc_cone = [ disc_backward[i] * disc_forward[i] for i in range(N) ]

    # 7) Build score table over DAG node ids
    score_table = {}
    for i, node in enumerate(nodes):
        score = alpha * disc_forward[i] + beta * disc_cone[i]
        score_table[node._node_id] = float(score)

    return score_table
