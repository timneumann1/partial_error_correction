
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


def compute_future_reachability(dag):
    """
    Compute per-gate 'future qubit influence' by propagating reachable qubits
    backward through layer structure.

    For each gate:
        score = number of qubits that can be affected in the future
                starting from this gate.
    
    """

    layers = list(dag.layers())
    num_layers = len(layers)

    # influence[layer][node_id] = set of qubits reachable from this gate forward
    influence = [{} for _ in range(num_layers)]

    # Initialize score table
    score_table = {}

    # --- Initialize final layer ---
    last = num_layers - 1
    for node in layers[last]["graph"].op_nodes():
        qs = set(node.qargs)
        influence[last][node] = set(qs)      # gate only influences its own qubits
        score_table[node._node_id] = len(influence[last][node])

    # --- Backward propagation ---
    for i in reversed(range(num_layers - 1)):
        next_layer = layers[i+1]["graph"].op_nodes()
        current_nodes = layers[i]["graph"].op_nodes()

        # Precompute qubit → union of influence of all next-layer gates that touch it
        qubit_to_future = {}

        for node_next in next_layer:
            infl = influence[i+1][node_next]    # set of future qubits for next-layer gate
            for q in node_next.qargs:
                if q not in qubit_to_future:
                    qubit_to_future[q] = set()
                qubit_to_future[q] |= infl

        # Now compute influence for nodes in current layer
        for node in current_nodes:
            qs = node.qargs
            union_set = set(qs)

            # For each qubit the gate touches, add the future influence coming from next layer
            for q in qs:
                if q in qubit_to_future:
                    union_set |= qubit_to_future[q]

            influence[i][node] = union_set
            score_table[node._node_id] = len(influence[i][node])


    return score_table
