from collections import defaultdict

def gen_q_dict_mis_sym(nx_G, penalty=2):
    Q_dic = defaultdict(int)
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = penalty
        Q_dic[(v, u)] = penalty
    for u in nx_G.nodes:
        Q_dic[(u, u)] = -1
    return Q_dic
