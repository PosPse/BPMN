def get_edge_y_category(node_i_type:int, node_j_type:int) -> int:
    '''
        none: 0
        activity -> activity: 1 [0,0]
        activity -> condition: 2 [0,1]
        activity -> sign-successor: 3 [0,2]
        activity -> sign-selection: 4 [0,3]
        activity -> sign-parallel: 5 [0,4]
        activity -> sign-loop: 6 [0,5]
        condition -> sign-successor: 7 [1,2]
        condition -> sign-selection: 8 [1,3]
        condition -> sign-parallel: 9 [1,4]
        condition -> sign-loop: 10 [1,5]
    '''
    edge_map = [[0,0], [0,1], [0,2],[0,3],[0,4],[0,5],[1,1],[1,2],[1,3],[1,4],[1,5]]
    edge = [node_i_type, node_j_type]
    edge.sort()
    if edge not in edge_map:
        return 0
    return edge_map.index(edge) + 1

print(get_edge_y_category(2, 1))