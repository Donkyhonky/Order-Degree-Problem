import networkx as nx
import argparse

# Argument parser
argumentparser = argparse.ArgumentParser()
argumentparser.add_argument('nnodes', type=int)
argumentparser.add_argument('degree', type=int)

def main(args):
    nnodes = args.nnodes
    degree = args.degree
    assert degree < nnodes, "Degree must be less than the number of nodes."
    assert (nnodes * degree) % 2 == 0, "The product of nodes and degree must be even for a regular graph. Handshaking lemma"

    # Try up to 100 times to generate a connected graph
    max_attempts = 100
    connected_graph = None

    for attempt in range(max_attempts):
        g = nx.random_regular_graph(degree, nnodes, seed=attempt)  # Use different seed for variability
        if nx.is_connected(g):
            connected_graph = g
            break

    if connected_graph:
        print(f"Connected graph found on attempt {attempt + 1}", file=sys.stderr)
    else:
        print(f"Unable to find a connected graph after {max_attempts} attempts.", file=sys.stderr)
        return  # Exit if no connected graph is found

    # Save the graph files
    '''
    basename = f"n{nnodes}d{degree}.random"
    save_edges(connected_graph, basename + ".edges")
    save_image(connected_graph, basename + ".png")
    save_adjacency_list(connected_graph, basename + ".adjacency_list")
    '''

    # Output adjacency list to stdout for Rust
    output_adjacency_list(connected_graph)

def save_edges(g, filepath):
    nx.write_edgelist(g, filepath, data=False)

def save_adjacency_list(graph, filename):
    with open(filename, 'w') as f:
        for node in sorted(graph.nodes()):
            neighbors = sorted(graph.neighbors(node))
            line = ' '.join(map(str, neighbors))
            f.write(f"{line}\n")

def output_adjacency_list(graph):
    # Output the adjacency list to stdout (Rust will capture this)
    for node in sorted(graph.nodes()):
        neighbors = sorted(graph.neighbors(node))
        print(' '.join(map(str, neighbors)))

def save_image(g, filepath):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    layout = nx.circular_layout(g)
    nx.draw(g, with_labels=False, node_size=50, linewidths=0, alpha=0.5, node_color='#3399ff', edge_color='#666666', pos=layout)
    plt.savefig(filepath)

if __name__ == '__main__':
    import sys
    main(argumentparser.parse_args())
