use rand::Rng;  
use std::collections::{HashSet};


pub fn forward_circular_distance(a: usize, b: usize, total_nodes: usize) -> usize {
    (b + total_nodes - a) % total_nodes
}

pub fn find_symmetrical_related_edges(initial_edge: (usize, usize), total_nodes: usize, g: usize) -> Vec<(usize, usize)> {

    /* Description */

    let n_div_g = total_nodes / g;
    let (start_vertex, end_vertex) = initial_edge;
    let group_offset = start_vertex % n_div_g;

    // Calculate forward circular distance
    let forward_circular_distance = forward_circular_distance(start_vertex, end_vertex, total_nodes);

    // Create the selected_edges list
    let mut selected_edges: Vec<(usize, usize)> = Vec::new();
    for m in 0..g {
        let first_node = m * n_div_g + group_offset;
        let second_node = (first_node + forward_circular_distance) % total_nodes;
        selected_edges.push((first_node, second_node));
    }

    selected_edges
}

pub fn select_random_edge(adj_matrix: &Vec<Vec<usize>>) -> (usize, usize) {
    /*
    Select a ranodm edge from a given adjacency_matrix
    */

    let mut rng = rand::thread_rng();
    let total_nodes = adj_matrix.len();

    let row_index = rng.gen_range(0..total_nodes); // Randomly select a row (node)

    // Ensure the row has at least one neighbor
    if adj_matrix[row_index].is_empty() {
        eprintln!("Selected node {} has no neighbors. Retrying...", row_index);
        return select_random_edge(adj_matrix);  // Retry if the row has no neighbors
    }

    // Randomly select a neighbor from the selected row
    let neighbor_index = rng.gen_range(0..adj_matrix[row_index].len());
    let neighbor = adj_matrix[row_index][neighbor_index];

    (row_index, neighbor)
}

pub fn modify_adjacency_matrix(adj_matrix: &mut Vec<Vec<usize>>, selected_edges: &Vec<(usize, usize)>, new_edges: &Vec<(usize, usize)>) {

    /* Function to remove selected_edges and add new_edges in the adjacency matrix */


    // Use sets for efficient lookup and modification
    let mut adj_sets: Vec<HashSet<usize>> = adj_matrix.iter().map(|neighbors| neighbors.iter().cloned().collect()).collect();

    // Remove selected edges
    for &(u, v) in selected_edges {
        adj_sets[u].remove(&v);
        adj_sets[v].remove(&u);
    }

    // Add new edges
    for &(u, v) in new_edges {
        adj_sets[u].insert(v);
        adj_sets[v].insert(u);
    }

    // Convert sets back to vectors
    for (i, neighbors) in adj_sets.iter().enumerate() {
        adj_matrix[i] = neighbors.iter().cloned().collect();
    }
}