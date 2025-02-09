/*
General Description
*/

use super::base::{forward_circular_distance};
use std::collections::{VecDeque};

pub fn is_graph_connected(adj_matrix: &Vec<Vec<usize>>) -> bool {
    let total_nodes = adj_matrix.len();
    if total_nodes == 0 {
        return true;
    }

    let mut visited = vec![false; total_nodes];
    let mut queue = VecDeque::new();

    // Start BFS from node 0
    queue.push_back(0);
    visited[0] = true;

    while let Some(node) = queue.pop_front() {
        for &neighbor in &adj_matrix[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }

    // Check if all nodes were visited
    visited.iter().all(|&v| v)
}

pub fn check_if_graph_g_symmetrical(adj_matrix: &Vec<Vec<usize>>, g: usize) -> bool {
    let total_nodes = adj_matrix.len();
    let n_div_g = total_nodes / g;

    // Step 1: Create the forward circular distance matrix
    let mut distance_matrix: Vec<Vec<usize>> = Vec::new();

    for (row_index, neighbors) in adj_matrix.iter().enumerate() {
        let mut distances: Vec<usize> = neighbors
            .iter()
            .map(|&neighbor| forward_circular_distance(row_index, neighbor, total_nodes))
            .collect();
        distances.sort_unstable();  // Sort distances for easy comparison
        distance_matrix.push(distances);
    }

    /*
    println!("\nDistance Matrix:");
    for (index, row) in distance_matrix.iter().enumerate() {
        println!("Node {}: {:?}", index, row);
    }
    */

    // Step 2: Check for symmetry
    for row in 0..n_div_g {
        let base_distances = &distance_matrix[row];

        for group in 1..g {
            let compare_row = row + group * n_div_g;

            if compare_row >= total_nodes {
                break;  // Avoid index out of bounds
            }

            if &distance_matrix[compare_row] != base_distances {
                println!("Mismatch found at row {} and row {}", row, compare_row);
                return false;  // Not symmetrical if any mismatch is found
            }
        }
    }
    true  // All groups are symmetrical
}