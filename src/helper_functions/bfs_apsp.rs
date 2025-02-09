use std::collections::VecDeque;

pub fn serial_bfs_apsp(adj_matrix: &Vec<Vec<usize>>, n_div_g: usize) -> (usize, f64) {
    /*
    Optimized BFS for g-symmetrical graphs:
    - Runs BFS only from the first n_div_g nodes (representative nodes of each group).
    - Computes the diameter and ASPL by leveraging symmetrical properties.
    */

    let n = adj_matrix.len();  // Total number of nodes in the graph
    let mut diameters = vec![0; n_div_g];
    let mut avg_distances = vec![0.0; n_div_g];

    // Run BFS from only the first n_div_g nodes
    for source in 0..n_div_g {
        let distances = bfs(adj_matrix, source);
        diameters[source] = *distances.iter().max().unwrap();  // Diameter from this source
        
        // Sum of distances to all other nodes
        let sum_distances: usize = distances.iter().sum();
        avg_distances[source] = sum_distances as f64 / (n as f64 - 1.0);  // Average distance from this source
    }

    // The diameter of the graph is the maximum diameter from the selected sources
    let max_diameter = *diameters.iter().max().unwrap();
    
    // Average distance over all nodes is calculated from the representative nodes
    let average_distance = avg_distances.iter().sum::<f64>() / n_div_g as f64;

    (max_diameter, average_distance)
}





fn bfs(adj_matrix: &Vec<Vec<usize>>, source: usize) -> Vec<usize> {
    /*
    Standard BFS starting from source node.
    Returns the list of distances from source to all other nodes.
    */
    let n = adj_matrix.len();
    let mut distances = vec![usize::MAX; n];
    let mut frontier = VecDeque::new();

    distances[source] = 0;
    frontier.push_back(source);

    let mut k = 1;

    while !frontier.is_empty() {
        topdown(adj_matrix, &mut frontier, &mut distances, k);
        k += 1;
    }

    distances
}

fn topdown(adj_matrix: &Vec<Vec<usize>>, frontier: &mut VecDeque<usize>, distances: &mut Vec<usize>, k: usize) {
    // TOPDOWN function to explore neighbors of the current frontier
    /*
    Processes the current frontier of nodes to explore their neighbors.
    Updates distances for newly discovered nodes.
    */

    let mut next_frontier = VecDeque::new();

    while let Some(v) = frontier.pop_front() {
        for &neighbor in &adj_matrix[v] {
            if distances[neighbor] == usize::MAX {
                distances[neighbor] = k;
                next_frontier.push_back(neighbor);
            }
        }
    }

    *frontier = next_frontier;  // Move to the next level of frontier
}