use std::collections::HashSet;

pub fn check_edge_symmetry_conditions(
    initial_edge1: (usize, usize),
    selected_edges1: &Vec<(usize, usize)>,
    selected_edges2: &Vec<(usize, usize)>,
    g: usize,
) -> bool {
    let mut condition_met = false;

    // Special case: Check if an edge is symmetrical to itself (only happens if g == 2)
    if g == 2 {
        if is_edge_symmetrical_to_itself(selected_edges1) {
            //println!("Edge 1 is symmetrical to itself: {:?}", selected_edges1);
            condition_met = true;
        }

        if is_edge_symmetrical_to_itself(selected_edges2) {
            //println!("Edge 2 is symmetrical to itself: {:?}", selected_edges2);
            condition_met = true;
        }
    }

    // Check if both initial selected edges are symmetrical to each other
    if are_edges_symmetrical(initial_edge1, selected_edges2) {
        //println!("The selected edges are symmetrical to each other.");
        condition_met = true;
    } else {
        //println!("The selected edges are NOT symmetrical to each other.");
    }

    condition_met
}


// Function to check if an edge is symmetrical to itself (special case for g = 2)
pub fn is_edge_symmetrical_to_itself(selected_edges: &Vec<(usize, usize)>) -> bool {
    let mut unique_vertices = HashSet::new();

    for &(start_vertex, end_vertex) in selected_edges {
        unique_vertices.insert(start_vertex);
        unique_vertices.insert(end_vertex);
    }

    unique_vertices.len() == 2
}

// Function to check if an edge (in either direction) exists in the list of symmetrical edges
pub fn are_edges_symmetrical(edge: (usize, usize), symmetrical_edges: &Vec<(usize, usize)>) -> bool {
    let (start_vertex, end_vertex) = edge;

    symmetrical_edges.contains(&(start_vertex, end_vertex)) || symmetrical_edges.contains(&(end_vertex, start_vertex))
}

pub fn check_duplicated_vertex(edge1: (usize, usize), edge2: (usize, usize)) -> bool {
    /* check if two edges have unique vertices */

    let (start_vertex1, end_vertex1) = edge1;
    let (start_vertex2, end_vertex2) = edge2;

    let mut unique_vertices = HashSet::new();
    unique_vertices.insert(start_vertex1);
    unique_vertices.insert(end_vertex1);
    unique_vertices.insert(start_vertex2);
    unique_vertices.insert(end_vertex2);

    unique_vertices.len() == 4
}


