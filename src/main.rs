mod helper_functions;

use helper_functions::selected_edge_requirements::{
    check_edge_symmetry_conditions,
    check_duplicated_vertex
};
use helper_functions::base::{
    find_symmetrical_related_edges,
    modify_adjacency_matrix, 
    select_random_edge
};
use helper_functions::graph_requirements::{
    is_graph_connected,
    check_if_graph_g_symmetrical
};
use helper_functions::bfs_apsp::{
    serial_bfs_apsp,
};

use helper_functions::lower_bounds::{
    calculate_diameter_lower_bound,
    calculate_aspl_lower_bound
};

use std::env;
use std::process::{Command, exit};
use rand::Rng;
use std::time::{Instant, Duration};
use plotters::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage:\n
            1. Single execution mode:
               cargo run -- SINGLE nodes <n> degree <d> groups <g> temperature <beta> max_candidates <max_candidates> max_time[minutes] <max_time>

            2. Comparison mode:
               cargo run -- COMPARISON nodes <n> degree <d> groups [<g1>,<g2>,...,<g_m>] temperatures [<beta_1>,...,<beta_k>] max_candidates <max_candidates> max_time[minutes] <max_time>
            ");
        exit(1);
    }

    match args[1].as_str() {
        "SINGLE" => {
            if args.len() != 14 {
                eprintln!(
                    "Error: Incorrect number of arguments for SINGLE mode.\nExpected format:\n
                     cargo run -- SINGLE nodes <n> degree <d> groups <g> temperature <beta> max_candidates <max_candidates> max_time[minutes] <max_time>"
                );
                exit(1);
            }

            // Parsing SINGLE mode arguments
            let n: i32 = args[3].parse().expect("Error: 'n' must be an integer.");
            let d: i32 = args[5].parse().expect("Error: 'd' must be an integer.");
            let g: i32 = args[7].parse().expect("Error: 'g' must be an integer.");
            let beta: f64 = args[9].parse().expect("Error: 'beta' must be a floating-point number.");
            let max_candidates: i32 = args[11].parse().expect("Error: 'max_candidates' must be an integer.");
            let max_time: u64 = args[13].parse().expect("Error: 'max_time' must be an integer (minutes).");

            // Validation checks
            if n < d {
                eprintln!("Error: 'n' must be greater than or equal to 'd'.");
                exit(1);
            }
            if n % g != 0 {
                eprintln!("Error: 'g' must be a true divisor of 'n'.");
                exit(1);
            }

            println!("\nRunning Fixed Temperature Simulated Annealing...");
            let (final_matrix, _) = fixed_temperature_simulated_annealing(n, d, g, beta, max_candidates, max_time);

            let n_div_g = (n / g) as usize;
            let (diameter, aspl) = serial_bfs_apsp(&final_matrix, n_div_g);
            
            let diameter_lower_bound = calculate_diameter_lower_bound(n, d);
            let aspl_lower_bound = calculate_aspl_lower_bound(n, d);
            
            // Displaying the final results
            println!("\nFinal Adjacency Matrix:");
            for (index, row) in final_matrix.iter().enumerate() {
                println!("Node {}: {:?}", index, row);
            }

            println!("\n----------------------- Final Results ----------------------------");
            println!("Achieved Diameter: {}", diameter);
            println!("Achieved ASPL: {:.12}", aspl);
            
            println!("\n--- Theoretical Lower Bounds ---");
            println!("Theoretical Lower Bound for Diameter (K_{{{},{}}}): {}", n, d, diameter_lower_bound);
            println!("Theoretical Lower Bound for ASPL (L_{{{},{}}}): {:.12}", n, d, aspl_lower_bound);
            
            println!("\n--- Comparison ---");
            println!("Diameter Difference: {}", diameter - (diameter_lower_bound as usize));
            println!("ASPL Difference: {:.12}", aspl - aspl_lower_bound);
        }

        "COMPARISON" => {
            if args.len() != 14 {
                eprintln!(
                    "Error: Incorrect number of arguments for COMPARISON mode.\nExpected format:\n
                     cargo run -- COMPARISON nodes <n> degree <d> groups [<g1>,<g2>,...,<g_m>] temperatures [<beta_1>,...,<beta_k>] max_candidates <max_candidates> max_time[minutes] <max_time>"
                );
                exit(1);
            }

            // Parsing COMPARISON mode arguments
            let n: i32 = args[3].parse().expect("Error: 'n' must be an integer.");
            let d: i32 = args[5].parse().expect("Error: 'd' must be an integer.");

            // Extracting groups and temperatures (comma-separated values)
            let groups: Vec<i32> = args[7]
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .map(|s| s.trim().parse().expect("Error: 'groups' must be integers separated by commas."))
                .collect();

            let temperatures: Vec<f64> = args[9]
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .map(|s| s.trim().parse().expect("Error: 'temperatures' must be floating-point numbers separated by commas."))
                .collect();

            let max_candidates: i32 = args[11].parse().expect("Error: 'max_candidates' must be an integer.");
            let max_time: u64 = args[13].parse().expect("Error: 'max_time' must be an integer (minutes).");

            // Validation checks
            if n < d {
                eprintln!("Error: 'n' must be greater than or equal to 'd'.");
                exit(1);
            }
            for &g in &groups {
                if n % g != 0 {
                    eprintln!("Error: Each 'g' must be a true divisor of 'n'.");
                    exit(1);
                }
            }

            let aspl_lower_bound = calculate_aspl_lower_bound(n, d);

            let mut results = Vec::new(); // Store (temperature, group, aspl_gap, candidate_count) data

            for &beta in &temperatures {
                for &g in &groups {
                    println!("Running SA for Temp: {}, Group: {}", beta, g);

                    let (final_matrix, candidate_count) = fixed_temperature_simulated_annealing(n, d, g, beta, max_candidates, max_time);
                    let n_div_g = (n / g) as usize;
                    let (_, aspl) = serial_bfs_apsp(&final_matrix, n_div_g);
                    
                    let aspl_gap = aspl - aspl_lower_bound;
                    println!("Temp: {}, Group: {}, ASPL Gap: {:.12}, Candidates: {}", beta, g, aspl_gap, candidate_count);

                    results.push((beta, g, aspl_gap, candidate_count));
                }
            }

            generate_rust_plot(&temperatures, &groups, &results);
        }

        _ => {
            eprintln!("Error: Invalid execution mode. Use -- SINGLE or -- COMPARISON.");
            exit(1);
        }
    }
}

fn generate_rust_plot(temperatures: &[f64], groups: &[i32], results: &Vec<(f64, i32, f64, i32)>) {
    let root = BitMapBackend::new("comparison_plot.png", (1200, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    let areas = root.split_evenly((1, temperatures.len()));

    for (i, &temp) in temperatures.iter().enumerate() {
        let temp_results: Vec<&(f64, i32, f64, i32)> = results.iter()
            .filter(|(beta, _, _, _)| *beta == temp)
            .collect();

        let aspl_gaps: Vec<f64> = temp_results.iter().map(|(_, _, gap, _)| *gap).collect();
        let candidate_counts: Vec<i32> = temp_results.iter().map(|(_, _, _, count)| *count).collect();

        let min_aspl_gap = aspl_gaps.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_aspl_gap = aspl_gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let padding = (max_aspl_gap - min_aspl_gap) * 0.1;


        let mut chart = ChartBuilder::on(&areas[i])
            .caption(format!("Temperature: {}", temp), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(groups[0]..groups[groups.len() - 1], (min_aspl_gap - padding)..(max_aspl_gap + padding))
            .unwrap();

        chart.configure_mesh()
            .x_desc("Groups")
            .y_desc("ASPL Gap")
            .draw()
            .unwrap();

        chart.draw_series(LineSeries::new(
            groups.iter().zip(aspl_gaps.iter()).map(|(&g, &gap)| (g, gap)),
            &BLUE,
        )).unwrap()
        .label("ASPL Gap")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart.draw_series(PointSeries::of_element(
            groups.iter().zip(aspl_gaps.iter()).zip(candidate_counts.iter()).map(|((&g, &gap), &count)| (g, gap, count)), // Extract `count`
            5,
            &RED,
            &|(g, gap, count), size, style| { // Use `(g, gap, count)` in the closure
                EmptyElement::at((g, gap))
                + Circle::new((0, 0), size, style.filled())
                + Text::new(format!("{}", count), (10, -10), ("sans-serif", 15).into_font()) // Display candidate count
            },
        )).unwrap();
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();
    }

    println!("\nMulti-plot saved as 'comparison_plot.png'.");
}


fn fixed_temperature_simulated_annealing(n: i32, d: i32, g: i32, beta: f64, max_candidates: i32, max_time: u64) -> (Vec<Vec<usize>>, i32) {
    let mut duplicated_base_graph_matrix = generate_duplicated_basegraph(n, d, g);
    let n_div_g = (n / g) as usize;
    let start_time = Instant::now(); // Start tracking time
    let max_duration = Duration::from_secs(max_time * 60); // Convert minutes to seconds

    let mut initial_edge;
    let mut current_state;

    // Keep selecting initial edge until a valid transformation is found
    loop {
        initial_edge = select_random_edge(&duplicated_base_graph_matrix);

        if let Some(matrix) = g1_opt_method(initial_edge, &mut duplicated_base_graph_matrix, g as usize) {
            current_state = matrix;
            break; // Successfully found a valid transformation
        }
    }

    let mut current_energy = calc_energy(&current_state, n_div_g);
    let aspl_lower_bound = calculate_aspl_lower_bound(n, d);
    let mut candidate_count = 0;

    // Track the best state found
    let mut best_adj_matrix = current_state.clone();
    let mut best_energy = current_energy;

    while candidate_count < max_candidates && current_energy > aspl_lower_bound {
        if start_time.elapsed() >= max_duration {
            println!("Max time ({:.2} minutes) reached. Terminating optimization.", max_time);
            break;
        }

        println!("----------------------------------------------------------------------------------");
        println!("Candidate Count: {}", candidate_count);
        println!("Candidate Energy: {:.12}", current_energy);

        // Termination condition: break if first 12 digits of ASPL match lower bound
        let energy_rounded = (current_energy * 1e12).trunc() as u64;
        let aspl_rounded = (aspl_lower_bound * 1e12).trunc() as u64;

        if energy_rounded == aspl_rounded {
            println!("First 12 digits of ASPL and Current Energy match! Terminating optimization.");
            break;
        }

        loop {
            // Check if max time in minutes is reached
            if start_time.elapsed() >= max_duration {
                println!("Max time ({:.2} minutes) reached. Terminating optimization.", max_time);
                break;
            }

            let candidate_state = edge_exchange(&mut current_state.clone(), g as usize);
            let candidate_energy = calc_energy(&candidate_state, n_div_g);
            let energy_difference = (candidate_energy - current_energy) * (n.pow(2) as f64) * (n as f64) / (g as f64);

            let mut rng = rand::thread_rng();
            let acceptance_probability = if energy_difference < 0.0 {
                1.0
            } else {
                f64::exp(-energy_difference / beta)
            };
            let random_value: f64 = rng.gen();

            if random_value < acceptance_probability {

                current_state = candidate_state;
                current_energy = candidate_energy;

                // Update the best state if the new one is better
                if current_energy < best_energy {
                    best_adj_matrix = current_state.clone();
                    best_energy = current_energy;
                }

                break;
            }
        }

        candidate_count += 1;
    }

    println!("Termination Condition Reached: {} iterations, ASPL Lower Bound ({:.12}) met, or Max Time ({:.2} minutes) reached.", 
             candidate_count, aspl_lower_bound, max_time);
    
    (best_adj_matrix, candidate_count) // Return the best found adjacency matrix
}


fn calc_energy(adj_matrix: &Vec<Vec<usize>>, n_div_g: usize) -> f64 {
    let (_, aspl) = serial_bfs_apsp(adj_matrix, n_div_g);
    //println!("ASPL: {}", aspl);
    aspl 
}

fn edge_exchange(adj_matrix: &mut Vec<Vec<usize>>, g: usize) -> Vec<Vec<usize>> {
    let total_nodes = adj_matrix.len();
    let mut rng = rand::thread_rng();

    loop {
        let (initial_edge1, initial_edge2) = loop {
            let edge1 = select_random_edge(adj_matrix);
            let edge2 = select_random_edge(adj_matrix);

            if check_duplicated_vertex(edge1, edge2) {
                /*
                println!("\nSelected Edges for Exchange:");
                println!("Edge 1: {:?}", edge1);
                println!("Edge 2: {:?}", edge2);
                */
                break (edge1, edge2);
            }
        };

        let selected_edges1 = find_symmetrical_related_edges(initial_edge1, total_nodes, g);
        let selected_edges2 = find_symmetrical_related_edges(initial_edge2, total_nodes, g);

        //println!("\nSelect Edges Init 1: {:?}", selected_edges1);
        //println!("Select Edges Init 2: {:?}", selected_edges2);

        if check_edge_symmetry_conditions(initial_edge1, &selected_edges1, &selected_edges2, g) {
            let chosen_edge = if rng.gen_bool(0.5) { initial_edge1 } else { initial_edge2 };
            //println!("\nCalling g1_opt_method with edge: {:?}", chosen_edge);

            if let Some(matrix) = g1_opt_method(chosen_edge, adj_matrix, g) {
                *adj_matrix = matrix;
            } else {
                //println!("Retrying edge selection in edge_exchange...");
                continue;
            }
        } else {
            //println!("\nNo symmetry conditions met. Now Performing g2_opt_method.");

            if let Some(matrix) = g2_opt_method(adj_matrix, initial_edge1, initial_edge2, &selected_edges1, &selected_edges2, g) {
                *adj_matrix = matrix;
            } else {
                //println!("Retrying edge selection in edge_exchange...");
                continue;
            }
        }

        if is_graph_connected(adj_matrix) {
            //println!("\nGraph is connected after modification!");
            break;
        } else {
            //println!("\nGraph is NOT connected. Continuing with further modifications...");
        }
    }
    adj_matrix.clone()
}


fn g1_opt_method(initial_edge: (usize, usize), adj_matrix: &mut Vec<Vec<usize>>, g: usize) -> Option<Vec<Vec<usize>>> {
    let total_nodes = adj_matrix.len();
    let mut rng = rand::thread_rng();
    let mut current_edge = initial_edge;

    loop {
        // Step 1: Find symmetrical related edges
        let selected_edges = find_symmetrical_related_edges(current_edge, total_nodes, g);
        //println!("Selected Edges: {:?}", selected_edges);

        // Step 2: Generate new edges based on random r
        let mut new_edges: Vec<(usize, usize)> = Vec::new();
        let total_edges = selected_edges.len();
        let r = rng.gen_range(1..g as usize);
        //println!("Randomly selected r: {}", r);

        for i in 0..total_edges {
            let start_node = selected_edges[i].0;
            let end_node = selected_edges[(i + r) % total_edges].1;  // Wrap-around using modulo
            new_edges.push((start_node, end_node));
        }

        //println!("New Edges: {:?}", new_edges);

        if edge_already_exists(adj_matrix, &new_edges) {
            //println!("One or more edges in new_edges already exist. Returning to reselection.");
            return None;
        }

        // Step 3: Modify the adjacency matrix
        modify_adjacency_matrix(adj_matrix, &selected_edges, &new_edges);

        // Step 4: Check if the graph is connected
        if is_graph_connected(adj_matrix) {
            //println!("\nGraph is connected after modification!");
            break;
        } else {
            //println!("\nGraph is NOT connected. Retrying with a new random edge...");
            current_edge = select_random_edge(adj_matrix);  // Select a new random edge for the next iteration
        }
    }

    /*
    println!("\ng1_opt Modified Adjacency Matrix:");
    for (index, row) in adj_matrix.iter().enumerate() {
        println!("Node {}: {:?}", index, row);
    }
    */

    if check_if_graph_g_symmetrical(&mut adj_matrix.clone(), g) {
        return Some(adj_matrix.clone());
    } else {
        eprintln!("Error After 1g-opt: The graph is not g-symmetrical.");
        exit(1);
    }
}

fn g2_opt_method(
    adj_matrix: &mut Vec<Vec<usize>>, 
    initial_edge1: (usize, usize), 
    initial_edge2: (usize, usize), 
    selected_edges1: &Vec<(usize, usize)>, 
    selected_edges2: &Vec<(usize, usize)>, 
    g: usize
) -> Option<Vec<Vec<usize>>> {

    let total_nodes = adj_matrix.len();
    let n_div_g = total_nodes / g;

    // Calculate the group for each initial edge based on the start vertex
    let (start_vertex1, _) = initial_edge1;
    let (start_vertex2, _) = initial_edge2;

    let group1 = start_vertex1 / n_div_g;
    let group2 = start_vertex2 / n_div_g;

    /*
    println!("Edge 1 Start Vertex: {}, Group: {}", start_vertex1, group1);
    println!("Edge 2 Start Vertex: {}, Group: {}", start_vertex2, group2);
    */

    let group_distance = if group1 > group2 {
        group1 - group2
    } else {
        group2 - group1
    };
    /*
    println!("Group Distance: {}", group_distance);

    */

    let mut new_edges_startcomb: Vec<(usize, usize)> = Vec::new();
    let mut new_edges_endcomb: Vec<(usize, usize)> = Vec::new();

    for i in 0..g {
        let start_node = selected_edges1[i].0;
        let end_node = selected_edges2[(i + group_distance) % g].0;
        new_edges_startcomb.push((start_node, end_node));
    }

    for i in 0..g {
        let start_node = selected_edges1[i].1;
        let end_node = selected_edges2[(i + group_distance) % g].1;
        new_edges_endcomb.push((start_node, end_node));
    }

    /*
    println!("New Edges Start Comb: {:?}", new_edges_startcomb);
    println!("New Edges End Comb: {:?}", new_edges_endcomb);
    */

    //for each edge in new_edges start_comb and new_edges_endcomb check that the first vertex and the second are not identical
    // Check for self-loops (edges where start and end vertex are identical)
    if contains_self_loops(&new_edges_startcomb) || contains_self_loops(&new_edges_endcomb) {
        //println!("Self-loop detected in new edges. Returning to reselection.");
        return None;
    }

    if edge_already_exists(adj_matrix, &new_edges_startcomb) || edge_already_exists(adj_matrix, &new_edges_endcomb) {
        //println!("One or more edges in new_edges already exist. Returning to reselection.");
        return None;
    }

    modify_adjacency_matrix(adj_matrix, selected_edges1, &new_edges_startcomb);
    modify_adjacency_matrix(adj_matrix, selected_edges2, &new_edges_endcomb);

    /*
    println!("\nAdjacency-matrix after 2g-opt:");
    for (index, row) in adj_matrix.iter().enumerate() {
        println!("Node {}: {:?}", index, row);
    }
    */

    if check_if_graph_g_symmetrical(&mut adj_matrix.clone(), g as usize) {
        return  Some(adj_matrix.clone());
    } else {
        eprintln!("Error After 2g-opt: The graph is not g-symmetrical.");
        exit(1);
    }
}      

/// Checks if any edge in `new_edges` already exists in `adj_matrix`
fn edge_already_exists(adj_matrix: &Vec<Vec<usize>>, new_edges: &Vec<(usize, usize)>) -> bool {
    for &(u, v) in new_edges {
        if adj_matrix[u].contains(&v) || adj_matrix[v].contains(&u) {
            return true; // Edge already exists
        }
    }
    false
}

/// Checks if the edge list contains self-loops (edges with identical start and end vertices)
fn contains_self_loops(edges: &Vec<(usize, usize)>) -> bool {
    edges.iter().any(|&(u, v)| u == v)
}


fn generate_duplicated_basegraph(n: i32, d: i32, g: i32) -> Vec<Vec<usize>> {

    let n_div_g = n / g;

    // Construct the command to call the Python script
    let python_script = "generate_base_graphs/create-random.py";
    let output = Command::new("python3")
        .arg(python_script)
        .arg(n_div_g.to_string())
        .arg(d.to_string())
        .output()
        .expect("Failed to execute Python script");

    if !output.status.success() {
        eprintln!("Error running Python script:");
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        exit(1);
    }

    // Parse adjacency list from stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    let base_graph_adjacency_matrix: Vec<Vec<usize>> = stdout
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|x| x.parse::<usize>().expect("Failed to parse neighbor"))
                .collect()
        })
        .collect();

    // Generate duplicated_base_graph_matrix by duplicating the base graph
    let mut duplicated_base_graph_matrix: Vec<Vec<usize>> = Vec::new();
    
    // First, push the base graph itself
    for row in &base_graph_adjacency_matrix {
        duplicated_base_graph_matrix.push(row.clone());
    }

    // Then, duplicate the graph for each group
    for m in 1..g as usize {
        let offset = m * n_div_g as usize;
        for row in &base_graph_adjacency_matrix {
            let new_row: Vec<usize> = row.iter().map(|&val| val + offset).collect();
            duplicated_base_graph_matrix.push(new_row);
        }
    }

    /*
    println!("\nDuplicated Base Graph Adjacency Matrix:");
    for (index, row) in duplicated_base_graph_matrix.iter().enumerate() {
        println!("Node {}: {:?}", index, row);
    }
    */

    if check_if_graph_g_symmetrical(&mut duplicated_base_graph_matrix.clone(), g as usize) {
        return duplicated_base_graph_matrix;
    } else {
        eprintln!("Error in Base Graph Generation: The graph is not g-symmetrical.");
        exit(1);
    }
}
