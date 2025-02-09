use std::f64;

// Function to calculate the theoretical lower bound for Diameter (K_{n,d})
pub fn calculate_diameter_lower_bound(n: i32, d: i32) -> i32 {
    if d == 2 {
        ((n - 1) as f64 / 2.0).ceil() as i32
    } else {
        let log_base = (d - 1) as f64;
        let value = ( ((n - 1) * (d - 2)) as f64) / (d as f64) + 1.0;

        let diameter = (value.log(log_base)).ceil() as i32 ;
        diameter
    }
}

// Function to calculate the theoretical lower bound for ASPL (L_{n,d})
pub fn calculate_aspl_lower_bound(n: i32, d: i32) -> f64 {
    let k_nd = calculate_diameter_lower_bound(n, d);

    if k_nd == 1 {
        return 1.0;
    }

    // Calculate S_{n,d} = sum_{i=1}^{K_{n,d}-1} i * d * (d-1)^{i-1}
    let s_nd: i32 = (1..k_nd)
        .map(|i| i * d * (d - 1).pow((i - 1) as u32))
        .sum();

    // Calculate R_{n,d} = (n - 1) - sum_{i=1}^{K_{n,d}-1} d * (d-1)^{i-1}
    let r_nd: i32 = (n - 1)
        - (1..k_nd)
            .map(|i| d * (d - 1).pow((i - 1) as u32))
            .sum::<i32>();

    // Final ASPL Calculation
    (s_nd as f64 + (k_nd as f64 * r_nd as f64)) / (n as f64 - 1.0)
}

