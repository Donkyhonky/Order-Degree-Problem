## Running the Rust Code

To run the code use one of the following commands, and make sure to go through the Prerequisites

1. Single execution mode:
    cargo run -- SINGLE nodes <n> degree <d> groups <g> temperature <beta> max_candidates <max_candidates> max_time[minutes] <max_time>

2. Comparison mode:
    cargo run -- COMPARISON nodes <n> degree <d> groups [<g1>,<g2>,...,<g_m>] temperatures [<beta_1>,...,<beta_k>] max_candidates <max_candidates> max_time[minutes] <max_time>

```bash
cargo run -- <n> <d> <g> <beta> <max_candidates>
```

Where:
- `<n>`: Number of nodes
- `<d>`: Degree
- `<g>`: Number of groups
- `<beta>`: Fixed temperature (floating-point)
- `<max_candidates>`: Maximum number of selected candidates in the annealing loop
- `<max_time>`: Maximum time in minutes to run the annealing loop

### Examples

1. Single execution mode:
```bash
cargo run -- SINGLE nodes 72 degree 4 group 9 temperature 144 max_candidates 1000 max_time[minutes] 10
```
2. Comparison mode:
```bash
cargo run -- COMPARISON nodes 72 degree 4 groups [2,3,4,6,8,9,12] temperature [180,96] max_candidates 1000 max_time[minutes] 10
```

## Prerequisites

Before running the code, ensure you have the following setup:

1. **Python Environment:**
   - Create a Python environment (using `venv`, `conda`, or another tool).
   - Install the required Python package `networkx`:
     - Using `pip`:
       ```bash
       pip install networkx
       ```
     - Using `conda`:
       ```bash
       conda install -c conda-forge networkx
       ```

2. **Graph Generation Requirement**
    - Install plotter for graph generation: https://docs.rs/plotters/latest/plotters/#quick-start
    - Using ubuntu:
      ```bash
      sudo apt install pkg-config libfreetype6-dev libfontconfig1-dev
      ```


## Initial Graph Generation

The initial graph generation is handled by the Python script `generate_base_graphs/create-random.py`. This script uses the `networkx` function `random_regular_graph` to create the base graph, as there was no identical function implemented in Rust. This script was provided by the Graph Golf competition.

Once the Python Envirenment is active the main.rs file will call this `create-random.py` file.

