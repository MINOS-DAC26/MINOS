# <u>MINOS</u>: <u>M</u>ulti-Chiplet <u>IN</u>ference <u>O</u>ptimization <u>S</u>ystem

**MINOS** is the accompanying artifact repository for the paper  
**“Scalable Energy-Aware Optimization of Static and Dynamic Inference Across Multiple Chiplets.”**

This codebase provides a unified framework for **multi-chiplet architecture, mapping, and scheduling**, centered around an **energy-aware MIQP formulation**. The goal is to expose a complete, extensible workflow that enables researchers to reproduce the core results, explore new hardware configurations, and build further experiments. It was developed under the premise that modern AI/ML inference is increasingly moving toward **many-chiplet systems**, each equipped with substantial **3D-integrated on-chip memory**. Supporting large models efficiently — from edge devices to cloud deployments — requires inference hardware that is simultaneously **low-energy**, **low-latency**, **high-throughput**, **cost-effective**, and applicable to both **static** and **dynamic** workloads. Achieving this demands coordinated co-optimization across inter-chiplet communication, chiplet memory organization, operator mapping, and fine-grained spatiotemporal power management; especially as workloads feature KV-cache transfers, Mixture of Experts (MoE), Test Time Compute (TTC), and other data-dependent operations.

**MINOS** is designed for these settings: It implements a flexible **Mixed-Integer Quadratic Program (MIQP)** that jointly optimizes architecture, mapping, scheduling, and power behavior under realistic and **verified hardware assumptions**. The formulation supports large workloads (e.g., 10k+ operator LLMs), heterogeneous and homogeneous chiplet types, dynamic inference variants, and wide-ranging interconnect, memory, and power-gating parameters. Through efficient warm-start heuristics and Dream-Chip-based bounds, MINOS produces **provably optimal or tightly bounded** solutions in **minutes**. 


## 1. Repository Structure

```text
MINOS/
│
├── src/                     # Main codebase for multi-chiplet mapping and scheduling
│   │
│   ├── model_graphs/        # Annotated model graphs used as workloads
│   │   ├── hw/              # Graphs derived from hardware measurements
│   │   └── rtl/             # Graphs derived from RTL simulations
│   │
│   └── partitioning/        # Core MIQP-based optimization and evaluation flow
│       └── scripts/         # Scripts to reproduce the paper’s main artifacts
│
├── figures/                 # Result figures used in the paper and documentation
│
└── requirements.txt
```

## 2. Requirements & Installation

MINOS requires:

- **Python 3.11+**
- **Gurobi** with a valid academic or personal license  
  → https://www.gurobi.com/academia/academic-program-and-licenses/

Set up the environment:

```bash
python -m venv minos-env
source minos-env/bin/activate
pip install -r requirements.txt
```

Verify that Gurobi is available:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

## 3. Usage and Experiment Setup

The `src/` submodule contains the core implementation of the **architecture, mapping, and scheduling** flow. The main entry point is:

```bash
python partition.py --model MODEL
```

This command runs MINOS with the default settings and architectural assumptions used in the study. All functionality is accessible through **command-line flags**, enabling fine-grained control over:

- **model choice** and **synthetically scaled generations**
- **chiplet** and **sub-accelerator configuration**
- **interconnect** specifications and **topology**
- **annotation method** (hardware/RTL-based estimates)  
- **power-management** settings (idle, power gating)  
- **solver behavior** (targets, heuristics, warm starts, early termination, ...)

For a complete list of options, call:

```bash
python partition.py --help
```

---

### 3.1. Model and Workload Selection

- **`--model`**
  Selects a built-in workload (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet151`, `mobilebert`, `bert_encoder`, `bert`, `bert-large`, `llama1b`, `llama8b`, `llama70b`).

- **Model name modifiers** (append to any supported model, including the base configuration):
  - `_deeper` → doubles model depth  
  - `_wider` → doubles model width  
  - `_hybrid` → doubles both depth and width  

- **`--generation`**  
  Enables synthetic scaling of an existing model graph (for `resnet18`, `mobilebert`, `bert_encoder`, `bert`).

---

### 3.2. Chiplet & Accelerator Configuration

- **`--chips`**  
  Number of chiplets in the system.

- **`--subAccs`**  
  Number of sub-accelerators (SAs) per chiplet.

- **`--topology`**  
  Inter-chiplet network topology (e.g., `1D_RING`, `1/2D_MESH`, `2/3D_TORUS`, `ALL_TO_ALL`).

---

### 3.3. Communication and Memory Scaling

- **`--communication_time_scale`**  
  Scales inter-chiplet (D2D) latency. Values `< 1` improve latency; values `> 1` increase it.

- **`--communication_energy_scale`**  
  Scales inter-chiplet energy cost analogously.

---

### 3.4. Power Management Settings

- **`--leakage`**  
  `1` uses traditional idle/leakage power;  
  `0` enables power-gated operation.

- **`--idling_scale`**  
  Sets idle power as a fraction of average dynamic power.

- **`--wakeup_time_scale`**  
  Scales the wakeup delay for power-gated SAs.

- **`--wakeup_cost_scale`**  
  Scales the wakeup energy overhead.

---

### 3.5. Annotation Method

- **`--estimates`**  
  Selects the annotation source (`hw` or `rtl`) used for operator energy/latency estimates.

---

### 3.6. Solver Configuration

- **`--target`**  
  Optimization objective (`Latency`, `Energy`, `EDP`, `EPT`, etc.).

- **`--heuristic`**  
  Selects the warm-start heuristic (`none`, `all`, or a specific heuristic).

- **`--preloading`**  
  Loads prior MIQP models to accelerate setup.

- **`--solution`**  
  Loads an existing solution as a warm start.

- **`--early_termination`**  
  Enables early stopping; solver execution can be interrupted at any time with `Ctrl+C`.


## 4. Further Documentation

For deeper explanations of the optimization flow, workload handling, and artifact reproduction, refer to the following module-level documents:

- **[src/partitioning/README.md](src/partitioning/README.md)** 
  Describes the full architecture, mapping, and scheduling formulation, including MIQP construction, constraint structure, and warm-start logic.

- **[src/partitioning/scripts/README.md](src/partitioning/scripts/README.md)** 
  Documents the scripts used to reproduce the paper’s artifacts, regenerate tables/figures, and run experiment sets.

- **[src/model_graphs/README.md](src/model_graphs/README.md)**
  Explains the model-graph format and how to add new workloads.

