# MiniGrid Agent Behavior Analysis

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project analyzes the sub-optimal behavior of a Proximal Policy Optimization (PPO) agent in the `MiniGrid-DoorKey-6x6-v0` environment. It identifies "critical states" where the agent's actions begin to deviate significantly from an optimal path calculated by an A* search algorithm.

The goal is to produce a dataset of these critical states, which can be used as input for further analysis or techniques like counterfactual generation to understand and correct agent failures.

---

### ## 🚀 Visual Snapshot

The final output of the analysis is a side-by-side comparison of the agent's actual path versus the optimal A* path for a given critical episode.

![Path Comparison Snapshot](comparison_plot.png)
*(Note: You should replace `comparison_plot.png` with an actual image file in your repository.)*

---

### ## ⚙️ Setup and Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    * **On macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

### ## 🏃‍♀️ How to Run the Project

The project is designed to be run in a sequence. The output of one script becomes the input for the next.

#### **Step 1: Train the PPO Agent**

First, train the reinforcement learning agent. This script will save the trained model to a `.zip` file.
```bash
python train_agent.py
```
* **Input**: None
* **Output**: A model file (e.g., `ppo_doorkey_6x6.zip`)

#### **Step 2: Generate Episode Logs**

Run the trained agent to collect data from successful episodes.
```bash
python generate_logs.py
```
* **Input**: The trained model `.zip` file from Step 1.
* **Output**: A log file of successful trajectories (`episode_logs/episodes_full.json`).

#### **Step 3: Find Critical States**

Analyze the logs to find episodes where the agent was sub-optimal and identify the first point of deviation.
```bash
python find_cs.py
```
* **Input**: The `episodes_full.json` file from Step 2.
* **Output**: A file containing all identified critical states (`critical_states.json`).

#### **Step 4: Visualize a Critical Path**

Generate a side-by-side plot comparing the agent's path and the A* path for a specific critical episode.
```bash
python compare_critical_paths.py
```
* **Input**: `critical_states.json` and `episodes_full.json`.
* **Output**: A Matplotlib window showing the visual comparison.
    *(You can change the `CRITICAL_STATE_INDEX` inside the script to view different episodes.)*

---

### ## Project Structure
```
.
├── train_agent.py             # Script to train the PPO model
├── generate_logs.py           # Script to log successful episodes
├── find_cs.py                 # Script to find critical states using A*
├── compare_critical_paths.py  # Script to visualize path comparisons
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```