
import math
import time
import random
import matplotlib
import csv
from pysyncobj import SyncObj, SyncObjConf, replicated
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from functools import lru_cache
# matplotlib.rcParams['text.usetex'] = True
import logging
from tabulate import tabulate
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
import seaborn as sns

# Set style, context, and font scale for the plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.75)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
# plt.rcParams['text.usetex'] = True
# Set color palette
colors = sns.color_palette("tab10")


log_table = []
N_AGENTS = 4
nodes = [f"localhost:{4321 + i}" for i in range(N_AGENTS)]
fig, ax = plt.subplots()

class ElectionTimer:
    def __init__(self):
        self.timeout = self.reset()

    def reset(self):
        election_timeout = 2.3
        self.timeout = random.uniform(election_timeout, 2 * election_timeout)
        return self.timeout
    
class DistributedFormation(SyncObj):
    def __init__(self, selfNodeAddr, otherNodeAddrs):
        conf = SyncObjConf(autoTick=True)
        super().__init__(selfNodeAddr, otherNodeAddrs, conf)
        self.leader_id = 0
        self.failed = False
        self.agent_history = []
        self.first_frame = True
        self.agents = [
            (random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(N_AGENTS)
        ]
        self.goals = [(0.0, 0.0) for _ in range(N_AGENTS)]
        self.set_goals()
        self.leader_history = []
        self.error_history = []
        self.term = 0
        self.last_heartbeat_received = time.time()
        self.status = 'follower'
        self.node_id = selfNodeAddr
        self.election_timer = ElectionTimer()
        self._otherNodesAddrs = otherNodeAddrs
        self.agent_velocities = [
            (random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(N_AGENTS)
        ]
        self.election_start_time = time.time() + self._reset_election_timer()

    def _reset_election_timer(self):
        return self.election_timer.reset()

    # @lru_cache(maxsize=None)
    def _isReady(self, frame):
        return not self.first_frame and (time.time() - self.last_heartbeat_received < self._reset_election_timer())

    # @replicated
    def _addNode(self, new_node_address):
        if new_node_address not in self._otherNodesAddrs:
            self._otherNodesAddrs.append(new_node_address)

    # @replicated
    def add_node(self, new_node_address):
        self._addNode(new_node_address)

    # @replicated
    def add_agent(self, new_position):
        self.agents.append(new_position)
        self.goals.append((0.0, 0.0))
        self.set_goals()
        print('n_agents add_agent', len(self.agents))


    # @replicated
    def remove_agent(self, agent_id):
        if 0 <= agent_id < len(self.agents):
            del self.agents[agent_id]
            del self.goals[agent_id]
            self.set_goals()

    # Modify the set_goals method in the DistributedFormation class
    def set_goals(self):
        n_agents = len(self.agents)
        polygon_radius = 1.0
        angle = 2 * math.pi / n_agents

        for i in range(n_agents):
            self.goals[i] = (
                polygon_radius * math.cos(i * angle + angle / 2),
                polygon_radius * math.sin(i * angle + angle / 2),
            )

    # @replicated
    def update_agent(self, agent_id, new_position):
        self.agents[agent_id] = new_position

    # @replicated
    def update_all_agents(self, new_positions):
        self.agents = new_positions

    def single_integrator_dynamics(self, agent_id, goal, dt=0.1, k=1.0):
        current_position = self.agents[agent_id]
        new_velocity = tuple(k * (g - c) for c, g in zip(current_position, goal))
        new_position = tuple(c + v * dt for c, v in zip(current_position, new_velocity))
        return new_position

    def move_leader(self):
        new_positions = [
            self.single_integrator_dynamics(i, self.goals[i])
            for i in range(len(self.agents))
        ]
        self.update_all_agents(new_positions)
    
    def move_follower(self):
        leader_node = distributed_formation_objects[self.leader_id]
        for i in range(len(self.agents)):
            new_position = self.single_integrator_dynamics(i, leader_node.goals[i])
            self.update_agent(i, new_position)

    def fail_node(self, frame_number):
        self.failed = True
        print(f"Node {self.node_id} simulate failure.")
        log_table.append({"type": "simulate failure", "node": self.node_id, "term": self.term, "frame": frame_number})

    def recover_node(self, frame_number):
        self.failed = False
        print(f"Node {self.node_id} simulate recovery.")
        log_table.append({"type": "simulate recovery", "node": self.node_id, "term": self.term, "frame": frame_number})

    def failed_action(self, frame_number):
        # Perform actions when the node fails, e.g., log the failure
        print(f"Node {self.node_id} failed.")
        log_table.append({"type": "failure", "node": self.node_id, "term": self.term, "frame": frame_number})

    def get_status_info(self):
        return (self.node_id, self.status, self.term)

    def leader_update(self):
        new_positions = [
            self.single_integrator_dynamics(i, self.goals[i])
            for i in range(len(self.agents))
        ]
        self.update_all_agents(new_positions)

    # @replicated
    def update_goals(self, new_goals):
        self.goals = new_goals

    @replicated
    def update_agents_positions(self, new_positions):
        self.agents = new_positions

def position_error(formation_obj):
    errors = [
        math.sqrt((px - gx)**2 + (py - gy)**2)
        for (px, py), (gx, gy) in zip(formation_obj.agents, formation_obj.goals)
    ]
    return errors

def detect_failed_nodes(distributed_formation_objects):
    failed_nodes = []
    for idx, formation_obj in enumerate(distributed_formation_objects):
        if formation_obj.failed:
            failed_nodes.append(idx)
    return failed_nodes

def create_distributed_formation_objects(nodes):
    distributed_formation_objects = []
    for i, node in enumerate(nodes):
        other_nodes = [n for j, n in enumerate(nodes) if j != i]
        distributed_formation_objects.append(DistributedFormation(node, other_nodes))
    return distributed_formation_objects

# Update the plot functions with the new settings
def plot_agents_positions(formation_objects, ax):
    ax.clear()
    for i, formation_obj in enumerate(formation_objects):
        for j, agent in enumerate(formation_obj.agents):
            x, y = agent
            ax.scatter(x, y, label=f'Agent {j}' if i == 0 else '', color=colors[j])

            # Add node label next to the agent
            # ax.text(x + 0.1, y + 0.1, f'Node {i}', fontsize=12, color='black')
            # Plot goal positions
            goal_x, goal_y = formation_obj.goals[j]
            ax.scatter(goal_x, goal_y, marker='x', s=82, color='red', label=f'Goal {j}' if i == 0 else '')

        # Display node status
        node_status = f"Node {i} ({formation_obj.status})"

        ax.text(-0.75, 0 - i * 0.15, node_status, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.set_xlabel('x-axis (decimeter)')
    ax.set_ylabel('y-axis (decimeter)')
    ax.set_title('Agents Positions')
    ax.legend()
    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    

def run_simulation( frame_number):
    print(frame_number)
    global distributed_formation_objects, N_AGENTS
    failed_nodes = detect_failed_nodes(distributed_formation_objects)

    if frame_number == 57:
        new_node_address = f"localhost:{4321 + len(distributed_formation_objects)}"
        other_nodes = [n for j, n in enumerate(nodes) if j != len(distributed_formation_objects)]

        
        new_node = DistributedFormation(new_node_address, other_nodes)
        distributed_formation_objects.append(new_node)

        new_agent_position = (random.uniform(-1, 1), random.uniform(-.2, .5))
        # Also, update other_nodes for all existing nodes
        for formation_obj in distributed_formation_objects:
            formation_obj.add_node(new_node_address)
            formation_obj.add_agent(new_agent_position)
            
        N_AGENTS = N_AGENTS +1
        

    # Simulate node failure and recovery
    if frame_number == 10:
        distributed_formation_objects[1].fail_node(frame_number)
    elif frame_number == 20:
        distributed_formation_objects[1].recover_node(frame_number)

    # Periodically check for leader election
    if frame_number % 3 == 0:
        for idx, formation_obj in enumerate(distributed_formation_objects):
            if formation_obj.failed:
                formation_obj.failed_action(frame_number)
                continue

            # Check if the node needs to start an election
            if time.time() - formation_obj.last_heartbeat_received > formation_obj._reset_election_timer():
                # Increment term and start election
                formation_obj.term += 1

                # Change the status to 'candidate' when starting an election
                formation_obj.status = 'candidate'

                print(f"Frame {frame_number}: Node {idx} started an election (term {formation_obj.term})")
                log_table.append({"type": "candidate", "node": idx, "term": formation_obj.term, "frame": frame_number})

                votes = 1  # vote for self

                # Request votes from other nodes
                for j, other_formation_obj in enumerate(distributed_formation_objects):
                    if j != idx and not other_formation_obj.failed:
                        # If the other node's term is less or equal, it grants its vote
                        if other_formation_obj.term <= formation_obj.term:
                            other_formation_obj.term = formation_obj.term

                            # Update the status of other nodes to 'follower' when they grant their vote
                            other_formation_obj.status = 'follower'

                            votes += 1

                # Check if the node has received the majority of votes
                if votes > len(distributed_formation_objects) // 2:
                    print(f"Frame {frame_number}: Node {idx} becomes the leader (term {formation_obj.term})")
                    log_table.append({"type": "leader", "node": idx, "term": formation_obj.term, "frame": frame_number})
                    # Change the status to 'leader' when the node becomes the leader
                    formation_obj.status = 'leader'
                    formation_obj.leader_id = idx
                    formation_obj.leader_id = idx
                    formation_obj.leader_history.append({
                        "type": "leader",
                        "leader": idx,
                        "time": frame_number
                    })
                    # Update leader goals and broadcast to followers
                    new_goals = formation_obj.goals[:]
                    formation_obj.update_goals(new_goals)
                    for other_formation_obj in distributed_formation_objects:
                        if other_formation_obj.status == 'follower':
                            other_formation_obj.update_goals(new_goals)

                    # Send heartbeat to all followers
                    for other_formation_obj in distributed_formation_objects:
                        other_formation_obj.last_heartbeat_received = time.time()
                # Reset the election timer and election start time
                formation_obj._reset_election_timer()
                formation_obj.election_start_time = time.time() + formation_obj._reset_election_timer()
                
    # Move the agents based on their current status
    for formation_obj in distributed_formation_objects:
        if formation_obj.status == 'leader':
            formation_obj.move_leader()
            # Broadcast the updated positions to followers
            new_positions = formation_obj.agents[:]
            for other_formation_obj in distributed_formation_objects:
                if other_formation_obj.status == 'follower':
                    other_formation_obj.update_agents_positions(new_positions)
        elif formation_obj.status == 'follower':
            formation_obj.move_follower()

    # Periodically send heartbeats from the leader to followers
    # if frame_number % 5 == 0 and ( frame_number > 50 and frame_number < 100):
    if frame_number % 15 == 0:
        leader_node = distributed_formation_objects[distributed_formation_objects[0].leader_id]
        for other_formation_obj in distributed_formation_objects:
            if other_formation_obj.status == 'follower':
                other_formation_obj.last_heartbeat_received = time.time()

        if leader_node._isReady(frame_number):
            leader_node.leader_update()

    # for idx, formation_obj in enumerate(distributed_formation_objects):
    #     if formation_obj.status == 'leader' and (not formation_obj.leader_history or formation_obj.leader_history[-1]['leader'] != idx):
    #         formation_obj.leader_history.append({
    #             "type": "leader",
    #             "leader": idx,
    #             "time": frame_number
    #         })

    plot_agents_positions(distributed_formation_objects, ax)

    for formation_obj in distributed_formation_objects:
        formation_obj.agent_history.append(formation_obj.agents[:])
        formation_obj.error_history.append(position_error(formation_obj))

    if frame_number == 0:
        for formation_obj in distributed_formation_objects:
            formation_obj.first_frame = False

    # Mark failed nodes with a red 'X' symbol
    for idx in failed_nodes:
        for agent_pos in distributed_formation_objects[idx].agents:
            ax.plot(agent_pos[0], agent_pos[1], marker='x', color='red', markersize=10)
            ax.text(agent_pos[0] + 0.2, agent_pos[1] + 0.2 * idx, f"Node {idx} (failed)", bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

distributed_formation_objects = create_distributed_formation_objects(nodes)
animation = FuncAnimation(fig, run_simulation, frames=122, interval=100, blit=False, repeat=False)
plt.show()

def plot_agent_trajectory(distributed_formation_objects):
    colors = ['purple', 'blue', 'red', 'orange', 'yellow', 'black', 'pink', 'green', 'white', 'gray']  # Different colors for different agents
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']  # Different line styles for different agents
    start_marker = '^'  # Triangle marker for start positions
    end_marker = 'v'  # Inverted triangle marker for end positions

    plt.figure()
    for i, formation_obj in enumerate(distributed_formation_objects):
        x_positions = [position[i][0] for position in formation_obj.agent_history]
        y_positions = [position[i][1] for position in formation_obj.agent_history]

        plt.plot(x_positions, y_positions, label=f"Agent {i}", color=colors[i], linestyle=linestyles[i])

        # Add markers for start and end positions
        plt.scatter(x_positions[0], y_positions[0], marker=start_marker, color=colors[i], s=100, label=f"Agent {i} Start")
        plt.scatter(x_positions[-1], y_positions[-1], marker=end_marker, color=colors[i], s=100, label=f"Agent {i} End")

    plt.xlabel(r"x Position (decimeter)")
    plt.ylabel(r"y Position (decimeter)")
    plt.title(r"Agent Trajectories from Start to End", y = 1.525)
    # plt.legend()
    plt.legend(loc='upper center', 
              bbox_to_anchor=(0.5, 1.575), 
              ncol=4,  # Change this value to control the number of columns
              fontsize=10,
              columnspacing=0.5,  # Adjust column spacing
              handletextpad=0.5)   # Adjust space between handle and text
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.show()

def plot_agent_history(distributed_formation_objects):
    time_axis = list(range(len(distributed_formation_objects[0].agent_history)))
    colors = ['purple', 'blue', 'red', 'orange', 'yellow', 'black', 'pink', 'green', 'white', 'gray']  # Different colors for different agents

    plt.figure()

    for i, formation_obj in enumerate(distributed_formation_objects):
        x_positions = [position[i][0] for position in formation_obj.agent_history]
        y_positions = [position[i][1] for position in formation_obj.agent_history]

        # Create a custom time_axis for shorter x_positions and y_positions
        custom_time_axis = time_axis[len(time_axis)-len(x_positions):]

        plt.plot(custom_time_axis, x_positions, label=f"Agent {i} $x$-axis", color=colors[i], linestyle='-')
        plt.plot(custom_time_axis, y_positions, label=f"Agent {i} $y$-axis", color=colors[i], linestyle='--')


        labels_added = set()
        for event in formation_obj.leader_history:
            if event["type"] == "leader":
                color = colors[event["leader"]]
                label = f'Leader: Agent {event["leader"]}'
                marker = 'o'  # Circle marker for leader
            elif event["type"] == "follower":
                color = colors[event["leader"]]
                label = f'Follower: Agent {event["leader"]}'
                marker = 'x'  # Cross marker for follower

            if label not in labels_added:
                plt.plot(event["time"], x_positions[event["time"]], marker=marker, color=color, markersize=8, label=label)
                plt.plot(event["time"], y_positions[event["time"]], marker=marker, color=color, markersize=8)
                labels_added.add(label)
            else:
                plt.plot(event["time"], x_positions[event["time"]], marker=marker, color=color, markersize=8)
                plt.plot(event["time"], y_positions[event["time"]], marker=marker, color=color, markersize=8)

    plt.xlabel(r"Time (deciseconds)")
    plt.ylabel(r"Position (decimeter)")
    plt.title("Agents Position Components Over Time (Leader Election)")
    # plt.legend()
    plt.legend(loc='upper left', 
            bbox_to_anchor=(0.65, 1), 
            ncol=1,  # Change this value to control the number of columns
            fontsize=10.5,
            columnspacing=0.5,  # Adjust column spacing
            handletextpad=0.5)   # Adjust space between handle and text

    plt.grid(True)
    plt.show()

def plot_error_history(distributed_formation_objects):
    time_axis = list(range(len(distributed_formation_objects[0].error_history)))
    colors = ['purple', 'blue', 'red', 'orange', 'yellow', 'black', 'pink', 'green', 'white', 'gray']  # Different colors for different agents

    plt.figure()

    for i, formation_obj in enumerate(distributed_formation_objects):
        errors = [error[i] for error in formation_obj.error_history]
        # Create a custom time_axis for shorter x_positions and y_positions
        custom_time_axis = time_axis[len(time_axis)-len(errors):]


        plt.plot(custom_time_axis, errors, label=f"Agent {i} Error", color=colors[i])

        labels_added = set()
        for event in formation_obj.leader_history:
            if event["type"] == "leader":
                color = colors[event["leader"]]
                label = f'Leader: Agent {event["leader"]}'
                marker = 'o'  # Circle marker for leader
            elif event["type"] == "follower":
                color = colors[event["leader"]]
                label = f'Follower: Agent {event["leader"]}'
                marker = 'x'  # Cross marker for follower

            if label not in labels_added:
                plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8, label=label)
                labels_added.add(label)
            else:
                plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8)

    plt.xlabel("Time (deciseconds)")
    plt.ylabel("Position Error (decimeter)")
    plt.title("Agents Position Error Over Time")
    # plt.legend()
    plt.legend(loc='upper left', 
            bbox_to_anchor=(0.55, 1), 
            ncol=1,  # Change this value to control the number of columns
            fontsize=14,
            columnspacing=0.5,  # Adjust column spacing
            handletextpad=0.5)   # Adjust space between handle and text

    plt.grid(True)

    plt.show()


# Call the new function after the existing functions
plot_agent_trajectory(distributed_formation_objects)
# # Plot agent trajectories
plot_agent_history(distributed_formation_objects)
# # Plot position error history
plot_error_history(distributed_formation_objects)


# After the simulation is finished
latex_table = tabulate(log_table, headers="keys", tablefmt="latex_booktabs")
print(latex_table)
with open("C:\\Users\\abbast\\OneDrive - Universitetet i Oslo\Hobbies\\Paper I\\Conference-LaTeX-template_10-17-19\\Imgs_f_a=n_v9\\log_table.csv", "w", newline="") as csvfile:
    fieldnames = ["type", "node", "term", "frame"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in log_table:
        writer.writerow(row)

