
import math
import time
import random
import matplotlib
from pysyncobj import SyncObj, SyncObjConf, replicated
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from collections import defaultdict
import seaborn as sns

# Set style, context, and font scale for the plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.75)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
# plt.rcParams['text.usetex'] = True
# Set color palette
colors = sns.color_palette("tab10")

class DistributedFormation(SyncObj):
    def __init__(self, selfNodeAddr, otherNodeAddrs):
        conf = SyncObjConf(autoTick=True)
        super(DistributedFormation, self).__init__(selfNodeAddr, otherNodeAddrs, conf)
        self.leader_id = 0
        self.failed = False
        self.agent_history = []
        self.leader_history = []
        self.first_frame = True  # Add this line
        self.agents = [
            (random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(3)
        ]  # initial agent positions
        self.goals = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]   # goal positions for agents
        self.set_goals()
        self.leader_change_counter = 0
        self.leader_offset = 0  # Add this line
        self.error_history = []

    def set_goals(self):
        n_agents = len(self.agents)
        polygon_radius = 1.0
        angle = 2 * math.pi / n_agents

        for i in range(n_agents):
            self.goals[i] = (
                polygon_radius * math.cos(i * angle + angle / 2),
                polygon_radius * math.sin(i * angle + angle / 2),
            )

    @replicated
    def update_agent(self, agent_id, new_position):
        self.agents[agent_id] = new_position

    @replicated
    def update_all_agents(self, new_positions):
        self.agents = new_positions

def position_error(formation_obj):
    errors = [
        math.sqrt((px - gx)**2 + (py - gy)**2)
        for (px, py), (gx, gy) in zip(formation_obj.agents, formation_obj.goals)
    ]
    return errors

def single_integrator_dynamics(formation_obj, agent_id, goal, dt=0.1, k=1):
    current_position = formation_obj.agents[agent_id]
    new_position = tuple(
        c + k * (g - c) * dt for c, g in zip(current_position, goal)
    )
    return new_position

def leader_update(formation_obj):
    new_positions = [
        single_integrator_dynamics(formation_obj, i, formation_obj.goals[i])
        for i in range(len(formation_obj.agents))
    ]
    formation_obj.update_all_agents(new_positions)

def plot_agents_positions(formation_objects, ax):
    ax.clear()
    for i, formation_obj in enumerate(formation_objects):
        x, y = formation_obj.agents[i]
        ax.scatter(x, y, label=f'Agent {i}')

        # Plot goal positions
        goal_x, goal_y = formation_obj.goals[i]
        ax.scatter(goal_x, goal_y, marker='x', s = 82, color='red', label=f'Goal {i}')

    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_title('Agents Positions')
    ax.legend()
    ax.grid(True)
    
def plot_agent_history(distributed_formation_objects):
    time_axis = list(range(len(distributed_formation_objects[0].agent_history)))
    colors = ['purple', 'blue', 'red', 'orange', 'yellow', 'black', 'pink', 'green', 'white', 'gray']  # Different colors for different agents

    plt.figure()

    legend_lines = {}
    legend_labels = []

    for i, formation_obj in enumerate(distributed_formation_objects):
        x_positions = [position[i][0] for position in formation_obj.agent_history]
        y_positions = [position[i][1] for position in formation_obj.agent_history]

        # Create a custom time_axis for shorter x_positions and y_positions
        custom_time_axis = time_axis[len(time_axis)-len(x_positions):]

        x_line, = plt.plot(custom_time_axis, x_positions, label=f"Agent {i} $x$-axis", color=colors[i], linestyle='-')
        y_line, = plt.plot(custom_time_axis, y_positions, label=f"Agent {i} $y$-axis", color=colors[i], linestyle='--')

        legend_lines[f"Agent {i} $x$-axis"] = x_line
        legend_lines[f"Agent {i} $y$-axis"] = y_line

        # Add leader and follower markers
        for event in formation_obj.leader_history:
            if event["type"] == "leader":
                if event["leader"] == i:  # Check if the agent is the leader at this time
                    color = colors[event["leader"]]
                    label = f'Leader: Agent {event["leader"]}'
                    marker = 'o'  # Circle marker for leader

                    plt.plot(event["time"], x_positions[event["time"]], marker=marker, color=color, markersize=8)
                    plt.plot(event["time"], y_positions[event["time"]], marker=marker, color=color, markersize=8)

                    if label not in legend_labels:
                        legend_labels.append(label)
                        legend_lines[label] = plt.Line2D([0], [0], marker=marker, color=color, linestyle='None', markersize=8)

            elif event["type"] == "follower":
                color = colors[event["leader"]]
                label = f'Follower: Agent {event["leader"]}'
                marker = 'x'  # Cross marker for follower

                plt.plot(event["time"], x_positions[event["time"]], marker=marker, color=color, markersize=8)
                plt.plot(event["time"], y_positions[event["time"]], marker=marker, color=color, markersize=8)

                if label not in legend_labels:
                    legend_labels.append(label)
                    legend_lines[label] = plt.Line2D([0], [0], marker=marker, color=color, linestyle='None', markersize=8)

            elif event["type"] == "failure":  # Handle failure events
                if event["leader"] == i:
                    color = colors[event["leader"]]
                    label = f'Failed: Agent {event["leader"]}'
                    marker = 'x'  # Cross marker for failed agent

                    plt.plot(event["time"], x_positions[event["time"]], marker=marker, color=color, markersize=8)
                    plt.plot(event["time"], y_positions[event["time"]], marker=marker, color=color, markersize=8)

                    if label not in legend_labels:
                        legend_labels.append(label)
                        legend_lines[label] = plt.Line2D([0], [0], marker=marker, color=color, linestyle='None', markersize=8)

    plt.xlabel(r"Time (deciseconds)")
    plt.ylabel(r"Position (decimeter)")
    plt.title(r"Agents Position Components Over Time", y = 1.01)

    lines = list(legend_lines.values())
    labels = list(legend_lines.keys())

    plt.legend(handles=lines, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.05),
               ncol= 3,
               fontsize=10,
               columnspacing=0.5,  # Adjust column spacing
               handletextpad=0.5)   # Adjust space between handle and text

    plt.grid(True)
    # plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.show()
          
nodes = ['localhost:4321', 'localhost:4322', 'localhost:4323']

distributed_formation_objects = []
for i, node in enumerate(nodes):
    other_nodes = [n for j, n in enumerate(nodes) if j != i]
    distributed_formation_objects.append(DistributedFormation(node, other_nodes))

####
fig, ax = plt.subplots()
start_time = time.time()

def update_frame(frame_number):
    leader_index = (frame_number // 20 + distributed_formation_objects[0].leader_offset) % len(distributed_formation_objects)
    leader_node = distributed_formation_objects[leader_index]

    for formation_obj in distributed_formation_objects:
        formation_obj.leader_id = leader_index

    if frame_number % 20 == 0 and not (leader_node.first_frame and frame_number == 0):
        print(f"Frame {frame_number}: Leader changed to Agent {leader_index}")
        leader_node.leader_history.append({"type": "leader", "leader": leader_index, "time": frame_number})


    # Simulate leader failure at frame 50 and log the event
    if frame_number == 70:
        leader_node.failed = True
        print(f"Frame {frame_number}: Leader (Agent {leader_index}) failed")
        for formation_obj in distributed_formation_objects:
            formation_obj.leader_history.append({"type": "failure", "leader": leader_index, "time": frame_number})

    # Elect a new leader if the current leader has failed
    if leader_node.failed:
        leader_index = (leader_index + 1) % len(distributed_formation_objects)
        leader_node = distributed_formation_objects[leader_index]

        # Log new leader election
        if frame_number == 71:
            print(f"Frame {frame_number}: New leader elected (Agent {leader_index})")
            for formation_obj in distributed_formation_objects:
                formation_obj.leader_history.append({"type": "leader", "leader": leader_index, "time": frame_number})
                formation_obj.leader_offset = 1  # Update the leader_offset


    if leader_node._isReady():
        leader_update(leader_node)

    plot_agents_positions(distributed_formation_objects, ax)
    # Record agent positions
    for formation_obj in distributed_formation_objects:
        formation_obj.agent_history.append(formation_obj.agents[:])
        formation_obj.error_history.append(position_error(formation_obj))
    # Add this at the end of the update_frame function
    if frame_number == 0:
        for formation_obj in distributed_formation_objects:
            formation_obj.first_frame = False
            
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
              ncol=3,  # Change this value to control the number of columns
              fontsize=10,
              columnspacing=0.5,  # Adjust column spacing
              handletextpad=0.5)   # Adjust space between handle and text
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.show()

def plot_error_history(distributed_formation_objects):
    time_axis = list(range(len(distributed_formation_objects[0].error_history)))
    colors = ['purple', 'blue', 'red', 'orange', 'yellow', 'black', 'pink', 'green', 'white', 'gray']  # Different colors for different agents

    plt.figure()

    seen_labels = set()

    for i, formation_obj in enumerate(distributed_formation_objects):
        errors = [error[i] for error in formation_obj.error_history]
        custom_time_axis = time_axis[len(time_axis)-len(errors):]

        plt.plot(custom_time_axis, errors, label=f"Agent {i} Error", color=colors[i])

        # # Add leader and follower markers
        # for event in formation_obj.leader_history:
        #     if event["type"] == "leader":
        #         if event["leader"] == i:  # Check if the agent is the leader at this time
        #             color = colors[event["leader"]]
        #             label = f'Leader: Agent {event["leader"]}'
        #             marker = 'o'  # Circle marker for leader

        #     elif event["type"] == "follower":
        #         color = colors[event["leader"]]
        #         label = f'Follower: Agent {event["leader"]}'
        #         marker = 'x'

        #     if label not in seen_labels:
        #         plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8, label=label)
        #         seen_labels.add(label)
        #     else:
        #         plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8)

        # Add leader and follower markers
        for event in formation_obj.leader_history:
            if event["type"] == "leader":
                if event["leader"] == i:  # Check if the agent is the leader at this time
                    color = colors[event["leader"]]
                    label = f'Leader: Agent {event["leader"]}'
                    marker = 'o'  # Circle marker for leader
                    if label not in seen_labels:
                        plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8, label=label)
                        seen_labels.add(label)
                    else:
                        plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8)


            elif event["type"] == "follower":
                color = colors[event["leader"]]
                label = f'Follower: Agent {event["leader"]}'
                marker = 'x'  # Cross marker for follower
                if label not in seen_labels:
                    plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8, label=label)
                    seen_labels.add(label)
                else:
                    plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8)

            elif event["type"] == "failure":  # Handle failure events
                if event["leader"] == i:
                    color = colors[event["leader"]]
                    label = f'Failed: Agent {event["leader"]}'
                    marker = 'x'  # Cross marker for failed agent
                    if label not in seen_labels:
                        plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8, label=label)
                        seen_labels.add(label)
                    else:
                        plt.plot(event["time"], errors[event["time"]], marker=marker, color=color, markersize=8)




    plt.xlabel("Time (deciseconds)")
    plt.ylabel("Position Error (decimeter)")
    plt.title("Agents Position Error Over Time")
    plt.legend(loc='upper left', 
            bbox_to_anchor=(0.55, 1), 
            ncol=1,  # Change this value to control the number of columns
            fontsize=14,
            columnspacing=0.5,  # Adjust column spacing
            handletextpad=0.5)   # Adjust space between handle and text

    plt.grid(True)

    plt.show()

animation = FuncAnimation(fig, update_frame, frames=122, interval=1, blit=False, repeat=False)
plt.show()

# Call the new function after the existing functions
plot_agent_trajectory(distributed_formation_objects)

# Plot agent trajectories
plot_agent_history(distributed_formation_objects)

# # Plot position error history
plot_error_history(distributed_formation_objects)