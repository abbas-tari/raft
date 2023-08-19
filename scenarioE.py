
import math
import time
import random
import matplotlib
from pysyncobj import SyncObj, SyncObjConf, replicated
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from collections import defaultdict
# matplotlib.rcParams['text.usetex'] = True
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
        self.first_frame = True  # Add this line
        self.agents = [
            (random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(N_AGENTS)
        ]  # initial agent positions
        self.goals = [(0.0, 0.0) for _ in range(N_AGENTS)]  # goal positions for agents
        
        self.set_goals()
        self.leader_history = []
        self.error_history = []
        self.term = 0
        self.last_heartbeat_received = time.time()
        self.status = 'follower'
        
    def _reset_election_timer(self):
        election_timeout = 1.0
        return random.uniform(election_timeout, 2 * election_timeout)

    def _isReady(self):
        return not self.first_frame and (time.time() - self.last_heartbeat_received < self._reset_election_timer())

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

    def _onLeaderCommit(self, logIndex, logEntry):
        if logEntry[:1] != '_':
            getattr(self, logEntry[0])(*logEntry[1:])


def single_integrator_dynamics(formation_obj, agent_id, goal, dt=0.1, k=1.0):
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

def get_active_agents(distributed_formation_objects):
    return [i for i, formation_obj in enumerate(distributed_formation_objects) if not formation_obj.failed]

def position_error(formation_obj):
    errors = [
        math.sqrt((px - gx)**2 + (py - gy)**2)
        for (px, py), (gx, gy) in zip(formation_obj.agents, formation_obj.goals)
    ]
    return errors

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
            fontsize=14,
            columnspacing=0.5,  # Adjust column spacing
            handletextpad=0.5)   # Adjust space between handle and text

    plt.grid(True)
    plt.show()

  
N_AGENTS = 3
nodes = [f"localhost:{4321 + i}" for i in range(N_AGENTS)]

distributed_formation_objects = []
for i, node in enumerate(nodes):
    other_nodes = [n for j, n in enumerate(nodes) if j != i]
    distributed_formation_objects.append(DistributedFormation(node, other_nodes))

####
fig, ax = plt.subplots()
start_time = time.time()

def update_frame(frame_number):
    global distributed_formation_objects
    for idx, formation_obj in enumerate(distributed_formation_objects):
        if not formation_obj.failed:
            # Check if the node needs to start an election
            if time.time() - formation_obj.last_heartbeat_received > formation_obj._reset_election_timer():
                # Increment term and start election
                formation_obj.term += 1
                
                # Change the status to 'candidate' when starting an election
                formation_obj.status = 'candidate'

                print(f"Frame {frame_number}: Node {idx} started an election (term {formation_obj.term})")
                votes = 1  # vote for self

                # Request votes from other nodes
                for j, other_formation_obj in enumerate(distributed_formation_objects):
                    if j != idx and not other_formation_obj.failed:
                        # If the other node's term is less or equal, it grants its vote
                        if other_formation_obj.term <= formation_obj.term:
                            other_formation_obj.term = formation_obj.term
                            
                            # Update the status of other nodes to 'follower' when they grant their vote
                            other_formation_obj.status = 'follower'
                            # other_formation_obj.leader_history.append({
                            #     "type": "follower",
                            #     "leader": idx,
                            #     "time": frame_number
                            # })
                                                        
                            votes += 1

                # Check if the node has received the majority of votes
                if votes > len(distributed_formation_objects) // 2:
                    print(f"Frame {frame_number}: Node {idx} becomes the leader (term {formation_obj.term})")
                    
                    # Change the status to 'leader' when the node becomes the leader
                    formation_obj.status = 'leader'
                    formation_obj.leader_id = idx
                    formation_obj.leader_history.append({
                        "type": "leader",
                        "leader": idx,
                        "time": frame_number
                    })

                    # Send heartbeat to all followers
                    for other_formation_obj in distributed_formation_objects:
                        other_formation_obj.last_heartbeat_received = time.time()


    leader_node = distributed_formation_objects[distributed_formation_objects[0].leader_id]

    if leader_node._isReady():
        leader_update(leader_node)


    plot_agents_positions(distributed_formation_objects, ax)
    # Record agent positions and error
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

def verify_formation(formation_objects, failed_agent):
    active_agents = get_active_agents(formation_objects)

    def distance(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    last_positions = [
        formation_objects[i].agents[-1] for i in active_agents if i != failed_agent
    ]
    n = len(last_positions)
    distances = [
        distance(last_positions[i], last_positions[(i + 1) % n])
        for i in range(n)
    ]

    # Check if all distances are equal (regular polygon)
    regular_polygon = all(
        math.isclose(distances[i], distances[(i + 1) % n], rel_tol=1e-9)
        for i in range(n)
    )

    centroid = (
        sum(pos[0] for pos in last_positions) / n,
        sum(pos[1] for pos in last_positions) / n,
    )

    # Check if centroid is at origin
    centroid_at_origin = math.isclose(
        centroid[0], 0.0, rel_tol=1e-9
    ) and math.isclose(centroid[1], 0.0, rel_tol=1e-9)

    # Check if all agents have the same distance to the origin
    origin = (0, 0)
    origin_distances = [distance(pos, origin) for pos in last_positions]
    equal_distances_to_origin = all(
        math.isclose(origin_distances[i], origin_distances[(i + 1) % n], rel_tol=1e-9)
        for i in range(n)
    )

    return regular_polygon, centroid_at_origin, equal_distances_to_origin

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
            fontsize=15,
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

# Plot position error history
plot_error_history(distributed_formation_objects)

regular_polygon, centroid_at_origin, equal_distances_to_origin = verify_formation(distributed_formation_objects, 1)
print("Regular 3-sided polygon formation:", regular_polygon)
print("Centroid at origin:", equal_distances_to_origin)
