import numpy as np
import matplotlib.pyplot as plt
from py_Utils import Tree,  CSpace
plt.ion()

class KINORRT(object):
    def __init__(self, env_map, max_step_size = 0.5, max_itr=5000, p_bias = 0.05, converter: CSpace =None ):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.p_bias = p_bias
        self.tree = Tree()
        self.map = env_map
        self.env_rows, self.env_cols = env_map.shape
        self.env_yaw_range = 2*np.pi
        self.converter = converter
        self.ackerman = Odom(converter)
        self.distance = np.inf
        
        
    def is_goal_reached(self, x_new, goal, threshold=9):
        # Check if current state is close enough to goal
        distance = np.sqrt((x_new[0] - goal[0])**2 + (x_new[1] - goal[1])**2)
        self.distance = min(distance, self.distance)
        return distance < threshold
        
    def find_path(self, start, goal):
        goal_reached = False 
        itr = 0
        self.tree.AddVertex(start)
        goal_idx = None
        while itr < self.max_itr and not goal_reached:
            # sample random vertex
            x_random = self.sample(goal)

            # find nearest neighbor
            x_near_idx, x_near = self.tree.GetNearestVertex(x_random)
            delta_time, steering, velocity = self.ackerman.sample_control_command()
            x_new, edge, edge_cost = self.ackerman.propagate(steering, velocity ,delta_time, x_near)
            
            # add vertex and edge
            if self.local_planner(edge):
                x_new_idx = self.tree.AddVertex(x_new)
                self.tree.AddEdge(x_near_idx,x_new_idx,edge_cost)
                self.tree.vertices[x_new_idx].set_waypoints(edge) # ?
                
                #threshold stuff
                if self.is_goal_reached(x_new,goal):
                    goal_reached = True
                    goal_idx = x_new_idx                
            itr += 1
            if itr%1000 ==0:
                print(f'itr: {itr}')
        print("shortest distance achieved = ", self.distance)
        print("iteration the path was found = ", itr)
        if goal_reached: 
            path, path_idx, cost = self.get_shortest_path(goal_idx)
            return path, path_idx, cost
        else:
            return None, None, None 

    def sample(self, goal):
        # if np.random.uniform(0,1) < self.p_bias:
        #     return goal  
        # else: 
        x = np.random.randint(0, self.env_cols)
        y = np.random.randint(0, self.env_rows)
        yaw = np.random.uniform(0, self.env_yaw_range)
        # # print(x,y)
        return (x, y,yaw) # cols~x, rows~y

    
    def is_in_collision(self, x_new):
        if x_new[0] < 0 or x_new[0] >= self.env_cols or x_new[1] < 0 or x_new[1] >= self.env_rows:
            return True    
        if (self.map[x_new[1],x_new[0]] != 0 ):
            return True 
        return False
        

    def local_planner(self, edge):
        for point in edge:
            if self.is_in_collision(point) == True:
                return False
            #check through all the points.
        return True 
    
    def get_shortest_path(self, goal_idx):
        '''
        Returns the path and cost from some vertex to Tree's root
        @param dest - the id of some vertex
        return the shortest path and the cost
        '''
        path = []
        path_idx = []
        current_idx = goal_idx
        total_cost = 0
        
        # Trace back from goal to root
        while current_idx is not None:
            path_idx.append(current_idx)
            vertex = self.tree.vertices[current_idx]
            path.append(vertex.conf)
            total_cost += vertex.cost
            
            # Get parent
            if current_idx in self.tree.edges:
                current_idx = self.tree.edges[current_idx]
            else:
                break
        
        # Reverse to get path from start to goal
        path.reverse()
        path_idx.reverse()
        
        return path, path_idx, total_cost
    
    

class Plotter():
    def __init__(self, inflated_map): 
        self.env_rows, self.env_cols = inflated_map.shape
        self.map = inflated_map
    
    def draw_tree(self, tree:Tree, start, goal, path=None, path_idx = None):
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.xlim([0, self.env_cols])
        plt.ylim([0, self.env_rows])
        if path is not None:
            for idx in path_idx:
                try:
                    vertex = tree.vertices[idx]
                    for waypoint in vertex.waypoints:
                        plt.scatter(waypoint[0], waypoint[1], s=20, c='m')
                except:
                    pass

        for i in range(len(tree.vertices)):
            conf = tree.vertices[i].conf
            plt.scatter(conf[0], conf[1], s=10, c='b')

        
        plt.scatter(start[0], start[1], s=100, c='g')
        plt.scatter(goal[0],goal[1], s=100, c='r')
        plt.imshow(self.map, origin="lower")
        plt.pause(100)
 

    


def inflate(map_, inflation):#, resolution, distance):
    cells_as_obstacle = int(inflation) #int(distance/resolution)
    map_[95:130, 70] = 100
    original_map = map_.copy()
    inflated_map = map_.copy()
    # add berrier
    rows, cols = inflated_map.shape
    for j in range(cols):
        for i in range(rows):
            if original_map[i,j] != 0:
                i_min = max(0, i-cells_as_obstacle)
                i_max = min(rows, i+cells_as_obstacle)
                j_min = max(0, j-cells_as_obstacle)
                j_max = min(cols, j+cells_as_obstacle)
                inflated_map[i_min:i_max, j_min:j_max] = 100
    return inflated_map       
    

class Odom(object):
    def __init__(self, converter:CSpace):
        self.wheelbase = 0.35
        self.max_steering_angle = np.deg2rad(35)
        self.min_velocity, self.max_velocity = 0.5, 1
        self.min_time, self.max_time = 1, 2
        self.converter = converter
    
    def sample_control_command(self):
        delta_time = np.random.uniform(self.min_time,self.max_time)
        steering = np.random.uniform(-self.max_steering_angle,self.max_steering_angle) #can steer in both directions
        velocity = np.random.uniform(self.min_velocity,self.max_velocity)
        return delta_time, steering, velocity
        

    def propagate(self,  steering, velocity ,delta_time, initial_x):
        initial_x = self.converter.pixel2meter(initial_x)
        x = initial_x[0]
        y = initial_x[1]
        theta= initial_x[2]
        theta_dot = velocity * np.tan(steering) / self.wheelbase
        dt = 0.03
        edge = [[x,y,theta]]
        cost = 0
        for _ in range(int(delta_time/dt)):
            theta += theta_dot * dt
            x_dot = velocity * np.cos(theta)
            y_dot = velocity * np.sin(theta)
            x += x_dot * dt
            y += y_dot * dt
            cost += ((edge[-1][0] - x)**2 + (edge[-1][1] - y)**2)**0.5
            edge.append([x,y,theta])
        edge = self.converter.pathmeter2pathindex(edge)
        new_state = edge[-1]
        return new_state, edge, cost

def remove_consecutive_duplicates(path):
    cleaned = [path[0]]
    for p in path[1:]:
        if not np.allclose(p, cleaned[-1]):
            cleaned.append(p)
    return cleaned



def main():
    map_original = np.array(np.load('maze_test.npy'), dtype=int)
    resolution=0.05000000074505806
    inflated_map = inflate(map_original, 0.2/resolution)
    converter = CSpace(resolution, origin_x=-4.73, origin_y=-5.66, map_shape=map_original.shape)
    start=converter.meter2pixel([0.0,0.0])
    goal = converter.meter2pixel([6.22, -4.22])
    print(start)
    print(goal)
    kinorrt_planner = KINORRT(env_map=inflated_map, max_step_size=20, max_itr=10000, p_bias=0.05,converter=converter )
    path, path_idx, cost = kinorrt_planner.find_path(start, goal)
    print(f'cost: {cost}')
    plotter = Plotter(inflated_map=inflated_map)
    plotter.draw_tree(kinorrt_planner.tree, start, goal, path, path_idx)
    
    # Ensure yaw = 0 for each point
    path_index = [list(p) + [0] if len(p) == 2 else list(p) for p in path]

    # Convert to meters
    path_meter = converter.pathindex2pathmeter(path_index)

    # Remove duplicate points to avoid spline errors
    path_meter_clean = remove_consecutive_duplicates(path_meter)

    # Save clean path
    np.save(str(cost)+".npy", np.array(path_meter_clean))


    


if __name__ == "__main__":
    main()


