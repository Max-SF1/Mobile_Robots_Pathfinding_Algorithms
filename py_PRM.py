import numpy as np
import matplotlib.pyplot as plt
from py_Utils import  CSpace
import heapq

import time
plt.ion()

class PRM(object):
    def __init__(self, env_map,  max_itr=1000,  dist=10):
        self.max_itr = max_itr
        self.map = env_map 
        self.env_rows, self.env_cols = env_map.shape #rows~y, cols~x
        self.max_dist=dist #[pixels]
        self.build_prm_graph()


    def build_prm_graph(self):
        # sampling
        configs = []
        self.graph = {} # `[(neighbor, cost), ...]`
        for i in range(self.max_itr):
            config = self.sample()
            if not self.is_in_collision(config):
                self.graph[config] = []
        vertices = list(self.graph.keys())
        for i, vertex in enumerate(vertices):
            neighbors = self.find_neighbors_in_range(vertex)
            self.graph[vertex] = neighbors      

    def find_neighbors_in_range(self, vertex1):
        distances = [] # tuples of [(nei, dist)]
        for vertex2 in self.graph.keys():
            if vertex1 != vertex2: 
                dist = self.get_cost(vertex1,vertex2)
                if dist < self.max_dist and self.local_planner(vertex1,vertex2,dist): 
                    distances.append((vertex2,dist))
                    
        return distances # tuples of [(nei, dist)]
    
    def get_cost(self, vertex1, vertex2):
        return np.sqrt((vertex1[0]-vertex2[0])**2+ (vertex1[1]-vertex2[1])**2)
    
    def sample(self):
        x = np.random.randint(0, self.env_cols)
        y = np.random.randint(0, self.env_rows)
        # print(x,y)
        return (x, y) # cols~x, rows~y
    

    def is_in_collision(self, config):
        if config[0] < 0 or config[0] >= self.env_cols or config[1] < 0 or config[1] >= self.env_rows:
            return True    
        if (self.map[config[1],config[0]] != 0 ):
            return True 
        return False

    
    def local_planner(self, config1, config2, dist): #can two vertices be connected
        num_points = int(dist) + 1 
        if (self.get_cost(config1,config2) > dist): 
            return False 
        for i in range(num_points + 1): 
            m = i / num_points
            x = int(config1[0] + m * (config2[0] - config1[0]))
            y = int(config1[1] + m * (config2[1] - config1[1]))           
            # Check if intermediate point is in collision
            if self.is_in_collision((x, y)):
                return False
        return True
            

class A_Star():
    def __init__(self, prm: PRM):
        self.prm = prm

    def h(self, current, goal):
        #euclidean distance as hueristic 
        return np.sqrt((current[0]-goal[0])**2+ (current[1]-goal[1])**2)


    def find_path(self, start, goal):
        start = (start[0], start[1])
        goal = (goal[0], goal[1])
        self.prm.graph[start] = self.prm.find_neighbors_in_range(start)
        for nei, dist in self.prm.graph[start]:
            self.prm.graph[nei].append((start, dist))
        self.prm.graph[goal] = self.prm.find_neighbors_in_range(goal)
        #TODO
        for nei, dist in self.prm.graph[goal]:
            self.prm.graph[nei].append((goal, dist))
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.h(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = self.reconstruct_path(current, came_from, start)
                cost = g_score[goal]
                return path, cost
            
            # Explore neighbors
            if current in self.prm.graph:
                for neighbor, edge_cost in self.prm.graph[current]:
                    tentative_g_score = g_score[current] + edge_cost
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.h(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None, float('inf') # No path found


    def reconstruct_path(self, current, came_from, start):
        path = [current]
        while current is not start:
            current = came_from[current]
            path.append(current)
        path.append(start)
        path.reverse()
        return path

        

class Plotter():
    def __init__(self, inflated_map): 
        self.env_rows, self.env_cols = inflated_map.shape
        self.map = inflated_map
    
    def draw_graph(self, graph, start, goal,path=None):
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        plt.xlim([0, self.env_cols])
        plt.ylim([0, self.env_rows])
        for vertex in graph.keys():
            plt.scatter(vertex[0], vertex[1], s=20, c='g')
        for vertex in graph.keys():
            for nei, dist in graph[vertex]:
                plt.plot([vertex[0],nei[0]], [vertex[1], nei[1]], color='b', linewidth=1)
        if path is not None:
            for i in range(len(path)-1):
                plt.plot([path[i][0],path[i+1][0]], [path[i][1],path[i+1][1]], color='r', linewidth=3)
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

def remove_consecutive_duplicates(path):
    cleaned = [path[0]]
    for p in path[1:]:
        if not np.allclose(p, cleaned[-1]):
            cleaned.append(p)
    return cleaned

def main():
    start_time = time.time()
    map_original = np.array(np.load('maze_test.npy'), dtype=int)
    resolution = 0.05000000074505806
    robot_radius = 0.15
    converter = CSpace(resolution, origin_x=-4.73, origin_y=-5.66, map_shape=map_original.shape)
    inflated_map = inflate(map_original, robot_radius / resolution)

    # Convert start and goal
    start = converter.meter2pixel([0.0, 0.0])
    goal = converter.meter2pixel([-2, 0])
    print(start)
    print(goal)

    # Path planning
    prm = PRM(env_map=inflated_map, max_itr=2000, dist=30)
    astar = A_Star(prm)
    path, cost = astar.find_path(start, goal)
    print(f'path cost: {cost}, time: {time.time() - start_time:.2f}s')

    # Ensure yaw = 0 for each point
    path_index = [list(p) + [0] if len(p) == 2 else list(p) for p in path]

    # Convert to meters
    path_meter = converter.pathindex2pathmeter(path_index)

    # Remove duplicate points to avoid spline errors
    path_meter_clean = remove_consecutive_duplicates(path_meter)

    # Save clean path
    np.save("path_in_meters.npy", np.array(path_meter_clean))


if __name__ == "__main__":
    main()


