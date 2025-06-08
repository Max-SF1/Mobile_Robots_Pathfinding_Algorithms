import numpy as np
import math
from utils import Trajectory
import car_consts
import matplotlib.pyplot as plt
from car_simulator import State, States, Simulator



class PurePursuit_Controller(object):
    def __init__(self, cx, cy, k=0.1, Lfc=0.5, Kp =1.0, WB =0.335, MAX_ACCEL=1.0, MAX_SPEED=3, MIN_SPEED=-3, MAX_STEER=np.deg2rad(27.0), MAX_DSTEER=np.deg2rad(150.0) ):
        self.k = k  # look forward gain
        self.Lfc = Lfc   # [m] look-ahead distance
        self.Kp = Kp  # speed proportional gain
        self.WB = WB # wheelbase
        self.old_nearest_point_index = None
        self.cx = cx # trajectory x coords
        self.cy = cy # trajectory y coords
        self.MAX_ACCEL = MAX_ACCEL
        self.MAX_SPEED  = MAX_SPEED
        self.MIN_SPEED = MIN_SPEED
        self.MAX_STEER = MAX_STEER
        self.MAX_DSTEER = MAX_DSTEER

    def pure_pursuit_steer_control(self, state: State, trajectory: Trajectory,  dt):
        '''
        input: 
        current state,
        trajectory,
        dt
        Return the target index, look ahead distance, and closest index of path coords
        '''
        # Find target point to aim for
        ind, Lf, closest_index = self.search_target_index(state)

        if ind < len(self.cx):
            tx = self.cx[ind]
            ty = self.cy[ind]
        else:
            # If reached end of trajectory, use last point
            tx = self.cx[-1]
            ty = self.cy[-1]
            ind = len(self.cx) - 1

        # Calculate the heading angle between vehicle and target point
        # Transform target position to vehicle coordinate system
        alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

        # Pure pursuit control law
        if Lf == 0.0:
            return 0.0, ind, closest_index  # Avoid division by zero

        # Calculate steering angle
        delta = math.atan2(2.0 * self.WB * math.sin(alpha), Lf)

        # Apply steering limits
        delta = max(min(delta, self.MAX_STEER), -self.MAX_STEER)

        return delta, ind, closest_index
    
    def proportional_control_acceleration(self, target_speed, current_speed, dt=0.1):
        '''
        returns updated linear speed
        '''
        # Basic proportional control for speed
        accel = self.Kp * (target_speed - current_speed)

        # Apply acceleration limits
        accel = max(min(accel, self.MAX_ACCEL), -self.MAX_ACCEL)

        # Update speed with acceleration and dt
        linear_velocity = current_speed + accel * dt

        # Apply speed limits
        linear_velocity = max(min(linear_velocity, self.MAX_SPEED), self.MIN_SPEED)

        return linear_velocity
        

    def search_target_index(self, state: State):
        '''
        input: current state
        output:
        target index, 
        updated look-ahead distance,
        index of closest coords on trajectory 
        '''
        # Find closest point on trajectory to current position
        if self.old_nearest_point_index is None:
            # If first time, search all points
            distances = [self.calc_distance(state.rear_x, state.rear_y, icx, icy)
                         for icx, icy in zip(self.cx, self.cy)]
            closest_index = distances.index(min(distances))
            self.old_nearest_point_index = closest_index
        else:
            # Search limited points around previous nearest point for efficiency
            closest_index = self.old_nearest_point_index
            distance = self.calc_distance(state.rear_x, state.rear_y,
                                          self.cx[closest_index],
                                          self.cy[closest_index])

            # Search the closest point in a local area
            search_range = 20
            for i in range(1, search_range + 1):
                # Forward search
                forward_index = (closest_index + i) % len(self.cx)
                forward_distance = self.calc_distance(state.rear_x, state.rear_y,
                                                      self.cx[forward_index],
                                                      self.cy[forward_index])
                # Backward search
                backward_index = (closest_index - i) % len(self.cx)
                backward_distance = self.calc_distance(state.rear_x, state.rear_y,
                                                       self.cx[backward_index],
                                                       self.cy[backward_index])

                # Update closest distance and index
                if forward_distance < distance:
                    distance = forward_distance
                    closest_index = forward_index
                if backward_distance < distance:
                    distance = backward_distance
                    closest_index = backward_index

            self.old_nearest_point_index = closest_index

        # Calculate look-ahead distance dynamically based on velocity
        Lf = self.k * state.v + self.Lfc

        # Search look ahead target point index
        target_ind = closest_index
        while Lf > self.calc_distance(state.rear_x, state.rear_y,
                                      self.cx[target_ind], self.cy[target_ind]):
            if target_ind + 1 >= len(self.cx):
                break  # reached end of trajectory
            target_ind += 1

        return target_ind, Lf, closest_index
    
    def calc_distance(self, rear_x, rear_y, point_x, point_y):
        '''
        calculates the distance between two coords
        '''
        dx = rear_x - point_x
        dy = rear_y - point_y
        return math.sqrt(dx ** 2 + dy ** 2)


def plot_error(closest_path_coords, states:States, trajectory:Trajectory):
    fig = plt.figure()
    ax = fig.add_subplot()
    total_tracking_error = 0
    for i in range(len(states.x)-1):
        tracking_error = ((closest_path_coords[i][0] -states.rear_x[i])**2 + (closest_path_coords[i][1] - states.rear_y[i])**2)**0.5
        total_tracking_error += tracking_error
        ax.scatter(i, tracking_error, c='b')
    print(f'average tracking error {total_tracking_error/(i+1)}')
    ax.set_xlabel('itr')
    ax.set_ylabel('Tracking Error')
    ax.grid()
    plt.show()

def main():
    #  hyper-parameters
    k = 0.1  # look forward gain
    Lfc = 0.75  # [m] look-ahead distance
    Kp = 1.0  # speed proportional gain
    dt = 0.1  # [s] time tick
    target_speed = 1.0  # [m/s]
    T = 100.0  # max simulation time
    WB = car_consts.wheelbase 
    MAX_STEER = car_consts.max_steering_angle_rad  # maximum steering angle [rad]
    MAX_DSTEER = car_consts.max_dt_steering_angle  # maximum steering speed [rad/s]
    MAX_SPEED = car_consts.max_linear_velocity  # maximum speed [m/s]
    MIN_SPEED = car_consts.min_linear_velocity  # minimum speed [m/s]
    MAX_ACCEL = 1.0  # maximum accel [m/ss]

    path = np.load(r'C:\Users\orife\Desktop\hw2- clean\648.3911333811134.npy')
    trajectory = Trajectory(dl=0.1, path =path, TARGET_SPEED=target_speed)
    state = State(x=trajectory.cx[0], y=trajectory.cy[0], yaw=trajectory.cyaw[0], v=0.0)
    lastIndex = len(trajectory.cx) - 1
    clock = 0.0
    states = States()
    states.append(clock, state)
    pp = PurePursuit_Controller(trajectory.cx, trajectory.cy, k, Lfc, Kp, WB, MAX_ACCEL, MAX_SPEED, MIN_SPEED, MAX_STEER, MAX_DSTEER)
    target_ind, _, nearest_index = pp.search_target_index(state)
    simulator = Simulator(trajectory, dt)
    closest_path_coords = []
    while T >= clock and lastIndex > target_ind:
        state.v = pp.proportional_control_acceleration(target_speed, state.v, dt)
        delta, target_ind, closest_index = pp.pure_pursuit_steer_control(state, trajectory, dt)
        state.predelta = delta
        state = simulator.update_state(state, delta)  # Control vehicle
        clock += dt
        states.append(clock, state, delta)
        closest_path_coords.append([trajectory.cx[closest_index], trajectory.cy[closest_index]])
    simulator.show_simulation(states, closest_path_coords)
    plot_error(closest_path_coords, states, trajectory)


if __name__ == '__main__':
    main()
