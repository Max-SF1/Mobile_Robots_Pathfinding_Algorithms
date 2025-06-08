import matplotlib.pyplot as plt
from car_simulator import State, States, Simulator
from utils import Trajectory
import car_consts
import numpy as np


class PID(object):
    def __init__(self, current_time, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.current_time = current_time
        self.acc_err = 0
        self.error = -1.5

    def pid_step(self, ref, current, clock):
        dt = clock - self.current_time
        self.current_time = clock
        prev_err = self.error
        self.error = ref - current
        self.acc_err += self.error * dt
        steering_command = (self.kp*self.error + self.ki*self.acc_err + self.kd*(self.error-prev_err)/dt)
        return steering_command


class WALL_FOLLOWING(object):
    def __init__(self,  localization_error):
        self.theta = 70
        self.theta_rad = self.theta * np.pi / 180
        self.L = 1.0
        self.localiztion_error = localization_error

    def get_predicted_dist2wall(self, state: State):
        x_err, y_err = np.random.rand(2) * self.localiztion_error 
        xr=0
        yr=-state.x/np.tan(state.yaw) + state.y
        right_beam = ((xr- (state.x + x_err))**2 + (yr-(state.y + y_err) )**2)**0.5
        xo=0
        yo= -state.x / np.tan(state.yaw+self.theta_rad) + state.y
        offset_beam = ((xo- (state.x + x_err))**2 + (yo-(state.y + y_err))**2)**0.5
        alpha = np.arctan((offset_beam*np.cos(self.theta_rad) - right_beam) / (offset_beam*np.sin(self.theta_rad)))
        dist2wall = right_beam * np.cos(alpha)
        predicted_dist2wall = dist2wall + self.L * np.sin(alpha)
        return predicted_dist2wall
        
def plot_error(errors, times, k_p, k_i, k_d):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(times, errors)
    ax.grid()
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Error (m)')
    ax.set_title("Error graph for Kp = " + str(k_p) + " Kd = " + str(k_d) + " Ki = " + str(k_i))
    print(f'Average_error: {sum(abs(np.array(errors)))/len(errors)}')
    plt.show()      



def main():
    pid = PID(current_time=0, kp=0.5, kd=0.25, ki=0.75)
    STEERING_ERROR_PRECENTAGE = 0.3
    STEERING_ERROR_CONST = 0.0
    LOCALIZATION_ERROR_METER = 0.5

    # ----- Do not change the code below ----------------
    dt = 0.1  # [s] time tick
    SIMULATION_TIME = 15.0  # max simulation time
    MAX_STEER = car_consts.max_steering_angle_rad  # maximum steering angle [rad]
    keep_distance_from_wall = 1.0
    line_coords = []
    for i in range(15):
        line_coords.append([0, i])
    trajectory = Trajectory(dl=0.1, path =np.array(line_coords), TARGET_SPEED=1.0)
    wall_following = WALL_FOLLOWING( LOCALIZATION_ERROR_METER)
    state = State(x=-2.5 , y=0, yaw=np.pi/2-np.pi/4, v=1.0)
    clock = 0.0
    states = States()
    states.append(clock, state)
    simulator = Simulator(trajectory=trajectory, DT=dt)
    errors = []
    times = []
    while SIMULATION_TIME >= clock:
        clock += dt
        predicted_dist2wall = wall_following.get_predicted_dist2wall(state)
        delta = pid.pid_step(ref=keep_distance_from_wall , current=predicted_dist2wall, clock=clock)
        times.append(clock)
        errors.append(state.x + keep_distance_from_wall)
        state.predelta = delta
        state = simulator.update_state(state, delta + np.random.rand(1)*MAX_STEER*STEERING_ERROR_PRECENTAGE + STEERING_ERROR_CONST) 
        states.append(clock, state, delta)
    simulator.show_simulation(states)
    plot_error(errors, times, pid.kp, pid.ki, pid.kd)
if __name__ == '__main__':
    main()
