import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat =3

        self.state_size = 27 #27 or 18
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        # variable to calculate reward and monitor
        self.last_z=0
        self.last_vel=np.array((0,0,0))
        self.step_taken = 0
        self.last_rotor_speeds=np.array((0,0,0,0))
        self.rotor_speed_change=np.array((0,0,0,0))

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.,0.,0.,0.]) 

    def get_reward(self,rotor_speeds):
        """Uses current pose of sim to return reward."""
        # reward for for the takeoff
        desplacement=self.sim.pose[:3]-self.target_pos[:3]
        dis=abs(desplacement)
        if (self.sim.v<=desplacement).all():
            vel=self.sim.v-desplacement
        else:
            vel=desplacement-self.sim.v
        angle=self.sim.pose[3:]-self.target_pos[3:]
        
        z= self.sim.pose[2]-self.last_z
        alpha=np.array((0.003,0.003,0.009))
        
#         reward= 1- np.tanh(dis*alpha).sum()+np.tanh(max(z,0))   # max|min. 1|0

#         reward=1- np.tanh(dis*alpha).sum() + np.tanh(vel*alpha).sum()+ np.tanh(z)   # 1|-1

#         reward=1- np.tanh(dis*0.003).sum() + np.tanh(vel*0.003).sum() - np.tanh(angle*0.003).sum()   # 1|-2

        reward=1- np.tanh(dis*alpha).sum()+np.tanh(max(self.sim.v[2]*0.009,0))+np.tanh(max(z,0))   # max|min. 1|0  .. for the situatuions of takeoff
        self.last_z=self.sim.pose[2]
        return reward
    
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            pose_all.append(self.sim.pose)
            pose_all.append(self.sim.v)
        next_state = np.concatenate(pose_all)

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
#         state = np.concatenate([self.sim.pose]* self.action_repeat) 
        state= np.concatenate(([self.sim.pose]+[self.sim.v])*self.action_repeat)   # with velocity
        self.step_taken=0
        return state