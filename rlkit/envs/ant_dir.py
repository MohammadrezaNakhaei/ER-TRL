import numpy as np

from rlkit.envs.mujoco.ant_multitask_base import MultitaskAntEnv, AntEnv
from rlkit.envs import register_env


@register_env('ant-dir')
class AntDirEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, forward_backward=False, max_episode_steps=200, randomize_tasks=False, **kwargs):
        self.forward_backward = forward_backward
        self._max_episode_steps = max_episode_steps 
        self.randomize_tasks = randomize_tasks
        super(AntDirEnv, self).__init__(task, n_tasks, max_episode_steps, **kwargs)


    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )
    
    def reset(self):
        self._step = 0
        return super().reset()

    def sample_tasks(self, num_tasks):
        if self.forward_backward:
            assert num_tasks == 2
            directions = np.array([0., np.pi])
        elif self.randomize_tasks:
            directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        else:
            directions = np.linspace(0, 2*np.pi, num_tasks+1)[:-1]
        tasks = [{'goal': direction} for direction in directions]
        return tasks



@register_env('ant-dir-custom')
class AntDirEnvCustom(AntEnv):

    def __init__(self, idx=0, **kwargs):
        self.num_tasks = 40
        self.num_train = 20
        self.num_moderate = 10
        self.num_extreme = 10
        self._goal_idx = idx
        self.tasks = self.sample_tasks()
        self._task = self.tasks[self._goal_idx]
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self._step = 0
        super(AntDirEnvCustom, self).__init__()
        
    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self,):
        directions = np.linspace(0, 2*np.pi, self.num_tasks+1)[:-1]
        tasks = [{'goal': direction} for direction in directions]
        return tasks


    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self._goal_idx = idx
        self.reset()
        
    def get_all_task_idx(self):
        return range(self.num_tasks)
    
    def reset(self):
        self._step = 0
        return super().reset()

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()
        
    def get_task(self, ):
        return self._task
    
    def get_idx(self,):
        return self._goal_idx 
    
    def task_modes(self,):
        return {
            'train': list(range(10,30)),
            'moderate': list(range(5, 10)) + list(range(30,35)),
            'extreme': list(range(0,5)) + list(range(35,40))
        }
        
    def get_mode(self, ):
        idx = self._goal_idx
        for k,v in self.task_modes().items():
            if idx in v:
                return k
