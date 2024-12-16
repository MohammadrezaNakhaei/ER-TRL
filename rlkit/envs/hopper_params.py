import numpy as np
import pdb
from gym import utils 
from gym.envs.mujoco import mujoco_env

from . import register_env


class HopperENV(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._max_episode_steps = 200
        self._step = 0
        mujoco_env.MujocoEnv.__init__(self, "hopper.xml", 4)
        utils.EzPickle.__init__(self)

                
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        # done = False
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20


@register_env('hopper-mass')
class HopperParamsMass(HopperENV):
    def __init__(self,idx=0, **kwargs):
        super(HopperParamsMass, self).__init__()
        self.original_mass = np.copy(self.model.body_mass)
        self.original_inertia = np.copy(self.model.body_inertia)
        self.num_tasks = 40
        self.num_train = 20
        self.num_moderate = 10
        self.num_extreme = 10
        self.tasks = self.sample_tasks()
        self._max_episode_steps = 200
        self._step = 0
        self.reset_task(idx)
        
    def sample_tasks(self, ):
        logscales = np.linspace(-2, 2.0, num=self.num_tasks)
        tasks = [{'scale': 1.5**logscale} for logscale in logscales]
        return tasks

    def get_all_task_idx(self):
        return range(self.num_tasks)
    
    def reset(self):
        self._step = 0
        return super().reset()

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = self._task['scale']
        self.model.body_mass[:] = self.original_mass*self._goal
        self.model.body_inertia[:] = self.original_inertia*self._goal
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

@register_env('hopper-friction')
class HopperParamsFriction(HopperENV):
    def __init__(self,idx=0, **kwargs):
        super(HopperParamsFriction, self).__init__()
        self.original_friction = np.copy(self.model.geom_friction)
        self.num_tasks = 40
        self.num_train = 20
        self.num_moderate = 10
        self.num_extreme = 10
        self.tasks = self.sample_tasks()
        self._max_episode_steps = 200
        self._step = 0
        self.reset_task(idx)

    def sample_tasks(self, ):
        logscales = np.linspace(-2, 2.0, num=self.num_tasks)
        tasks = [{'scale': 1.5**logscale} for logscale in logscales]
        return tasks

    def get_all_task_idx(self):
        return range(self.num_tasks)
    
    def reset(self):
        self._step = 0
        return super().reset()

    def reset_task(self, idx):
        self._goal_idx = idx
        self._task = self.tasks[idx]
        self._goal = self._task['scale']
        self.model.geom_friction[:] = self.original_friction*self._goal
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
