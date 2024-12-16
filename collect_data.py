import hydra 

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from rlkit.torch.pytorch_sac.train import Workspace
import pdb


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="data_collection")
def main(cfg):
    env = NormalizedBoxEnv(ENVS[cfg.env_name](**dict(cfg.env_params)))
    if cfg.seed is not None:
        env.seed(cfg.seed) 
    env.reset_task(cfg.goal_idx)
    workspace = Workspace(cfg=cfg, env=env,)
    workspace.run()
    

if __name__ == '__main__':
    #add a change 
    main()
