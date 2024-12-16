# ER-TRL
Code base for AAAI submission, *Entropy Regularized Task Representation Learning for Offline Meta-RL*

## Instructions
### Install dependencies:
```sh
conda env create -f environment.yml
conda activate omrl
```

### install MuJoCo 2.10
* Download the Mujoco library from this [link](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz).
* Create a hidden folder :
```
mkdir /home/username/.mujoco
```
* Extract the library to the .mujoco folder.
```
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
```
* Include these lines in .bashrc file.
```
# Replace user-name with your username
echo -e 'export LD_LIBRARY_PATH=/home/user-name/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
export PATH="$LD_LIBRARY_PATH:$PATH" 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc
```
* Source bashrc.
```
source ~/.bashrc
```
* Test that the library is installed.
```
cd ~/.mujoco/mujoco210/bin
./simulate ../model/humanoid.xml
```

### Collecting datasets:
``` sh
python collect_data.py +env=cheetah-vel goal_idx=0 use_wandb=true
```
goal_idx is the task id which is between 0 and 39 (40 tasks). The ```+env``` argeument is compulsory. By default, we log metrics with wandb. You can ignore this by setting ```use_wandb=false```. 
The datasets are stored in ```offline_dataset``` directory.

You can see the list of environments in ```cfgs/env```.

### Training OMRL methods
``` sh
python launch_experiment.py +env=cheetah-vel +algo=ertrl use_wandb=true seed=0
```
Please note that you need to collect datasets for all of the tasks (```goal_idx=1:39```). The ```+algo``` and ```+env``` argeuments are compulsory. By default, metrics for each experiments are logged with wandb and default seed is 0. You can see the baselines in ```cfgs/algo``` and by changing ```+algo``` argeument, you can run other baselines.

## Ackowledgement 
We thanks the [CSRO](https://github.com/MoreanP/CSRO) authors for their open-source code.