import pandas as pd 
import pickle

env_ids = ['cheetah-vel-custom', 'ant-goal-custom', 'ant-dir-custom', 'humanoid-dir-custom', 
           'hopper-mass', 'hopper-friction', 'walker-mass', 'walker-friction'] 


task_id_conv = {}
for env in env_ids:
    if env == 'ant-goal-custom':
        task_id_conv[env] = {
            'train': list(range(0,20)),
            'moderate': list(range(20, 30)),
            'extreme': list(range(30,40))
        }
    else:
        task_id_conv[env] = {
            'train': list(range(10,30)),
            'moderate': list(range(5, 10)) + list(range(30,35)),
            'extreme': list(range(0,5)) + list(range(35,40))
        }

with open('normalizer.pkl', 'rb') as f:
    normalizer = pickle.load(f)



def load_csv(env_id, algo, seed):
    path = f'../output/{env_id}/{algo}/seed{seed}/progress.csv'
    df = pd.read_csv(path)
    return df

def normalize_df(df, env_id, normalize=True):
    modes = dict(train=20, moderate=10, extreme=10)
    for mode, n_task in modes.items():
        for context in ['offline', 'online', 'nonprior']:
            cols = []
            for i in range(n_task):
                value = df[f'{mode}/{i}_final_{context}']
                idx = task_id_conv[env_id][mode][i]
                min_val, max_val = normalizer[env_id][idx]
                normalized = 100 * (value-min_val)/(max_val-min_val) 
                col = f'{mode}/{context}_normal_{i}'
                if normalize:
                    df[col] = normalized
                else:
                    df[col] = value
                cols.append(col)
            df[f'{mode}/{context}_normal_avg'] = df[cols].mean(1)
    return None

env_names = {
    'ant-dir-custom': 'Ant-Dir', 
    'ant-goal-custom': 'Ant-Goal',
    'humanoid-dir-custom': 'Humanoid-Dir',
    'cheetah-vel-custom': 'Cheetah-Vel', 
    'hopper-mass': 'Hopper-Mass',
    'hopper-friction': 'Hopper-Friction',
    'walker-mass': 'Walker-Mass',
    'walker-friction': 'Walker-Friction'
}

algo_names = ['OfflinePearl', 'FOCAL', 'CSRO', 'UNICORN', 'ER-TRL']

def load_total_dfs(n_seed=5, normalize=True):
    total_dfs = {}
    errors = []
    for algo in algo_names:
        total_dfs[algo] = {}
        for env in env_ids:
            result = []
            for seed in range(n_seed):
                try:
                    df = load_csv(algo=algo, env_id=env, seed=seed)
                    normalize_df(df, env_id=env, normalize=normalize)
                    result.append(df)
                except:
                    errors.append((env, algo, seed)) 
            total_dfs[algo][env] = result
    return total_dfs, errors