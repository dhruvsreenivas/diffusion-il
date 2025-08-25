# Baseline DGAIL with 3 seeds.
for seed in 0 1 2
do
    sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/can env.n_envs=25 train.n_steps=600 train.normalize_reward=True
    sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/lift env.n_envs=25 train.n_steps=600 train.normalize_reward=True
    sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/square env.n_envs=25 train.n_steps=800 train.normalize_reward=True
    sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/transport env.n_envs=25 train.n_steps=800 train.normalize_reward=True
done

# DGAIL, but with no model init.
for seed in 0 1 2
do
    sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/can env.n_envs=25 train.n_steps=600 base_policy_path=null train.normalize_reward=True
    sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/lift env.n_envs=25 train.n_steps=600 base_policy_path=null train.normalize_reward=True
    sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/square env.n_envs=25 train.n_steps=800 base_policy_path=null train.normalize_reward=True
    sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/transport env.n_envs=25 train.n_steps=800 base_policy_path=null train.normalize_reward=True
done