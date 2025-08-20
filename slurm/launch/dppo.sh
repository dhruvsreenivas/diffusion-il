# Baseline DPPO with 3 seeds.
for seed in 0 1 2
do
    sbatch slurm/run.sh --config-name=ft_ppo_diffusion_mlp_img --config-dir=cfg/robomimic/finetune/can env.n_envs=25 train.n_steps=600
    sbatch slurm/run.sh --config-name=ft_ppo_diffusion_mlp_img --config-dir=cfg/robomimic/finetune/lift env.n_envs=25 train.n_steps=600
    sbatch slurm/run.sh --config-name=ft_ppo_diffusion_mlp_img --config-dir=cfg/robomimic/finetune/square env.n_envs=25 train.n_steps=800
    sbatch slurm/run.sh --config-name=ft_ppo_diffusion_mlp_img --config-dir=cfg/robomimic/finetune/transport env.n_envs=25 train.n_steps=800
done