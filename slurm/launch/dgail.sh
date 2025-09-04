# Baseline DGAIL with 3 seeds.
for divergence in wass rkl js
do
    for normalize_reward in True False
    do
        for seed in 0 1 2
        do
            sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/can env.n_envs=25 train.n_steps=600 train.normalize_reward=$normalize_reward expert_dataset.max_n_episodes=50 model.divergence=$divergence model.max_discriminator_diff=10.0 train.n_train_itr=201
            sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/lift env.n_envs=25 train.n_steps=600 train.normalize_reward=$normalize_reward expert_dataset.max_n_episodes=50 model.divergence=$divergence model.max_discriminator_diff=10.0 train.n_train_itr=131
            sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/square env.n_envs=25 train.n_steps=800 train.normalize_reward=$normalize_reward expert_dataset.max_n_episodes=50 model.divergence=$divergence model.max_discriminator_diff=10.0 train.n_train_itr=401
            sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/transport env.n_envs=25 train.n_steps=800 train.normalize_reward=$normalize_reward expert_dataset.max_n_episodes=50 model.divergence=$divergence model.max_discriminator_diff=10.0 train.n_train_itr=251
        done
    done
done

# DGAIL, but with no model init.
# for divergence in wass rkl js
# do
#     for normalize_reward in True False
#     do
#         for seed in 0 1 2
#         do
#             sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/can env.n_envs=25 train.n_steps=600 base_policy_path=null train.normalize_reward=$normalize_reward expert_dataset.max_n_episodes=50 model.divergence=$divergence model.max_discriminator_diff=10.0
#             sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/lift env.n_envs=25 train.n_steps=600 base_policy_path=null train.normalize_reward=$normalize_reward expert_dataset.max_n_episodes=50 model.divergence=$divergence model.max_discriminator_diff=10.0
#             sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/square env.n_envs=25 train.n_steps=800 base_policy_path=null train.normalize_reward=$normalize_reward expert_dataset.max_n_episodes=50 model.divergence=$divergence model.max_discriminator_diff=10.0
#             sbatch slurm/run.sh --config-name=ft_gail_diffusion_mlp_img --config-dir=cfg/robomimic/imitation/transport env.n_envs=25 train.n_steps=800 base_policy_path=null train.normalize_reward=$normalize_reward expert_dataset.max_n_episodes=50 model.divergence=$divergence model.max_discriminator_diff=10.0
#         done
#     done
# done
