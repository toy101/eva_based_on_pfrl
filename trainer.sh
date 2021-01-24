envs=(
    # "PongNoFrameskip-v4"
    "MsPacmanNoFrameskip-v4"
    "QbertNoFrameskip-v4"
)

for env_name in "${envs[@]}"; do
     python3 train.py --gpu 0 --agent EVA --env ${env_name}
done