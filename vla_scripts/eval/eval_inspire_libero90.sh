export cuda_visible_devices=0,1,2,3

num_gpus=4
num_processes=4
task_suite_names=(
    "libero_90"
    "libero_goal"
    "libero_object"
    "libero_spatial"
    "libero_10"
)

name=minivla-inspire-libero-90
steps="50000"

for task_suite_name in "${task_suite_names[@]}"; do
    python /u/xzhang42/Inspire/vla_scripts/parallel_libero_evaluator.py \
        --num-trails-per-task 10 \
        --num-gpus $num_gpus \
        --num-processes $num_processes \
        --task-suite-name $task_suite_name \
        --pretrained-checkpoint /work/nvme/bfbo/xzhang42/Inspire/runs/$name \
        --save-root /work/nvme/bfbo/xzhang42/Inspire/results/$name \
        --with-vqa true \
        --steps $steps 
done
