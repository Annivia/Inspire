pip install huggingface_hub

cd /projects/bfbo/xzhang42/Inspire/pretrained/
current_dir=$(pwd)
cd $HF_HOME/hub
huggingface_cache_dir=$(pwd)
cd $current_dir

huggingface-cli download Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b
if [ ! -d "prism-qwen25-extra-dinosiglip-224px-0_5b" ]; then
    ln -s ${huggingface_cache_dir}/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498 ${current_dir}/prism-qwen25-extra-dinosiglip-224px-0_5b
fi


cd /projects/bfbo/xzhang42/Inspire/runs/

current_dir=$(pwd)
cd $HF_HOME/hub
huggingface_cache_dir=$(pwd)
cd $current_dir

huggingface-cli download InspireVLA/minivla-libero-90
if [ ! -d "minivla-libero-90" ]; then
    ln -s ${huggingface_cache_dir}/models--InspireVLA--minivla-libero-90/snapshots/701f0245ae413beba3ebe35dc5e0057fdeb33bb2 ${current_dir}/minivla-libero-90
fi

huggingface-cli download InspireVLA/minivla-inspire-libero-90
if [ ! -d "minivla-inspire-libero-90" ]; then
    ln -s ${huggingface_cache_dir}/models--InspireVLA--minivla-inspire-libero-90/snapshots/b87707dd847b9361e267ee8ee2dbb98e852e13c1 ${current_dir}/minivla-inspire-libero-90
fi

cd /projects/bfbo/xzhang42/Inspire/pretrained/

current_dir=$(pwd)
cd $HF_HOME/hub
huggingface_cache_dir=$(pwd)
cd $current_dir

huggingface-cli download Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b
if [ ! -d "prism-qwen25-extra-dinosiglip-224px-0_5b" ]; then
    ln -s ${huggingface_cache_dir}/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498 ${current_dir}/prism-qwen25-extra-dinosiglip-224px-0_5b
fi

cd /projects/bfbo/xzhang42/Inspire/

current_dir=$(pwd)
cd $HF_HOME/hub
huggingface_cache_dir=$(pwd)
cd $current_dir

huggingface-cli download Stanford-ILIAD/pretrain_vq
if [ ! -d "vq" ]; then
    ln -s ${huggingface_cache_dir}/models--Stanford-ILIAD--pretrain_vq/snapshots/30ef2227f97dde1abb6d522ea80af85384235008 ${current_dir}/vq
fi