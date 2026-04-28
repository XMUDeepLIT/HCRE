
declare -A model2name
declare -A model2template

# LLaMA 3.1
model2name["llama3.1"]="Meta-Llama-3.1-8B-Instruct"
model2template["llama3.1"]="llama3"

# LLaMA 3
model2name["llama3"]="Meta-Llama-3-8B-Instruct"
model2template["llama3"]="llama3"

# Gemma 2
model2name["gemma2"]="gemma-2-9b-it"
model2template["gemma2"]="gemma"

# Qwen 2.5
model2name["qwen2.5"]="Qwen2.5-7B-Instruct"
model2template["qwen2.5"]="qwen"


no_prev=0
model=""
substitute=""
original_prompt=0
gpus=""
for arg in "$@"; do
  case "$arg" in
    --gpus=*) gpus="${arg#*=}" ;;
    --no_prev) no_prev=1 ;;
    --model=*) model="${arg#*=}" ;;
    --substitute=*) substitute="${arg#*=}" ;;
    --original_prompt) original_prompt=1 ;;
    --tree_name=*) tree_name="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done
tree_name=$( echo ${tree_name} | sed 's/\//-/g' )

dataset_name="multistep-${tree_name}"
exp="ms-${tree_name}"
if [ $no_prev -eq 1 ]; then
    exp="${exp}-NoPrev"
    dataset_name="${dataset_name}-NoPrev"
fi
if [ $original_prompt -eq 1 ]; then
    exp="${exp}-original_prompt"
    dataset_name="${dataset_name}-original_prompt"
fi
if [ -n "$substitute" ]; then
    exp="${exp}-Sub${substitute}"
    dataset_name="${dataset_name}-Sub${substitute}"
fi
dataset_name="${dataset_name}-CodRED-train"

model_name=${model2name[${model}]}
template_name=${model2template[${model}]}

train_dir="saves/${model_name}/lora/template_${template_name}/train-${exp}"


# Calculate batch size for each device, to keep a constant total batch size
function count_gpus () {
    local gpus=$1
    if [[ -z "$gpus" ]]; then
        echo "Error: CUDA_VISIBLE_DEVICES should not be empty." >&2
        echo 0
    fi
    echo $(echo "$gpus" | tr ',' '\n' | wc -l)
}
total_bs=32
num_gpus=$(count_gpus $gpus)
per_device_train_batch_size=$(( ${total_bs} / ${num_gpus} ))


# Train
echo "=========================== Training ==========================="
mkdir -p ${train_dir}
cmd="CUDA_VISIBLE_DEVICES=${gpus} llamafactory-cli train \
    --seed 42 \
    --stage sft \
    --do_train True \
    --model_name_or_path pretrained_models/${model_name} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template ${template_name} \
    --flash_attn fa2 \
    --dataset_dir data \
    --dataset ${dataset_name} \
    --cutoff_len 2400 \
    --learning_rate 5e-05 \
    --weight_decay 0.1 \
    --max_steps 6400 \
    --max_samples 1000000 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 800 \
    --warmup_steps 480 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir ${train_dir} \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --lora_target all"

cmd="${cmd} 2>&1 | tee ${train_dir}/${exp}.log"
echo $cmd
eval $cmd

