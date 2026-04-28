
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
original_prompt=0
tree_name=""
vmethods=""
fd_model=0
subsets="dev"
run_id=""
gpus=""
for arg in "$@"; do
  case "$arg" in
    --gpus=*) gpus="${arg#*=}" ;;
    --no_prev) no_prev=1 ;;
    --model=*) model="${arg#*=}" ;;
    --subsets=*) subsets="${arg#*=}" ;;
    --vmethods=*) vmethods="${arg#*=}" ;;
    --fd_model) fd_model=1 ;;
    --original_prompt) original_prompt=1 ;;
    --tree_name=*) tree_name="${arg#*=}" ;;
    --run_id=*) run_id="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done
unified_tree_name=$( echo ${tree_name} | sed 's/\//-/g' )

exp="ms-${unified_tree_name}"
if [ $no_prev -eq 1 ]; then
    exp="${exp}-NoPrev"
fi
if [ $original_prompt -eq 1 ]; then
    exp="${exp}-original_prompt"
fi
if [ -n "$vmethods" ]; then
    if [ $fd_model -eq 1 ]; then
        exp="${exp}-Subbest.suboptimal.double"
    else 
        exp="${exp}-Sub${vmethods}"
    fi
fi
model_name=${model2name[${model}]}
template_name=${model2template[${model}]}

na_type=$( echo ${tree_name} | grep -oP 'NA\K\d+' )
tree_type=$( echo ${tree_name} | cut -d'/' -f2 )

# for step in 6400 5600 4800
for step in 6400
do
    train_dir="saves/${model_name}/lora/template_${template_name}/train-${exp}/checkpoint-${step}"
    for subset in ${subsets}
    do
        echo "=========================== eval-${exp}-${subset}-step${step} ==========================="
        eval_dir="${train_dir}/eval-${vmethods}-${subset}"
        if [ -n "$run_id" ]; then
            eval_dir="${eval_dir}-Run${run_id}"
        fi
        mkdir -p $eval_dir
        
        cmd="CUDA_VISIBLE_DEVICES=${gpus} python main.py \
            --seed 42 \
            --model_name_or_path pretrained_models/${model_name} \
            --preprocessing_num_workers 16 \
            --template ${template_name} \
            --dataset_dir data \
            --eval_dataset CodRED-${subset} \
            --cutoff_len 2400 \
            --max_new_tokens 10 \
            --top_p 1.0 \
            --temperature 0.0 \
            --lora_adapter_path ${train_dir} \
            --tree_instance data/tree_instances/${tree_name} \
            --output_dir ${eval_dir} \
            --na_type ${na_type} \
            --multistep_gen"

        if [ $no_prev -eq 1 ]; then
            cmd="${cmd} --no_prev"
        fi
        if [ $original_prompt -eq 1 ]; then
            cmd="${cmd} --original_prompt"
        fi
        if [ -n "$vmethods" ]; then
            cmd="${cmd} --verification_methods ${vmethods}"
        fi

        cmd="${cmd} 2>&1 | tee ${eval_dir}/${exp}.log"

        echo $cmd
        eval $cmd

        eval_cmd="python eval_predictions.py --path_label --tree_name ${tree_type} --na_type ${na_type} --eval_subset ${subset} --eval_dir ${eval_dir} 2>&1 | tee ${eval_dir}/${exp}.eval.log"
        echo $eval_cmd
        eval $eval_cmd
        
    done
done

