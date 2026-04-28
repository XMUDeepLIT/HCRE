
# HCRE

Our checkpoints and data are released at [Google Drive](https://drive.google.com/drive/folders/1O_WjHWFNrU1jeC26AjmkK530KJOFV8MD?usp=sharing). 

## Environment Setup

1. Prepare environment for data preprocessing and model training. 

    ```bash
    conda create -n cdre python=3.10 -y
    conda activate cdre

    cd train/
    pip install --upgrade pip
    pip install -e ".[torch,metrics]"
    ```

2. Prepare environment for model inference.

    ```bash
    conda create -n vllm python=3.10 -y

    cd inference/
    pip install --upgrade pip
    pip uninstall fschat torchaudio torchvision -y
    pip install -r requirements.txt
    ```

## Data Preprocessing

1. Follow [the preprocessing guideline of CodRED](https://github.com/thunlp/CodRED) to prepare your redis database.

2. Convert dataset formats to JSON.
    ```bash
    conda activate cdre
    cd train/data/rawdata/
    python raw2json.py
    ```

3. Preprocess text paths using ECRIM's document-context filter.

    ```bash
    PRETRAINED_MODELS=../../pretrained_models python preprocess_data_ecrim_ic.py 0 129547 0 train
    PRETRAINED_MODELS=../../pretrained_models python preprocess_data_ecrim_ic.py 0 40739 0 dev
    PRETRAINED_MODELS=../../pretrained_models python preprocess_data_ecrim_ic.py 0 77934 0 open_dev
    ```

4. Construct the hierarchical relation tree. 
    ```bash
    cd auto-tree/
    python main-meaningful_levels_v2.py --exp_name "mlv2-gpt4o"
    ```

5. Prepare training data tailored to our inference strategy and format dev sets for evaluation.

    ```bash
    python generate_multistep_dataset.py --subsets train,dev,open_dev --na_type 0 --tree_name mlv2-gpt4o --no_prev --substitute 'best,suboptimal,double'

    mkdir -p ../cdre_data/
    mv multistep/* ../cdre_data/
    mv CodRED-dev.json ../cdre_data/
    mv CodRED-open_dev.json ../cdre_data/
    ```

<!-- Please note that we also provide the preprocessed data placed in the `cdre_data/` director.  -->

## Model Training

Our models are trained based on [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory).

1. For training `HCRE`: 

    ```bash
    cd train/
    conda activate cdre
    bash scripts/ms.sh --no_prev --model="llama3.1" --substitute="best.suboptimal.double" --tree_name="NA0/mlv2-gpt4o" --gpus="0,1,2,3"
    ```

2. For training the `Vanilla` baseline: 

    ```bash
    cd train/
    conda activate cdre
    bash scripts/plain.sh --model="llama3.1" --na_type=0 --gpus="0,1,2,3"
    ```

All your training results will placed at the `saves/` directory. 

## Model Inference

1. For running `HCRE`: 

    ```bash
    cd inference/
    conda activate vllm
    bash scripts/ms.sh --no_prev --model="llama3.1"  --subsets="dev open_dev" --vmethods="best.suboptimal.double" --fd_model --tree_name="NA0/mlv2-gpt4o" --gpus="0,1,2,3"
    ```

2. For running the `Vanilla` baseline: 

    ```bash
    cd inference/
    conda activate vllm
    bash scripts/plain.sh --model="llama3.1" --na_type=0 --subsets="dev open_dev" --gpus="0,1,2,3"
    ```

All your evaluation results will placed at the `saves/` directory. 



