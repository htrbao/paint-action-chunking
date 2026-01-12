# 🚀 Pi0 torch version


## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install core dependencies

```bash
conda create -n pitorch python=3.10 -y
conda activate pitorch
pip install uv
uv pip install -e ".[pi0]"
uv pip install python-dotenv pytest serial nvitop 
uv pip install transformers==4.48.1 accelerate datasets==3.0.0


# download pretrained VLM backbone
export HF_HOME=...
export HF_TOKEN=... (your HF_TOKEN)
huggingface-cli download google/paligemma-3b-pt-224

# download pretrained VLA
cd <dir your wanna store pretrained weight>
git clone https://huggingface.co/datasets/ducido/temp

# install robocasa simulation for evaluation 
<I will send ASAP>

```

---

### Note:
- If you wanna set up custom dataset, you should set up in `configs/robocasa_config/policy_config.json` and modify `input_feature` or `output_feature`
- Additional, You should check following files somtime:
```
# lerobot/common/policies/pi0/modeling_pi0.py
# line 310 or 311, change to your model dir
self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", local_files_ony=False)
# self.language_tokenizer = AutoTokenizer.from_pretrained("/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/hf_cache/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c", local_files_ony=True)
        

# line 445, 446
# present_img_keys = [key for key in self.config.image_features if key in batch and "depth" not in key and "mask" not in key]
present_img_keys = ["left_image","right_image", "wrist_image"]


# lerobot/common/constants.py

# OBS_STATE = "observation.state"
OBS_STATE = "state"
# OBS_ACTION = "action"
OBS_ACTION = "actions"
```


## 🧪 Training


```bash
# down load data
ROBOCASA 30 DEMOS: https://huggingface.co/datasets/binhng/robocasa_30_demos_lerobot_5_chosen_tasks_v3

ROBOCASA 100 DEMOS: https://huggingface.co/datasets/binhng/robocasa_100_demos_lerobot_5_chosen_tasks_v2


# run bash file to train, change paths in the following files to your paths
bash script/debug.sh
bash scripts/ft_robocasa_30.sh
bash scripts/ft_robocasa_100.sh
# if your server blocks the internet, change the wandb model in bash file to offline
```

## ROBOCASA EVAL
## ROBOCASA EVAL
Install robocasa simulation
```
pip install uv
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
uv pip install -e .

cd ..
git clone https://github.com/jibby2803/robocasa_regenerate.git ./robocasa
cd robocasa
uv pip install -e .

python robocasa/scripts/download_kitchen_assets.py  
python robocasa/scripts/setup_macros.py  

```

Change the robocasa path, model path, exp_name, etc. NOTE: args.gpus is just used for save log file (Sorry for my stupid !!!!). Check file: `scripts/eval_robocasa.py`.
Make sure that 25 trials use seed 0 and 25 trials use seed 1 for reproduce
```
python scripts/eval_robocasa.py \
   --args.exp-name base_robocasa_100demos_60k \
   --args.num-trials 25 \
  --args.seed 0 \
   --args.gpu 0

python scripts/eval_robocasa.py \
   --args.exp-name base_robocasa_100demos_60k \
   --args.num-trials 25 \
  --args.seed 1 \
   --args.gpu 1
```