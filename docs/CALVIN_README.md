

## 🛠️ CALVIN Environment Setup

First, download zip file here: https://drive.google.com/file/d/172bQqDPygSsgcrekYLSZdCdfgsH7wvn6/view?usp=sharing, put it in ./

```bash
# Install calvin_models
cd calvin/calvin_models
pip install -e .

# Install calvin_env
cd ../calvin_env
pip install -e .

# Install tacto
cd ../tacto
pip install -e .

# Install additional packages
pip install moviepy==1.0.3
pip install networkx==3.4.2
pip install jupyterlab termcolor pyhash pytorch-lightning tensorflow
```