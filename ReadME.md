# Environment setup: create viturual environment with Anaconda
conda create -n openai_gym python=3.11
pip install gymnasium[classic-control]
pip install gymnasium[toy-text]
pip install gymnasium[mujoco]
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
conda install -n openai_gym ipykernel --update-deps --force-reinstall
pip install matplotlib
pip install torch torchvision torchaudio

# Run 
frozen_lake_Q.py runs the process with a basic Q-table  
frozen_lake_8x8_DeepQ.py runs the DeepQNetwork process
# Note: ALE Documentation Contributions folder contains the results of my documeentation changes that were accepted to the official package