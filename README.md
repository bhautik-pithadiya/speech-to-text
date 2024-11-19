## Installing the packages

create virtual env
python3 -m venv <env_name>

now update the pip package
pip install --upgrade pip

Installing Nemo toolkit and whisperx via git link
    python -m pip install git+https://github.com/NVIDIA/NeMo.git@52d50e9e09a3e636d60535fd9882f3b3f32f92ad
    python -m pip install whisperx @ git+https://github.com/m-bain/whisperx.git

now to install youtokentome
1. install cython
        pip install Cython
2. install youtokentome
        pip install youtokentome

Now install the remaining packages from requirements.txt
    pip install -r requirements


# After completing the installation and then also you are facing an error named
 - ImportError: cannot import name 'ModelFilter' from 'huggingface_hub' while importing from nemo.collections.asr.models import EncDecMultiTaskModel
    - then its nothing but a version problem with nemo-toolkit[all] pac