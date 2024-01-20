import socket
import numpy as np
import os  


def format_and_save(script, template, **kwargs):

    temp = template.format(**kwargs)

    sc = open(script, 'w')
    sc.write(temp)
    sc.close()


slurm_script_template = ()

postf = ""
scriptsfolder = f""
datfolder = f""

os.makedirs(scriptsfolder, exist_ok=True)

expfolder = ""

names = []
for j in range(5):
    for i in range(5):
        names.append(f'r{j}{i}')

for i, n1 in enumerate(names):
    _temp = slurm_script_template[:]

    prefix = f'TORCH_HOME={expfolder} HF_DATASETS_CACHE= TRANSFORMERS_CACHE= CUDA_VISIBLE_DEVICES=0 '
    _temp += prefix + f"python main.py -n 1 -a gpu -b 16 -lr 1e-5 -p -version_name test --load-name lightning_logs/{n1}/checkpoints/best.ckpt --eval_only --evalgen_only --saveto {scriptsfolder}/{n1}_realtrain"
    _temp += "\n"
    _temp += prefix + f"python main.py -n 1 -a gpu -b 16 -lr 1e-5 -p -version_name test --load-name lightning_logs/{n1}/checkpoints/best.ckpt --eval_only --saveto {scriptsfolder}/{n1}_realtest"
    _temp += "\n"

    format_and_save(
        name = f"base{i}.sh",
        script = f"{scriptsfolder}/base{i}.sh",
        run_script = f"{scriptsfolder}/base{i}.sh",
        template = _temp,
        virt_env= "",
        exec_dir= f"{expfolder}",
    )







