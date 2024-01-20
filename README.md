# The Curse of Recursion: Training on Generated Data Makes Models Forget 


This repository contains code for a publication "The Curse of Recursion: Training on Generated Data Makes Models Forget". 

The paper can be found on <a href="https://arxiv.org/abs/2305.17493">here</a>.

In case of questions please do not hesitate reaching out! To cite please use:
```
@misc{shumailov2023curse,
      title={The Curse of Recursion: Training on Generated Data Makes Models Forget}, 
      author={Ilia Shumailov and Zakhar Shumaylov and Yiren Zhao and Yarin Gal and Nicolas Papernot and Ross Anderson},
      year={2023},
      eprint={2305.17493},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

The codebase is not really a codebase, I cut out most things related to our specific hardware and slurm setup. Should be an easy backbone to replicate the experiments.

Our runner script is in the `runme_base.py`, `dataset.py` does data loading and `main.py` has all of the lightning specifics.
