
# ReweightingDisfluency
This is the PyTorch implementation of the
- COLING2022 paper [Adaptive Unsupervised Self-training for Disfluency Detection](https://aclanthology.org/2022.coling-1.632.pdf)
<div align="center">
    <image src='image/structure.jpg'>
</div>
All the code and model are released. Thank you for your patience!

# About Model
We release our self-supervised model trained by pseudo data and grammar check model. 
Please download it in the following link, and put model in "./ckpt/teacher" and "./ckpt/judge " folder.

- [teacher_model](https://drive.google.com/file/d/1UgqDcWo0gB4DpUCew2XJF848KYrLHTHx/view?usp=sharing) in "./ckpt/teacher"

- [judge_model](https://drive.google.com/file/d/1A7tuE0PDKN8_1RSsl74uz5GQPkGFUKvA/view?usp=sharing) in "./ckpt/judge"

You need to put your data and model in the parallel folder of this repo:
```text
    - ckpt/
        - electra_en_base
            - config.json
            - pytorch_model.bin
            - vocab.txt
        - teacher
            - pytorch_model.bin
        - judge
            - pytorch_model.bin
    - self_training/
        - run_data/
            - 500/
                - unlabel.tsv
                - dev.tsv
                - test.tsv
        - run_model/
    - src/
        - model.py
        ...
    - run.sh
```
# About data
Due to copyright issues, we do not have the right to distribute the SWBD dataset and can purchase it for your own use.

# Requirements
- transformers==**4.7.0**
- pytorch==1.9
- numpy
- tensorboardX

# How to use
The file path and training details can be set in the script `run.sh`
```shell
nohup sh run.sh > log_run 2>&1 &
```

# Citation
@inproceedings{wang2022adaptive,
  title={Adaptive Unsupervised Self-training for Disfluency Detection},
  author={Wang, Zhongyuan and Wang, Yixuan and Wang, Shaolei and Che, Wanxiang},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={7209--7218},
  year={2022}
}

# Contact
If you have any question about this code, feel free to open an issue or contact yixuanwang@ir.hit.edu.cn.
