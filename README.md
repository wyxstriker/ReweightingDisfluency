
# ReweightingDisfluency
This is the PyTorch implementation of the
- COLING2022 paper [Adaptive Unsupervised Self-training for Disfluency Detection](http://www.baidu.com)


All the code and model are released. Thank you for your patience!
# About Model
We release our self-supervised model trained by pseudo data and grammar check model. 
Please download it in the following link, and put model in "./ckpt/teacher" and "./ckpt/judge " folder.

- [teacher_model](http://www.baidu.com) in "./ckpt/teacher"

- [judge_model](http://www.baidu.com) in "./ckpt/judge"

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

# Requirements
- transformers==**4.7.0**
- pytorch==1.9
- numpy
- tensorboardX

# How to use

```shell
nohup sh run.sh > log_run 2>&1 &
```

# Citation
//TODO

# Contact
If you have any question about this code, feel free to open an issue or contact yixuanwang@ir.hit.edu.cn.
