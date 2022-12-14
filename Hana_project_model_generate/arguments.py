# -*- coding: utf-8 -*-
"""arguments.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LlyKddUG2uBLqFGmLs3E46yr3Wr5TFPE
"""

# arguments를 설정합니다.

import os
from glob import glob
from dataclasses import dataclass, field

@dataclass
class TrainArguments:

    pretrained_model_name: str = field(
        default="beomi/kcbert-base",
    )
    downstream_task_name: str = field(
        default="document-classification",
    )
    downstream_corpus_name: str = field(
        default="Hana_power_v1",
    )
    downstream_corpus_root_dir: str = field(
        default="/content/drive/MyDrive/Hana_project/data",
    )
    downstream_model_dir: str = field(
        default="/content/drive/MyDrive/Hana_project/model",
    )
    downstream_model_checkpoint_fpath: str = field(
        default="/content/drive/MyDrive/Hana_project/model/checkpoint",
    )
    max_seq_length: int = field(
        default=128,
    )
    doc_stride: int = field(
        default=64,
    )
    max_query_length: int = field(
        default=32,
    )
    threads: int = field(
        default=4,
    )
    cpu_workers: int = field(
        default=os.cpu_count(),
    )
    save_top_k: int = field(
        default=1,
    )
    monitor: str = field(
        default="min val_loss",
    )
    seed: int = field(
        default=None,
    )
    overwrite_cache: bool = field(
        default=False,
    )
    force_download: bool = field(
        default=False,
    )
    test_mode: bool = field(
        default=False,
    )
    learning_rate: float = field(
        default=5e-5,
    )
    epochs: int = field(
        default=3,
    )
    batch_size: int = field(
        default=32,
    )
    fp16: bool = field(
        default=False,
    )
    tpu_cores: int = field(
        default=0,
    )
    tqdm_enabled: bool = field(
        default=True,
    )

