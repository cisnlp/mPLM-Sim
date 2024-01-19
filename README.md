# mPLM-Sim: Better Cross-Lingual Similarity and Transfer in Multilingual Pretrained Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2305.13684-b31b1b.svg)](https://arxiv.org/abs/2305.13684)

mplm-sim is a language similarity tool providing:

- `Loader`: Accessing high-quality language similarity results directly.
- `Executor`: Obtaining similarity results from scratch.

## Quickstart

Download the repo for use or alternatively install with PyPi

`pip install mplm_sim`

or directly with pip from GitHub

`pip install --upgrade git+https://github.com/cisnlp/mPLM-Sim.git#egg=mplm_sim`

## Loader

```python
from mplm_sim import Loader

# loading existing results given model_name and corpus_name
loader = Loader.from_pretrained(model_name='cis-lmu/glot500-base', corpus_name='flores200')
# Or loading results given similarity file
# loader = Loader.from_tsv('your_similarity_file.tsv')

# Getting similarity given language pairs
# iso3_script
sim = loader.get_sim('eng_Latn', 'cmn_Hani')
# or language name
sim = loader.get_sim('English', 'Chinese')
```

## Executor

```python
from mplm_sim import Loader

# model_name: any text/speech language model support by Huggingface
# corpus_name: specific corpus name for saving
# corpus_path: path for multi-parallel corpora, see corpora_demo for file formatting
# corpus_type: text or speech
executor = Executor(model_name='cis-lmu/glot500-base', corpus_name='own',
                    corpus_path='corpora/own', corpus_type='text')

# Run
executor.run()
```

## Citation

```
@article{DBLP:journals/corr/abs-2305-13684,
  author       = {Peiqin Lin and
                  Chengzhi Hu and
                  Zheyu Zhang and
                  Andr{\'{e}} F. T. Martins and
                  Hinrich Sch{\"{u}}tze},
  title        = {mPLM-Sim: Unveiling Better Cross-Lingual Similarity and Transfer in
                  Multilingual Pretrained Language Models},
  journal      = {CoRR},
  volume       = {abs/2305.13684},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.13684},
  doi          = {10.48550/ARXIV.2305.13684},
  eprinttype    = {arXiv},
  eprint       = {2305.13684},
  timestamp    = {Mon, 05 Jun 2023 15:42:15 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-13684.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

