import sys
import logging

model2type = {'bert-base-multilingual-cased': 'mbert',
              'google/canine-s': 'mbert', 'google/canine-c': 'mbert',
              'xlm-roberta-base': 'xlmr', 'xlm-roberta-large': 'xlmr',
              'facebook/xlm-roberta-xl': 'xlmr', 'facebook/xlm-roberta-xxl': 'xlmr',
              'microsoft/infoxlm-base': 'infoxlm', 'microsoft/infoxlm-large': 'infoxlm',
              'microsoft/xlm-align-base': 'infoxlm', 'google/mt5-small': 'mt5',
              'google/mt5-base': 'mt5', 'google/mt5-large': 'mt5',
              'google/byt5-small': 'mt5', 'google/byt5-base': 'mt5',
              'google/byt5-large': 'mt5', 'facebook/m2m100_1.2B': 'm2m',
              'sberbank-ai/mGPT': 'mgpt', 'bigscience/bloom-560m': 'bloom',
              'bigscience/bloom-1b1': 'bloom', 'bigscience/bloom-1b7': 'bloom',
              'bigscience/bloom-3b': 'bloom', 'bigscience/bloom-7b1': 'bloom',
              'facebook/nllb-200-1.3B': 'nllb', 'facebook/nllb-200-3.3B': 'nllb',
              'facebook/wav2vec2-xls-r-300m': 'xlsr', 'facebook/wav2vec2-xls-r-1b': 'xlsr',
              'cis-lmu/glot500-base': 'glot500'}


def read_file(corpora_path, corpus_type, lang):
    if corpus_type == 'text':
        with open(corpora_path + '/' + lang + '.txt', 'r') as f:
            sents = []
            lines = f.readlines()
            sents.extend([line.strip() for line in lines])
            return sents
    elif corpus_type == 'speech':
        with open(corpora_path + '/' + lang + '.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
            return dataset


def param_check(model_name, corpus_name):
    # Check arguments
    if model_name not in model2type:
        raise ValueError('%s not exist in %s' % (model_name, model2type.keys()))
    if corpus_name not in corpora:
        raise ValueError('%s not exist in %s' % (corpus_name, corpora))

def get_lang_list(model_name, corpus_name):
    # Get common languages
    model_type = model2type[model_name]
    with open(lang_path + model_type + '.txt', 'r') as f:
        lines = f.readlines()
        model_langs = [line.strip() for line in lines]

    with open(lang_path + corpus_name + '.txt', 'r') as f:
        lines = f.readlines()
        corpus_langs = [line.strip() for line in lines]
    logger.info('Model name: %s; Corpus name: %s' % (model_name, corpus_name))
    lang_scripts = list(set(model_langs) & set(corpus_langs))
    lang_scripts.sort()
    langs = list(set([lang.split('_')[0] for lang in lang_scripts]))
    logger.info('Model langs: %s; Corpus langs: %s; Common lang_scripts: %s; Common langs: %s' % (
        len(model_langs), len(corpus_langs), len(lang_scripts), len(langs)))
    logger.info('Pooling: %s; Corpus size (sentence number): %s' % (pooling_type, corpus_size))

def get_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    return logger

