import os
import pandas as pd
from fuzzywuzzy import process

class Loader:
    dir_path = os.path.join(os.path.dirname(__file__), 'sim/model_corpus')

    @classmethod
    def initialize_class_attributes(cls):
        cls.dir_names = [name for name in os.listdir(cls.dir_path) if os.path.isdir(os.path.join(cls.dir_path, name))]
        cls.support_models = set([name.split('_')[0] for name in cls.dir_names])
        cls.support_corpora = set([name.split('_')[1] for name in cls.dir_names])

    def __init__(self, tsv_file):  
        self.tsv_file = os.path.join(self.dir_path, tsv_file)

    @classmethod
    def from_pretrained(cls, model_name, corpus_name, layer=None):
        cls.initialize_class_attributes()
             
        model_name = model_name.split('/')[-1] 
        # check whether model is supported
        if model_name not in cls.support_models:
            raise ValueError(f'Model {model_name} not supported, the current supported models are {cls.support_models}.')
        if corpus_name not in cls.support_corpora:
            raise ValueError(f'Corpus {corpus_name} not supported, the current supported models are {cls.support_corpora}.')
   
        if not layer:
            layer = cls.get_best_layer(model_name, corpus_name)
            # print('Best layer for ' + model_name + '_' + corpus_name + ' is ' + str(layer))
        tsv_file = model_name + '_' + corpus_name + '/sents_layer' + str(layer) + '.tsv'
        return cls(tsv_file)

    @classmethod
    def from_tsv(self, tsv_file):
        return cls(tsv_file)

    # Get best layer given model_name, corpus_name
    def get_best_layer(model_name, corpus_name):
        with open(os.path.join(os.path.dirname(__file__), 'infos/best_layer.tsv'), 'r') as f:
            for line in f:
                if model_name in line and corpus_name in line:
                    return int(line.split('\t')[2])

    def get_sim(self, lang1, lang2):
        langscript = []
        name2langscript = {}
        with open(os.path.join(os.path.dirname(__file__), 'infos/langscript2name.tsv'), 'r') as f:
            for line in f:
                name2langscript[line.split('\t')[1].strip()] = line.split('\t')[0]
                langscript.append(line.split('\t')[0])

        # deal with language name
        if lang1 in langscript:
            l1 = lang1
        elif lang1 in name2langscript.keys():
            l1 = name2langscript[lang1]
        else:
            # Find the closest match to lang1 in langscript and name2langscript.keys()
            closest_match = process.extractOne(lang1, list(langscript) + list(name2langscript.keys()))
            #print(closest_match)
            l1 = None

        if lang2 in langscript:
            l2 = lang2
        elif lang2 in name2langscript.keys():
            l2 = name2langscript[lang2]
        else:
            # Find the closest match to lang2 in langscript and name2langscript.keys()
            closest_match = process.extractOne(lang2, list(langscript) + list(name2langscript.keys()))
            #print(closest_match)
            l2 = None

        # raise error and hint by one most similar language name
        if not l1 or not l2:
            raise ValueError(f'Language name not found, please check your input. Did you mean {closest_match[0]} instead?')
        
        # deal with supported languages
        df_sim = pd.read_csv(self.tsv_file, sep='\t', header=0, index_col=0)
        supported_langs = df_sim.index.tolist()
        if l1 not in supported_langs:
            raise ValueError(f'Language {l1} not supported by the model for this corpus, please use get_language_list() to check the supported languages.')
        if l2 not in supported_langs:
            raise ValueError(f'Language {l2} not supported by the model for this corpus, please use get_language_list() to check the supported languages.')
        # get sim score
        return df_sim.loc[l1, l2]
    

    def get_language_list(self):
        df_sim = pd.read_csv(self.tsv_file, sep='\t', header=0, index_col=0)
        langscript2name = {}
        with open(os.path.join(os.path.dirname(__file__), 'infos/langscript2name.tsv'), 'r') as f:
            for line in f:
                langscript2name[line.split('\t')[0]] = line.split('\t')[1].strip()

        # use index to get language list
        index_list =  df_sim.index.tolist()
        language_list = [langscript2name[i] for i in index_list]
        return language_list

