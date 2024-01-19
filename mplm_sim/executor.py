import os
import math
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, to_tree
from scipy.spatial.distance import squareform
from scipy.stats.stats import pearsonr, spearmanr
import torch
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2Model)
from mplm_sim.utils import *

class EmbeddingLoader(object):
    def __init__(self, model, lang, device=torch.device('cuda:0')):
        self.device = device
        self.config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        if 'GPT' in model or 'bloom' in model:
            self.emb_model = AutoModelForCausalLM.from_pretrained(model, config=self.config)
        elif 'mt5' in model or 'm2m' in model or 'nllb' in model or 'byt5' in model:
            self.emb_model = AutoModel.from_pretrained(model, config=self.config).encoder
        elif 'wav2vec2' in model:
            self.emb_model = Wav2Vec2Model.from_pretrained(model, config=self.config)
        else:
            self.emb_model = AutoModel.from_pretrained(model, config=self.config)
        self.emb_model.eval()
        self.emb_model.to(self.device)

        if 'wav2vec2' in model:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model, return_attention_mask=True,
                                                                      cache_dir=cache_dir)
        elif 'nllb' in model:
            self.tokenizer = AutoTokenizer.from_pretrained(model, src_lang=lang)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

    def get_embed_list_encoder(self, sent_batch):
        with torch.no_grad():
            inputs = self.tokenizer(sent_batch, truncation=True, return_tensors='pt')
            hiddens = self.emb_model(**inputs.to(self.device))['hidden_states']
            mask = inputs['attention_mask']
            emb = []
            for hidden in hiddens:
                emb.append(((hidden * mask.unsqueeze(2).float()).sum(1) / mask.sum(1).unsqueeze(
                    -1).float()).cpu().detach().numpy()[0])
            return emb

    def get_embed_list_decoder(self, sent_batch):
        with torch.no_grad():
            inputs = self.tokenizer(sent_batch, truncation=True, return_tensors='pt')
            hiddens = self.emb_model(**inputs.to(self.device))['hidden_states']

            input_ids = inputs['input_ids']
            seq_len = input_ids.size(1)
            # position weights for each token
            pos_weights = torch.arange(seq_len, 0, -1.0, device=self.device)
            pos_weights = pos_weights / pos_weights.sum()

            # weight the hidden states by their position
            # Warning: the following code only work for batch size=1
            emb = []
            for hidden in hiddens:
                weighted_hiddens = hidden.view(seq_len, -1) * pos_weights.view(-1, 1)
                emb.append(weighted_hiddens.sum(0).cpu().detach().numpy())
            return emb

    @staticmethod
    def get_downsampling_mask(attention_mask, downsampling_rate):
        new_seq_length = attention_mask.size(1) // downsampling_rate
        new_attention_mask = torch.zeros((attention_mask.size(0), new_seq_length), device=attention_mask.device)
        for i in range(new_seq_length):
            start_index = i * downsampling_rate
            end_index = start_index + downsampling_rate
            new_attention_mask[:, i] = attention_mask[:, start_index:end_index].sum(1)
        return new_attention_mask

    def get_embed_list_canine(self, sent_batch):
        with torch.no_grad():
            inputs = self.tokenizer(sent_batch, truncation=True, return_tensors='pt')
            hiddens = self.emb_model(**inputs.to(self.device))['hidden_states']
            attn_mask = inputs['attention_mask']
            downsampling_mask = self.get_downsampling_mask(attn_mask, self.config.downsampling_rate)
            emb = []
            for hidden in hiddens:
                mask = attn_mask if hidden.size(1) == attn_mask.size(1) else downsampling_mask
                emb.append(((hidden * mask.unsqueeze(2).float()).sum(1) / mask.sum(1).unsqueeze(
                    -1).float()).cpu().detach().numpy()[0])
            return emb

    def get_embed_list_wav2vec2(self, audio_batch):
        with torch.no_grad():
            inputs = self.processor(audio_batch, padding=True, return_tensors="pt", sampling_rate=16000)
            hiddens = self.emb_model(**inputs.to(self.device))['hidden_states']
            emb = []
            for hidden in hiddens:
                emb.append(hidden.mean(1).cpu().detach().numpy()[0])
            return emb


def get_emb(dataset, model, lang, device):
    embed_loader = EmbeddingLoader(model=model, lang=lang, device=device)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    embs = []
    for _, data in enumerate(tqdm(data_loader)):
        if 'GPT' in model or 'bloom' in model:
            emb = embed_loader.get_embed_list_decoder(data)
        elif 'canine' in model:
            emb = embed_loader.get_embed_list_canine(data)
        elif 'wav2vec2' in model:
            emb = embed_loader.get_embed_list_wav2vec2(data['audio']['array'].view(-1))
        else:
            emb = embed_loader.get_embed_list_encoder(data)
        embs.append(emb)
    return embs

class Executor:
    def __init__(self, model_name, corpus_name, corpus_path, corpus_type='text', save_path=None, device='0'):
        if model_name not in model2type:
            raise ValueError('%s not exist in %s' % (model_name, model2type.keys()))
        if corpus_type not in ['text', 'speech']:
            raise ValueError('corpus_type should be text or speech')
        
        self.model_name = model_name
        self.model_type = model2type[model_name]
        
        self.corpus_name = corpus_name
        self.corpus_type = corpus_type
        self.corpus_path = corpus_path

        self.lang_path = os.path.join(os.path.dirname(__file__), 'lang_list/')
        self.metadata_sim_path = os.path.join(os.path.dirname(__file__), 'sim/')
        if not save_path:
            self.save_path = 'save/' + model_name.split('/')[-1] + '_' + corpus_name + '/'
        else:
            self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.device = device

        self.logger = get_logger(self.save_path + 'result.log')


    def run(self):
        # Get common languages
        with open(self.lang_path + self.model_type + '.txt', 'r') as f:
            lines = f.readlines()
            model_langs = [line.strip() for line in lines]

        corpus_langs = [f.replace('.txt', '') for f in os.listdir(self.corpus_path) if '.txt' in f]

        lang_scripts = list(set(model_langs) & set(corpus_langs))
        lang_scripts.sort()
        langs = list(set([lang.split('_')[0] for lang in lang_scripts]))
        self.logger.info('Model langs: %s; Corpus langs: %s; Common lang_scripts: %s; Common langs: %s' % (
            len(model_langs), len(corpus_langs), len(lang_scripts), len(langs)))

        # Read corpora
        lang2emb = {}
        for lang in lang_scripts:
            save_fname = self.save_path + lang
            if os.path.exists(save_fname):
                with open(save_fname, 'rb') as f:
                    embs = pickle.load(f)
            else:
                sents = read_file(self.corpus_path, self.corpus_type, lang)
                # Sent Num * Layer Num * Hidden Size
                embs = get_emb(sents, self.model_name, lang, torch.device('cuda:' + self.device))
                with open(save_fname, 'wb') as f:
                    pickle.dump(embs, f)
            lang2emb[lang] = embs
            sent_num, layer_num, hidden_size = len(embs), len(embs[0]), len(embs[0][0])
        torch.cuda.empty_cache()
        self.logger.info('Sent num: %s; Layer num: %s; Hidden size: %s' % (sent_num, layer_num, hidden_size))

        # Cosine similarity for each sentence
        df_layer = [pd.DataFrame(index=lang_scripts, columns=lang_scripts) for _ in range(layer_num)]
        for layer_id in range(layer_num):
            for lang1 in lang_scripts:
                for lang2 in lang_scripts:
                    if lang2 <= lang1:
                        continue
                    df_layer[layer_id][lang2][lang1] = 0
                    df_layer[layer_id][lang1][lang2] = 0
        for sent_id in range(sent_num):
            emb1, emb2 = [], []
            for layer_id in range(layer_num):
                for lang1 in lang_scripts:
                    for lang2 in lang_scripts:
                        if lang2 <= lang1:
                            continue
                        emb1.append(lang2emb[lang1][sent_id][layer_id])
                        emb2.append(lang2emb[lang2][sent_id][layer_id])

            emb1, emb2 = np.array(emb1), np.array(emb2)
            sim = np.sum(emb1 * emb2, axis=1) / (norm(emb1, axis=1) * norm(emb2, axis=1))

            sim_id = 0
            for layer_id in range(layer_num):
                df = pd.DataFrame(index=lang_scripts, columns=lang_scripts)
                for lang1 in lang_scripts:
                    for lang2 in lang_scripts:
                        if lang2 <= lang1:
                            continue
                        df[lang1][lang2] = sim[sim_id]
                        df[lang2][lang1] = sim[sim_id]
                        sim_id += 1
                df_layer[layer_id] = df_layer[layer_id].add(df.loc[lang_scripts, lang_scripts])

        for layer_id in range(layer_num):
            # Average cosine similarity
            tsv_path = self.save_path + 'sents_' + 'layer' + str(layer_id) + '.tsv'
            df_layer[layer_id] = df_layer[layer_id] / sent_num
            df_layer[layer_id].to_csv(tsv_path, sep='\t')

        # Get metadata similarity
        features_origin = ['edit_' + self.corpus_name, 'genetic', 'geographic', 'syntactic', 'inventory',
                    'phonological', 'featural']
        features = []
        feature2df = {}
        for feature in features_origin:
            tsv_path = self.metadata_sim_path + feature + '.tsv'
            if os.path.exists(tsv_path):
                df_feature = pd.read_csv(tsv_path, sep='\t', index_col=0)
                features.append(feature)
                feature2df[feature] = df_feature

        # Correlation between LM sim and Metadata sim
        df_p = pd.DataFrame(index=features, columns=range(layer_num))
        df_lang_p = {lang: pd.DataFrame(index=features, columns=range(layer_num)) for lang in lang_scripts}
        df_feature_p = {feature: pd.DataFrame(index=lang_scripts, columns=range(layer_num)) for feature in features}
        df_s = pd.DataFrame(index=features, columns=range(layer_num))
        df_lang_s = {lang: pd.DataFrame(index=features, columns=range(layer_num)) for lang in lang_scripts}
        df_feature_s = {feature: pd.DataFrame(index=lang_scripts, columns=range(layer_num)) for feature in features}

        lang2result = {layer_id: [] for layer_id in range(layer_num)}

        for layer_id in range(layer_num):
            # Get the model and feature ranking for each language
            lang2mrank = {}
            lang2frank = {feature: {} for feature in features}
            for lang in lang_scripts:
                mrank = df_layer[layer_id][lang].sort_values(ascending=False).index.tolist()
                mrank = [rank for rank in mrank if rank != lang and rank in lang_scripts]
                lang2mrank[lang] = mrank
                for feature, df_feature in feature2df.items():
                    if lang not in df_feature:
                        lang2frank[feature][lang] = []
                        continue
                    indexs = df_feature[lang].index.tolist()
                    values = df_feature[lang].values.tolist()
                    flist = {index: value for index, value in zip(indexs, values) if index in mrank}
                    flist = sorted(flist.items(), key=lambda x: -x[1])
                    frank = []
                    cur_value = -1
                    group = []
                    for index, value in flist:
                        if value == cur_value or cur_value == -1:
                            group.append(index)
                        else:
                            frank.append(group)
                            group = [index]
                        cur_value = value
                    frank.append(group)
                    lang2frank[feature][lang] = frank

            # Correlation between LM sim and Metadata sim
            for feature, df_feature in feature2df.items():
                # Get two sim lists for each langauge to compute correlation
                sim_lang_list1, sim_lang_list2 = {lang: [] for lang in lang_scripts}, {lang: [] for lang in lang_scripts}
                for lang1 in lang_scripts:
                    for lang2 in lang_scripts:
                        if lang2 <= lang1:
                            continue
                        if lang1 not in df_feature or lang2 not in df_feature:
                            continue
                        sim_lang_list1[lang1].append(df_layer[layer_id][lang1][lang2])
                        sim_lang_list2[lang1].append(df_feature[lang1][lang2])
                        sim_lang_list1[lang2].append(df_layer[layer_id][lang1][lang2])
                        sim_lang_list2[lang2].append(df_feature[lang1][lang2])

                coff_p_sum, coff_s_sum, coff_lang_cnt = 0, 0, 0
                for lang in lang_scripts:
                    mrank = lang2mrank[lang]
                    coff_p = pearsonr(sim_lang_list1[lang], sim_lang_list2[lang])
                    coff_s = spearmanr(sim_lang_list1[lang], sim_lang_list2[lang])
                    if not math.isnan(coff_p[0]):
                        coff_p_sum += coff_p[0]
                        coff_s_sum += coff_s[0]
                        coff_lang_cnt += 1
                        df_lang_p[lang][layer_id][feature] = coff_p[0]
                        df_lang_s[lang][layer_id][feature] = coff_s[0]
                        df_feature_p[feature][layer_id][lang] = coff_p[0]
                        df_feature_s[feature][layer_id][lang] = coff_s[0]

                coff_p = coff_p_sum / coff_lang_cnt
                coff_s = coff_s_sum / coff_lang_cnt
                df_p[layer_id][feature] = coff_p
                df_s[layer_id][feature] = coff_s

            # Hierarchical Clustering
            mat = (1 - df_layer[layer_id].fillna(1)).values
            dists = squareform(mat)
            linkage_matrix = linkage(dists, 'complete')
            dendro_width = len(lang_scripts)
            if dendro_width > 50:
                plt.figure(figsize=(16, dendro_width / 2))
                dendrogram(linkage_matrix, labels=df_layer[layer_id].columns.values.tolist(), orientation='left',
                           leaf_font_size=16)

                for line in plt.gca().get_lines():
                    line.set_linewidth(3)

            else:
                plt.figure(figsize=(dendro_width / 4, 4))
                dendrogram(linkage_matrix, labels=df_layer[layer_id].columns.values.tolist(), leaf_rotation=90,
                           leaf_font_size=16)
                for line in plt.gca().get_lines():
                    line.set_linewidth(3)

            tag = 'dendrogram_' + str(layer_id)
            # plt.title(tag)
            plt_path = self.save_path + tag + '.pdf'
            # plt.autoscale()
            plt.savefig(plt_path)
            plt.close()

        tsv_path = self.save_path + 'coff_p.tsv'
        df_p.to_csv(tsv_path, sep='\t')
        for feature in features:
            df_feature_p[feature]['mean'] = df_feature_p[feature].mean(axis=1)
            df_feature_p[feature]['max'] = df_feature_p[feature].max(axis=1)
            df_feature_p[feature].to_csv(tsv_path, sep='\t')
            mean = df_feature_p[feature][df_feature_p[feature]['max'] != 0]['max'].mean()
            median = df_feature_p[feature][df_feature_p[feature]['max'] != 0]['max'].median()
            logging.info('Pearson of %s: mean %s, median %s' % (feature, mean, median))

        tsv_path = self.save_path + 'coff_s.tsv'
        df_s.to_csv(tsv_path, sep='\t')
        for feature in features:
            df_feature_s[feature]['mean'] = df_feature_s[feature].mean(axis=1)
            df_feature_s[feature]['max'] = df_feature_s[feature].max(axis=1)
            df_feature_s[feature].to_csv(tsv_path, sep='\t')
            mean = df_feature_s[feature][df_feature_s[feature]['max'] != 0]['max'].mean()
            median = df_feature_s[feature][df_feature_s[feature]['max'] != 0]['max'].median()
            logging.info('Spearman of %s: mean %s, median %s' % (feature, mean, median))

