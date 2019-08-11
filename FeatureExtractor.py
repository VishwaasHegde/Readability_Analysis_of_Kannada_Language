import re
import pickle
dir_path = 'E:\\Vishwaas\\Kannada'
import pandas as pd
import numpy as np
import csv
from collections import defaultdict
import math
import operator
r1='[ಅ-ಹ]'
w2v_file='supporting_data_files\\wiki.kn.vec'

word_count_file='supporting_data_files\\word_count.csv'


lex_file = 'supporting_data_files\\kannada.lex'
prefix_file = 'supporting_data_files\\prefix.pickle'
suffix_file = 'supporting_data_files\\suffix.pickle'
word_cnt={}
prefix = {}
suffix = {}
word_cnt_pickle = 'suporting_data_files\\word_cnt.pickle'

s_map = None
p_map = None



class FeatureExtractor:
    df = None

    def __init__(self, df):
        self.df = df

    def word_ott_avg_cnt(self, sent):
        chrs = len(re.findall(r1, sent))
        ottakshara = sent.count('್')
        N = len(sent.split(' '))
        return (chrs-ottakshara)/N, ottakshara/N, N

    def add_features(self):
        df=self.df
        A=[]
        B=[]
        C=[]
        for sent in df['message'].values:
            a,b,c = self.word_ott_avg_cnt(sent)
            A.append(a)
            B.append(b)
            C.append(c)
        df['syllable']=A
        df['ottakshara']=B
        df['word_count']=C
        self.df=df



    def get_word2vec_stem(self):
        word_emb = pd.read_csv(w2v_file, delimiter=' ', encoding="utf-8", quoting=csv.QUOTE_NONE, header=None)
        word_emb = word_emb.set_index([0])
        for folder in [4]:
            folder=str(folder)
            file_k = dir_path+'\\Classes\\'+folder + '\\g_stem.csv'
            df= pd.read_csv(file_k)
            avg_dp=[]
            for mes in df['stem_words']:
                words = mes.split(',')
                if len(words)==1:
                    avg_dp.append(0)
                    continue
                s=0
                for i in range(len(words)-1):
                    if words[i] in word_emb.index and words[i+1] in word_emb.index:
                        s+=np.dot(word_emb.loc[words[i], :][0:300],word_emb.loc[words[i+1], :][0:300])

                avg_dp.append(s/(len(words)-1))
            df['avg_dp_stem']=avg_dp
            df.to_csv(dir_path + '\\Classes\\' + folder + '\\g_stem.csv', index=False)

    def get_word_dist_stem(self):
        word_emb = pd.read_csv(w2v_file, delimiter=' ', encoding="utf-8", quoting=csv.QUOTE_NONE, header=None)
        word_emb = word_emb.set_index([0])
        for folder in [1,2,3,5,6,10]:
            folder=str(folder)
            file_k = dir_path+'\\Classes\\'+folder + '\\g_stem.csv'
            df= pd.read_csv(file_k)
            avg_dp=[]
            for mes in df['stem_words']:
                words = mes.split(',')
                if len(words)==1:
                    avg_dp.append(0)
                    continue
                s=0
                for i in range(len(words)-1):
                    if words[i] in word_emb.index and words[i+1] in word_emb.index:
                        s+= self.get_eucl_dist(word_emb.loc[words[i], :][0:300],word_emb.loc[words[i+1], :][0:300])

                avg_dp.append(s/(len(words)-1))
            df['avg_dist_stem']=avg_dp
            df.to_csv(dir_path + '\\Classes\\' + folder + '\\g_stem.csv', index=False)

    def  get_word2vec(self):
        with open('supporting_data_files\word2vec_map.pickle', 'rb') as handle:
            word_emb = pickle.load(handle)
        df=self.df
        avg_dp=[]
        avg_dist=[]
        for mes in df['stem_words']:
            words = mes.split(',')
            if len(words) == 1:
                avg_dp.append(0)
                avg_dist.append(0)
                continue
            s=0
            t=0
            for i in range(len(words)-1):
                if words[i] in word_emb and words[i+1] in word_emb:
                    x=np.array(word_emb[words[i]])
                    y=np.array(word_emb[words[i+1]])
                    s += np.dot(x,y)
                    t += self.get_eucl_dist(x, y)
            avg_dp.append(s/(len(words)-1))
            avg_dist.append(t / (len(words) - 1))
        df['avg_dp']=avg_dp
        df['avg_dist'] = avg_dist
        self.df=df

    word_vec_map={}

    def build_word_vec_map(self):
        word_emb = pd.read_csv(w2v_file, delimiter=' ', encoding="utf-8", quoting=csv.QUOTE_NONE, header=None)
        word_emb = word_emb.set_index([0])

        for ind, row in word_emb.iterrows():
            self.word_vec_map[ind]=list(row[0:300])

        with open('supporting_data_files\word2vec_map.pickle', 'wb') as handle:
            pickle.dump(self.word_vec_map, handle, protocol = pickle.HIGHEST_PROTOCOL)


    def get_word_dist(self):

        with open('supporting_data_files\word2vec_map.pickle', 'rb') as handle:
            word_emb = pickle.load(handle)
        df=self.df
        avg_dp=[]
        j=0
        for mes in df['message'].values:
            words = mes.split(' ')
            if len(words) == 1:
                avg_dp.append(0)
                continue
            s=0
            for i in range(len(words)-1):
                if words[i] in word_emb and words[i+1] in word_emb:
                    s+= self.get_eucl_dist(word_emb[words[i]][0:300],word_emb[words[i+1]][0:300])
            avg_dp.append(s/(len(words)-1))
        df['avg_dist']=avg_dp
        self.df=df


    def get_eucl_dist(self, a, b):
        d = np.array(a)-np.array(b)
        return np.sqrt(np.sum(d**2))


    def get_local_word_cnt(self):
        df=self.df
        word_cnt={}
        for sent in df['message'].values:
            words = sent.split(' ')
            for word in words:
                if word in word_cnt:
                    word_cnt[word]+=1
                else:
                    word_cnt[word]= 1

        with open(word_count_file, "w", encoding="utf-8") as f:
            for w in sorted(word_cnt, key=word_cnt.get, reverse=True):
                f.write(str(w) + '\t' + str(word_cnt[w]) + '\n')

        self.df = df

    def remove_outliers(self):
        Ts = []
        df=self.df
        for a in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]:
            T = df[df['target'] == a]
            T = T[(np.abs(T.word_count - T.word_count.mean()) <= (3 * T.word_count.std()))]
            Ts.append(T)
        T = pd.concat(Ts).reset_index()
        self.df=T

    def process_word(self,word, c):
        for i in range(3, len(word)):
            prefix_str = word[0:i]
            suffix_str = word[i: len(word)]

            if prefix_str in prefix:
                prefix[prefix_str] +=  c
            else:
                prefix[prefix_str] = c

            if suffix_str in suffix:
                suffix[suffix_str] += c
            else:
                suffix[suffix_str] = c

    def get_pos_inde_dict(self):
        pos_set = set()
        pos_dict = {}
        word_pos = {}
        pos_cnt=defaultdict(int)
        with open(lex_file, encoding = "utf8", errors = 'ignore') as f:
            for line in f:
                split = line.split('\t')
                w = split[0].strip()
                pos = split[2].strip()
                pos = pos[:pos.find('.')]
                pos_set.add(pos)
                word_pos[w] = pos
                pos_cnt[pos]+=1
        for i,a in enumerate(pos_set):
            pos_dict[a] = i
        sorted_x = sorted(pos_cnt.items(), key=operator.itemgetter(1))
        return pos_dict, word_pos

    def build_pos_cols(self):
        df= self.df
        pos_dict, word_pos = self.get_pos_inde_dict()

        pos_sums = []
        NNl = []
        VMl= []
        NNPl = []
        JJl = []
        RBl = []
        for sent in df['message']:
            words = sent.split(' ')
            pos_sum = 0
            NN=0
            VM=0
            NNP=0
            JJ=0
            RB=0
            for w in words:
                if w in word_pos:
                    pos = word_pos[w]
                    if pos=='NN':
                        NN+=1
                    elif pos=='VM':
                        VM+=1
                    elif pos == 'NNP':
                        NNP+=1
                    elif pos == 'JJ':
                        JJ+=1
                    elif pos == 'RB':
                        RB+=1
                    pos_sum += pos_dict[word_pos[w]]
            pos_sums.append(pos_sum/len(words))
            NNl.append(NN/len(words))
            VMl.append(VM/len(words))
            NNPl.append(NNP/len(words))
            JJl.append(JJ/len(words))
            RBl.append(RB/len(words))

        df['pos_cnt_avg'] = pos_sums
        df['NN'] = NNl
        df['VM'] = VMl
        df['NNP'] = NNPl
        df['JJ'] = JJl
        df['RB'] = RBl
        # pos_sums = []
        # for sent in df['stem_words']:
        #     words = sent.split(',')
        #     pos_sum = 0
        #     for w in words:
        #         if w in word_pos:
        #             pos_sum += pos_dict[word_pos[w]]
        #     pos_sums.append(pos_sum/len(words))
        # df['pos_cnt_avg_stem'] = pos_sums
        self.df=df

    def get_pre_suf(self):
        try:
            my_file = open(prefix_file)
            my_file = open(suffix_file)
        except IOError:
            with open(lex_file, encoding = "utf8", errors = 'ignore') as f:
                for line in f:
                    split = line.split('\t')
                    word, cnt = split[0].strip(), int(split[1].strip())
                    if len(word) < 3:
                        if word in prefix:
                            prefix[word] +=  cnt
                        else:
                            prefix[word] = cnt
                        continue
                    self.process_word(word, cnt)

            with open(prefix_file, 'wb') as handle:
                pickle.dump(prefix, handle, protocol = pickle.HIGHEST_PROTOCOL)

            with open(suffix_file, 'wb') as handle:
                pickle.dump(suffix, handle, protocol = pickle.HIGHEST_PROTOCOL)


    def p_score_sent(self, sent):
        if not isinstance(sent, str):
            return 'nan'
        words = sent.split(' ')

        global s_map
        global p_map
        if s_map is None:
            with open(suffix_file, 'rb') as handle:
                s_map = pickle.load(handle)

        if p_map is None:
            with open(prefix_file, 'rb') as handle:
                p_map = pickle.load(handle)

        root_words = []
        pss = 0
        root_cnt = 0
        word_cnt = 0
        N = len(words)
        for w in words:
            root, p_s = self.p_score_word(w, s_map, p_map)
            root_words.append(str(root))
            pss += p_s
            if root in p_map:
                root_cnt+= p_map[root]
            if w in p_map:
                word_cnt +=  p_map[w]

        return ','.join(root_words), pss/N, root_cnt/N, word_cnt/N


    def p_score_word(self,word, s_map, p_map):
        p_s = 0

        root = word
        if len(word) < 4:
            if root in p_map:
                return root, len(root)*math.log10(p_map[root])
            else:
                return root,0
        for i in range(3, len(word)):
            prefix_str = word[0:i]
            suffix_str = word[i: len(word)]
            if prefix_str in p_map:
                p_c = p_map[prefix_str]
            else:
                p_c = 0
            if suffix_str in s_map:
                s_c = s_map[suffix_str]
            else:
                s_c = 0
            if p_c > 0 and s_c > 0:
                p_s_temp = self.p_score(prefix_str, p_c, suffix_str, s_c)
                if p_s_temp > p_s:
                    p_s = p_s_temp
                    root = prefix_str
                    best_p=prefix_str
                    best_s=suffix_str

        return root, p_s

    r7 = '[ಅ-ಹ]'
    def count_chrs(self, word):
        chrs = len(re.findall(self.r7, word))
        ottakshara = word.count('್')
        return chrs-ottakshara


    def chr_cnt(self, word):
        p_len = len(re.findall(r1, word))
        p_ottakshara = word.count('್')

        return p_len-p_ottakshara

    def p_score(self, p, p_c, s, s_c):
        # if type = =  1:
        #     return chr_cnt(p)*math.log10(p_c)+chr_cnt(s)*math.log10(s_c)
        return len(p)*math.log(p_c)+len(s)*math.log(s_c)

    def stem_gadya(self):
        df=self.df
        self.get_pre_suf()
        i = 0
        sent_1 = []
        pss_avg_1 = []
        root_cnt_avg_1 = []
        word_cnt_avg_1 = []
        for sent in df['message']:
            root_words, pss_avg, root_cnt_avg, word_cnt_avg = self.p_score_sent(sent)
            sent_1.append(root_words)
            pss_avg_1.append(pss_avg)
            root_cnt_avg_1.append(root_cnt_avg)
            word_cnt_avg_1.append(word_cnt_avg)
            i += 1
        df['stem_words'] = sent_1
        df['pss_avg'] = pss_avg_1
        df['root_cnt_avg'] = root_cnt_avg_1
        df['word_cnt_avg'] = word_cnt_avg_1
        # del df['pc_avg']
        # df.to_csv(dir_path+'\\Classes\\'+folder + '\\g_stem.csv',index = False)
        self.df=df
        # df.to_csv(file_k)


    def get_sw(self):
        sw = set()
        with open('E:\\Vishwaas\\Kannada\\kannada_stopwords.txt', encoding = "utf8") as f:
            for line in f:
                a = line.replace('\n','')
                sw.add(a)
        return sw

    def cnt_remove_sw(self):
        df=self.df
        sents = []
        swcs = []
        sw=self.get_sw()
        for sent in df['message']:
            words = []
            swc = 0

            for w in sent.split(' '):
                if w in sw:
                    swc += 1
                else:
                    words.append(w)
            if len(words) == 0:
                sents.append('')
            else:
                sents.append(' '.join(words))
            swcs.append(swc/(len(words)+1))

        df['message_sw'] = sents
        df['swcs'] = swcs
        self.df=df

    def build_word_cnt_map(self):
        try:
            my_file = open(lex_file)
        except IOError:
            with open(lex_file, encoding="utf8", errors='ignore') as f:
                for line in f:
                    split = line.split('\t')
                    w=split[0]
                    cnt=split[1]
                    word_cnt[w]=cnt

            with open(word_cnt_pickle, 'wb') as handle:
                pickle.dump(word_cnt, handle, protocol = pickle.HIGHEST_PROTOCOL)

    def word_ott_avg_cnt(self, sent):
        chrs = len(re.findall(r1, sent))
        ottakshara = sent.count('್')
        N = len(sent.split(' '))
        return (chrs-ottakshara)/N, ottakshara/N, N


    def get_word_cnt(self):
        df=self.df
        with open(word_cnt_pickle, 'rb') as handle:
            word_cnt_dict = pickle.load(handle)
        word_cnt=[]
        for sent in df['message_sw']:
            words = sent.split(' ')
            cnt=0
            for w in words:
                if w in word_cnt_dict:
                    c = int(word_cnt_dict[w])
                    if c>0:
                        cnt+=np.log(c)
            word_cnt.append(cnt/len(words))
        df['word_cnt_avg']=word_cnt
        self.df=df

    def get_hard_word_thr_cnt(self):
        df=self.df
        opt_thr=0
        corr=0
        target= np.array(df['target'])
        for thr in range(2,15):
            word_cnt=[]
            for sent in df['message']:
                words = sent.split(' ')
                cnt=0
                for w in words:
                    c= self.count_chrs(w)
                    if c>thr:
                        cnt+=1
                word_cnt.append(cnt/len(words))
            corr_curr= np.corrcoef(np.array(word_cnt), target)[0][1]
            if corr_curr>corr:
                corr=corr_curr
                opt_thr=thr
                print(corr)
        print('optimal hard word threshold: '+str(opt_thr))
        self.df = df

    def get_hard_word_cnt(self):
        df=self.df
        word_cnt=[]
        for sent in df['message']:
            words = sent.split(' ')
            cnt=0
            for w in words:
                c = self.count_chrs(w)
                if c > 4:
                    cnt += 1
            word_cnt.append(cnt/len(words))
        df['hard_word_cnt_avg']=word_cnt
        self.df=df

    def get_df(self):
        return self.df


# combine_all_data()
# add_features()
#remove_outliers()
# get_local_word_cnt()
# get_word2vec()
# get_word2vec()
# get_word_dist()
# build_word_vec_map()
# sample()
# get_word2vec()
