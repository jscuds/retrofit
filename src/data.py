"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from pathlib import Path #js jscud
import pickle
import sys
import warnings #js jscud
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning) #js jscud
#sys.path.append('../../bilm_tf/bilm') #js jscud  commented out
cwd = str(Path(__file__).resolve().parent.parent.parent) #js jscud 
#https://stackoverflow.com/questions/30218802/get-parent-of-current-directory-from-python-script
sys.path.append(f'{cwd}/bilm-tf') #js jscud #print(sys.path) #js jscud
#print(sys.path)
#print(cwd)

from bilm import TokenBatcher
import random
import re
from datetime import datetime
import numpy as np


class Data(object):
    # member variables like dictionaries and lists goes here
    def __init__(self, length = 0, use_synonym=False):
        self.para_tuples = [] # [(sent_id, sent_id, index_of_an_overlapping/synonym_token, index_of_an_overlapping/synonym_token), ... ]
        self.neg_tuples = [] # [(sent_id, sent_id, index_of_an_overlapping/synonym_token, index_of_an_overlapping/synonym_token), ... ]
        self.token_pair2neg_tuples = {} # {(token_id\, token_id) : set([neg_tuple_id, ...])}
        self.id2sent = [] # a list of arrays, where each array is a list of token ids (which represent a sentence). # eventually, make this an numpy array
        self.sent2id = {}
        self.paraphrases = set([]) # a set of {(sent_id, sent_id), ...} to quickly check whether two sentences are paraphrases or not.
        self.token2sents = {} # reverse index of sentences given tokens. This is a map { token_id : set([(sent_id, index_of_the_token_in_the_sentence), ...]) }.
        self.synonyms = {} # {token_id : set([token_id, ... ])}
        self.use_synonym = use_synonym
        self.stop_word_ids = set([])
        self.length = length
        # self.batch_sizeK = None # To be readed by tester

        # build token_batcher
        self.word2id = {}
        self.id2word = []


    def build(self, vocab_file, stop_word_file, synonym_file=None):
        # 1. build TokenBatcher
        self.token_batcher = TokenBatcher(vocab_file)
        self.word2id = self.token_batcher._lm_vocab._word_to_id
        self.id2word = self.token_batcher._lm_vocab._id_to_word
        # 2. if synonym_file is not None, populate synonyms (two directions).
        with open(synonym_file, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                if(line[0] in self.word2id and line[2] in self.word2id):
                    id0 = self.word2id[line[0]]
                    id1 = self.word2id[line[2]]
                    if(id1 == id0):
                        continue
                    self.synonyms.setdefault(id0, set()).add(id1)
                    self.synonyms.setdefault(id1, set()).add(id0)

        # 3. if stop_word_file is not None, populate stop_word_ids
        with open(stop_word_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line in self.word2id:
                    self.stop_word_ids.add(self.word2id[line])


    # The dataset is formatted as sentence\tsentence\tlabel
    def load_sentece_pairs(self, data_file_list, bad_words, data_type_list):
        # 1. populate sentence_tuples, update sentences (check stop_word_ids), paraphrases, token2sents.
        # 2. populate negative cases of sentence pairs into neg_tuples, and correspondingly update token2neg_tuples, sentences (check stop_word_ids), token2sents.
        s_len = []
        for data_file, data_type in zip(data_file_list,data_type_list):
            with open(data_file, "rt", encoding="utf-8") as f:
                count = 0
                for line in f:
                    count += 1
                    if (count >= 20000):        #js they just stop after grabbing 20k samples.
                        break
                    line = line.strip().split('\t')
                    label = line[0]
                    if(data_type == "mrpc"):
                        s1 = line[3].split()
                        s2 = line[4].split()
                    else:
                        s1 = line[1].split()
                        s2 = line[2].split()

                    exist_bad_word = False
                    for i in bad_words:
                        if (i in s1 or i in s2):
                            exist_bad_word = True
                    if (exist_bad_word == True):
                        continue

                    # s1_tokenid = self.token_batcher.batch_sentences([s1])[0][1:][:-1]
                    # s2_tokenid = self.token_batcher.batch_sentences([s2])[0][1:][:-1]

                    # 1
                    s1_tokenid = self.token_batcher.batch_sentences([s1])[0]  #js s#_tokenid is the tokenized sentence into token_ids
                    s2_tokenid = self.token_batcher.batch_sentences([s2])[0]


                    # zero-pad/ truncate sentences to self.length
                    #check if len(s1) > self.length
                    s_len.append(len(s1_tokenid))
                    s_len.append(len(s2_tokenid))
                    if(len(s1_tokenid) > self.length or len(s1_tokenid) < 3):
                        print(s1_tokenid, s1)
                        continue
                    if (len(s2_tokenid) > self.length or len(s2_tokenid) < 3):
                        print(s2_tokenid, s2)
                        continue

                    if len(s1_tokenid) > self.length:
                        s1_tokenid = s1_tokenid[:self.length]
                    else:
                        s1_tokenid = np.pad(s1_tokenid, (0, self.length - len(s1_tokenid)), 'constant', constant_values=(0))
                    if len(s2_tokenid) > self.length:
                        s2_tokenid = s2_tokenid[:self.length]
                    else:
                        s2_tokenid = np.pad(s2_tokenid, (0, self.length - len(s2_tokenid)), 'constant', constant_values=(0))

                    #js add to the sent2id dict and reset s1_id and s2_ids to new index values.
                    #   sent2id = dict(k:v) = dict(tuple(tok1, tok2, tok3...) : s1_id)
                    if not (tuple(s1_tokenid) in self.sent2id): #js checks if the list of token_ids is in the sent2id structure.
                        self.id2sent.append(s1_tokenid)
                        s1_id = len(self.id2sent) - 1       #js they just increment a counter; RESET qid indices so they match their relative index position in the `self.id2sent`` LIST
                        self.sent2id.update({tuple(s1_tokenid): s1_id})
                    else:
                        s1_id = self.sent2id[tuple(s1_tokenid)]
                    if not (tuple(s2_tokenid) in self.sent2id):
                        self.id2sent.append(s2_tokenid)
                        s2_id = len(self.id2sent) - 1
                        self.sent2id.update({tuple(s2_tokenid): s2_id})
                    else:
                        s2_id = self.sent2id[tuple(s2_tokenid)]

                    #update paraphrases, para_tuples, neg_tuples
                    overlap_index_pairs, synonym_index_pairs = self.overlap(s1_tokenid,s2_tokenid)
                    # print(s1_tokenid)
                    # print(s2_tokenid)
                    # print("overlap", overlap_index_pairs)
                    # if synonym_index_pairs:
                    #     print("synonym_index_pairs", synonym_index_pairs)
                    total_index_pairs = overlap_index_pairs + synonym_index_pairs #js combined list of synonym indices and overlap indices
                    if (label == "1"):
                        self.paraphrases.add((s1_id, s2_id))    #js if `is_duplicate`==1; append both combinations s1&s2 ++ s2&s1 to `self.paraphrases`
                        self.paraphrases.add((s2_id, s1_id))
                        for p in total_index_pairs:             #js for every pair of overlap words:
                            sent_tuple = (s1_id,s2_id,p[0],p[1]) #js (s1_idx, s2_idx, s1_overlap_word_index, s2_overlap_word_idx) **SO THERE COULD BE MULTIPLE TUPLES STARTING WITH  `s1_idx, s2_idx`** if they have multiple overlapping words
                            self.para_tuples.append(sent_tuple) #js a list of tuples in the format from the line above
                    else: #js this from NON-PARAPHRASE questions
                        for p in total_index_pairs:
                            sent_tuple = (s1_id, s2_id, p[0], p[1])
                            self.neg_tuples.append(sent_tuple)
                            w1 = s1_tokenid[p[0]]
                            w2 = s2_tokenid[p[1]]
                            if w1 in self.stop_word_ids or w2 in self.stop_word_ids: #js filter out stopwords from negative sentences
                                continue
                            self.token_pair2neg_tuples.setdefault((w1, w2), set()).add(len(self.neg_tuples)-1)
                            #js ^ dict((token_id\, token_id) : set([neg_tuple_id, ...]))
                            #   `.setdefault()` adds the (w1,w2) key if it's not present and if the (w1,w2) key _is_ present, `.setdefault()`
                            #   will add the new index of the neg_tuple id to the set of dictionary values
                            #   EX. a = {(52, 53): {4}}
                            #       a.setdefault((20,21),set()).add(3);  a == {(20, 21): {3}, (52, 53): {4}}
                            #       a.setdefault((20,21),set()).add(5);  a == {(20, 21): {3, 5}, (52, 53): {4}}
                            #


                    # update token2sents
                    # js token2sents is a dict {token_id: {(updated_s1_id, new_index)}} updated_s1_id from ~line 132; updated index from enumerating for-loop in line 176    
                    #   *NOTE* `index` is the index of the word in the tokenized sentence (`s1_tokenid`) 
                    for index,token_id in enumerate(s1_tokenid):
                        if(token_id == 2 or token_id == 1):
                            continue
                        sid_index = (s1_id, index)
                        self.token2sents.setdefault(token_id, set()).add(sid_index) 
                    for index,token_id in enumerate(s2_tokenid):
                        if (token_id == 2 or token_id == 1):
                            continue
                        sid_index = (s2_id, index)
                        self.token2sents.setdefault(token_id, set()).add(sid_index)
        self.neg_tuples, self.para_tuples, self.id2sent = np.array(self.neg_tuples), np.array(self.para_tuples), np.array(self.id2sent)
        s_len = np.array(s_len)
        print("s length", np.min(s_len), np.max(s_len), np.mean(s_len), np.median(s_len)) #js s_len is the list of ints representing the sentences lengths

    def overlap(self, s1, s2):      #js this returns a list of pairs of the *INDICES* of the same word between sentences.  
                                    #   Ex. "I love life" + "Love life" -> [[2,1],[3,2]] ; 
                                    #   indices _seem_ to be off by 1 because the indices account for the [CLS] and [SEP] tokens
        # check intersection
        s1_dict = dict((k, i) for i, k in enumerate(s1))
        s2_dict = dict((k, i) for i, k in enumerate(s2))
        word_pairs = []
        inter = set(s1_dict).intersection(set(s2_dict)) #js this finds all the common words between sentences; then removes token_ids for </S>, <S>, <UNK>
        if(1 in inter):
            inter.remove(1)
        if(2 in inter):
            inter.remove(2)
        if (0 in inter):
            inter.remove(0)
        inter.difference_update(self.stop_word_ids) #js removes all token_ids of stopwords
        # check digit                   
        for i in inter.copy():                      #js convert back to strings; remove digits and words beginning with '-'
            if (self.id2word[i].isdigit()):
                inter.remove(i)
            if (self.id2word[i].startswith('-')):
                inter.remove(i)
        for w in inter:
            w1_id = s1_dict[w]
            w2_id = s2_dict[w]
            word_pairs.append([w1_id, w2_id])

        synonym_pairs = []
        if self.use_synonym:
            for id in s1_dict.keys():
                if id in self.synonyms:
                    for s in self.synonyms[id]:
                        if s in s2_dict.keys():
                            synonym_pairs.append((s1_dict[id], s2_dict[s]))
            # print(synonym_pairs)
            for id in s2_dict.keys():
                if id in self.synonyms:
                    for s in self.synonyms[id]:
                        if s in s1_dict.keys():
                            synonym_pairs.append((s1_dict[s], s2_dict[id]))
            # print(synonym_pairs)
            # print("------")
        synonym_pairs = list(set(synonym_pairs))
        return word_pairs, synonym_pairs
    
    def corrupt(self, para_tuple, tar=None):    #js Return (sent_id, sent_id, index_of_an_overlapping/synonym_token, index_of_an_overlapping/synonym_token) for a negative sample.
        # corrupt para tuple into a negative sample. Return (sent_id, sent_id, index_of_an_overlapping/synonym_token, index_of_an_overlapping/synonym_token) for a negative sample.
        if tar == None:                     
            tar = random.randint(0,1)       #js randomly 0 or 1 to pick whether or 'corrupt' s1_id or s2_id
        s1 = para_tuple[0]                  #js assigns appropriate variable to the elements of `para_tuple = (s1_idx, s2_idx, s1_overlap_word_index, s2_overlap_word_idx)`
        s1_index = para_tuple[2]
        s2 = para_tuple[1]
        s2_index = para_tuple[3]

        if(tar == 0):   #js corrupt s1
            token = self.id2sent[s1][s1_index]      #js token_id of ONE OVERLAP WORD; INDEXING: finds s1_id in `self.id2sent` array, then uses the overlap word index to get the token_id of that word
            sents_list = self.token2sents[token]    #js uses `token` to get the *set* of {(sent_id, index)} from `token2sents` dict.

            if ((s1, s1_index) in sents_list):      #js remove s1 and s2 out of the set of possible sentences with the target word
                sents_list.remove((s1, s1_index))
            if ((s2, s2_index) in sents_list):
                sents_list.remove((s2, s2_index))
            if (len(sents_list) == 0):                  #js if the only overlap was between the two paraphrase sentences; *RANDOMLY RETURN A NEGATIVE TUPLE??* 
                return random.choice(self.neg_tuples)   #js TODO there has to be a better way
            else:
                corrupt_s = random.choice(list(sents_list)) #js find a sentence in the remaining set of overlap sentences (after casting to a list)
            ind = 0
            while self.is_paraphrase(corrupt_s[0], s1):     #js a 2nd check to see if the randomly selected set of overlap setneces is a set of paraphrases
                corrupt_s = random.choice(list(sents_list)) #   if the randomly selected sentence is a paraphrase, try 10 times to find a non-paraphrase
                ind += 1
                if ind > 10:
                    # print("ind", ind)
                    random.choice(self.neg_tuples)          #js if there isn't a non-paraphrase: ranomdly return a negative tuple:
                    break                                   #   TODO: `.remove()` the paraphrased sentence pair if `.is_paraphrase() = True` and find overlap with negative tuple?
            return (corrupt_s[0],s1,corrupt_s[1], s1_index)

        if(tar == 1):   #js corrupt s2 (SAME AS ABOVE)
            token = self.id2sent[s2][s2_index]
            sents_list = self.token2sents[token]

            if ((s1, s1_index) in sents_list):
                sents_list.remove((s1, s1_index))
            if ((s2, s2_index) in sents_list):
                sents_list.remove((s2, s2_index))
            if (len(sents_list) < 2):
                return random.choice(self.neg_tuples)
            else:
                corrupt_s = random.choice(list(sents_list))
            ind = 0
            while self.is_paraphrase(corrupt_s[0], s2):
                corrupt_s = random.choice(list(sents_list))
                ind += 1
                if ind > 10:
                    # print("ind", ind)
                    random.choice(self.neg_tuples)
                    break
            c_tuple = (corrupt_s[0],s2,corrupt_s[1],s2_index)
            return c_tuple
    
    def neg(self, para_tuple):  #js in trainer.py this seems to be used intstead of 
        s1 = para_tuple[0]
        s1_index = para_tuple[2]
        s2 = para_tuple[1]
        s2_index = para_tuple[3]
        s1_token = self.id2sent[s1][s1_index]
        s2_token = self.id2sent[s2][s2_index]
        if((s1_token, s2_token) in self.token_pair2neg_tuples):
            neg_tuple_id = random.choice(list(self.token_pair2neg_tuples[(s1_token, s2_token)]))
            neg_tuple = self.neg_tuples[neg_tuple_id]
            return neg_tuple
        else:
            return None

    #js corrput_n seems unused in trainer.py
    def corrupt_n(self, para_tuple, n=2):
        # in case we use logistic loss, use the corrupt function n times to generate and return n negative samples. Before each corruption, the random seed needs to be reset.
        corrupt_tuples = []
        for i in range(n):
            random.seed(datetime.now())
            corrupt_tuple = self.corrupt(para_tuple)
            if not corrupt_tuple:
                return None
            else:
                corrupt_tuples.append(corrupt_tuple)
        return corrupt_tuples

    #js is_synonym seems unused in trainer.py
    def is_synonym(self, token_id1, token_id2):
        if(token_id1 in self.synonyms(token_id2)):
            return True
        else:
            return False
    
    #js is_paraphrase used in `corrupt()`
    def is_paraphrase(self, sent_id1, sent_id2):
        if((sent_id1,sent_id2) in self.paraphrases):
            return True
        else:
            return False

    def save(self, filename):
        f = open(filename,'wb')
        #self.desc_embed = self.desc_embed_padded = None
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)

    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)
        print("===============\nCaution: need to reload desc embeddings.\n=====================")


vocab_file = "../bilm_tf/save/vocab-2016-09-10.txt"
stopword_file = "../data/stop_words_en.txt"
synonyms_file = "../data/synonyms.tsv"
bad_words = ["-LSB-", "\\", "``", "-LRB-", "????", "n/a", "'"]

# data_file = ["../data/driver"]
# mrpc = Data(30, use_synonym = False)
# data_type_list = ["quora"] 
# mrpc.build(vocab_file, stopword_file, synonyms_file)
# mrpc.load_sentece_pairs(data_file,"mrpc", data_type_list)
# mrpc.save("../data/driver-40-f.pk")

