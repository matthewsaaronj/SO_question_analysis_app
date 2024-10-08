import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import string
import nltk
from nltk.corpus import stopwords


class GoogleQuestFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, featengineer):
        self.featengineer = featengineer
        

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):

        out = X.copy()

        eng_stopwords = set(stopwords.words('english'))

        # define functions for data processing-------------

        if self.featengineer == 'baseline':

            # count the words in the three input components (title, body, answer)
            for colname in ['question_title', 'question_body', 'answer']:
                newname = colname + '_word_len'

                out[newname] = out[colname].str.split().str.len()

            # check for nonames, i.e. users with logins like user124
            for colname in ['question', 'answer']:
                out['is_' + colname + '_no_name_user'] = out[colname + '_user_name'].str.contains('^user\d+$') + 0
                
            # check lexical diversity (unique words count vs total)
            # lexical diversity 
            out['answer_div'] = out['answer'].apply(lambda s: len(set(s.split())) / len(s.split()) )

            # domain components ------------------------------------
            out['domcom'] = out['question_user_page'].apply(lambda s: s.split('://')[1].split('/')[0].split('.'))
            # count components
            out['dom_cnt'] = out['domcom'].apply(lambda s: len(s))
            # pad the length in case some domains have fewer components in the name
            out['domcom'] = out['domcom'].apply(lambda s: s + ['none', 'none'])

            # components
            for ii in range(0,4):
                out['dom_'+str(ii)] = out['domcom'].apply(lambda s: s[ii])

            # clean up
            out.drop('domcom', axis = 1, inplace = True)
            # --------------------------------------------------------

            # shared elements ----------------------------------------
            out['q_words'] = out['question_body'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )
            out['a_words'] = out['answer'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )
            out['qa_word_overlap'] = out.apply(lambda s: len(np.intersect1d(s['q_words'], s['a_words'])), axis = 1)
            out['qa_word_overlap_norm1'] = out.apply(lambda s: s['qa_word_overlap']/(1 + len(s['a_words'])), axis = 1)
            out['qa_word_overlap_norm2'] = out.apply(lambda s: s['qa_word_overlap']/(1 + len(s['q_words'])), axis = 1)
            out.drop(['q_words', 'a_words'], axis = 1, inplace = True)
            # -----------------------------------------------------------

                ## Number of characters in the text ##
            out["question_title_num_chars"] = out["question_title"].apply(lambda x: len(str(x)))
            out["question_body_num_chars"] = out["question_body"].apply(lambda x: len(str(x)))
            out["answer_num_chars"] = out["answer"].apply(lambda x: len(str(x)))

            ## Number of stopwords in the text ##
            out["question_title_num_stopwords"] = out["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
            out["question_body_num_stopwords"] = out["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
            out["answer_num_stopwords"] = out["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

            ## Number of punctuations in the text ##
            out["question_title_num_punctuations"] =out['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
            out["question_body_num_punctuations"] =out['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
            out["answer_num_punctuations"] =out['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

            ## Number of title case words in the text ##
            out["question_title_num_words_upper"] = out["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
            out["question_body_num_words_upper"] = out["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
            out["answer_num_words_upper"] = out["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

        return out