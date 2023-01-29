

# import 

import networkx as nx
import random
import numpy as np
from typing import List
import tqdm
from gensim.models.word2vec import Word2Vec

import matplotlib.pyplot as plt

from IPython.display import display
from PIL import image 

# encoding:utf-8
import sys

sys.path.append("..")
from mf import MF
from reader.trust import TrustGetter
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp, cosine_sp, jacc_sp
from utility import util


import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

class SocialReg_NV(MF):
    """
    docstring for SocialReg

    Ma H, Zhou D, Liu C, et al. Recommender systems with social regularization[C]//Proceedings of the fourth ACM international conference on Web search and data mining. ACM, 2011: 287-296.
    """

    def __init__(self):
        super(SocialReg_NV, self).__init__()
        # self.config.lambdaP = 0.001
        # self.config.lambdaQ = 0.001
        self.config.alpha = 0.1
        self.tg = TrustGetter()
        
        # self.init_model()

    def init_model(self, k):
        super(SocialReg_NV, self).init_model(k)
        from collections import defaultdict
        #self.user_sim = SimMatrix()
        self.user_sim = self.get_n2V_emb(0.25,1,10,180)
        print('constructing user-user similarity matrix...')

        # self.user_sim = util.load_data('../data/sim/ft_cf_soreg08_cv1.pkl')

        #for u in self.rg.user:
        #    for f in self.tg.get_followees(u):
        #        if self.user_sim.contains(u, f):
        #            continue
        #        sim = self.get_sim(u, f)
        #        self.user_sim.set(u, f, sim)

        # util.save_data(self.user_sim,'../data/sim/ft_cf_soreg08.pkl')
    def get_n2V_emb(self,p=0.25,q=1,window_size=10,embedding_vector_size=180):

        G = self.create_graphe("./data/epinion_ratings.txt")
        from collections import defaultdict 
        probs = defaultdict(dict)
        for node in G.nodes():
            probs[node]['probabilities'] = dict()
        
        cp = self.compute_probabilities(G,probs,p,q)
        walks = self.generate_random_walks(G,cp,5,61028)
        n2v_emb = self.Node2Vec(walks,window_size,embedding_vector_size)
        

        return n2v_emb.vectors
    
    def create_graphe(self,file_path):

        nx.write_edgelist(nx.path_graph(4), "tmp.edgelist")
        G = nx.read_edgelist("tmp.edgelist")
        i = 0
        with open(file_path,'r') as f,  open("tmp.edgelist", "w") as fh:
            
            for textline in f:
                if i > 0:
                    fh.write(textline)
                i+=1
            fh.close()
            
        G = nx.read_edgelist("tmp.edgelist", nodetype=int, data=(("weight", float),))
        return G

    def compute_probabilities(self,graph,probs,p,q):

        G = graph
        for source_node in G.nodes():
            for current_node in G.neighbors(source_node):
                probs_ = list()
                for destination in G.neighbors(current_node):
                    if source_node == destination:
                        prob_ = G[current_node][destination].get('weight',1)*(1/p)
                    elif destination in G.neighbors(source_node):
                        prob_ = G[current_node][destination].get('weight',1)
                    else:
                        prob_ = G[current_node][destination].get('weight',1)*(1/q)
                    
                    probs_.append(prob_)
                
                probs[source_node]['probabilities'][current_node] = probs_ / np.sum(probs_)
        return probs


    def genrate_random_walks(self,graph,probs,max_walks,walk_len):
        G = graph
        walks = list()
        for start_node in G.nodes():
            for i in range(max_walks):
                walk = [start_node]
                walk_options = list(G[start_node])
                if len(walk_options) == 0:
                    break
                first_step = np.random.choice(walk_options)
                walk.append(first_step)
                for k in range(walk_len-2):
                    walk_options = list(G[walk[-1]])
                    if len(walk_options) == 0:
                        break
                    probabilities = probs[walk[-2]]['probabilities'][walk][-1]
                    next_step = np.random.choice(walk_options,p=probabilities)
                    walk.append(next_step)
                walks.append(walk)
            
            np.random.shuffle(walks)
            walks = [list(map(str,walk)) for walk in walks]
            return walks

    def Nod2Vec(self,generated_walks,window_size,embedding_vector_size):

        model = Word2Vec(sentences=generated_walks,window=window_size,vector_size=embedding_vector_size)

        return model.wv

    def get_sim(self, u, k):

        sim_pearson = (pearson_sp(self.rg.get_row(u), self.rg.get_row(k)) + 1.0) / 2.0  # fit the value into range [0.0,1.0]
        print(type(sim_pearson))
        return sim_pearson

    def train_model(self, k):
        super(SocialReg_NV, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                print(self.user_sim[user])
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += 0.5 * error ** 2
                p, q = self.P[u], self.Q[i]

                social_term_p, social_term_loss = np.zeros((self.config.factor)), 0.0
                followees = self.tg.get_followees(user)
                for followee in followees:
                    if self.rg.containsUser(followee):
                        s = self.user_sim[user][followee]
                        uf = self.P[self.rg.user[followee]]
                        social_term_p += s * (p - uf)
                        social_term_loss += s * ((p - uf).dot(p - uf))

                social_term_m = np.zeros((self.config.factor))
                followers = self.tg.get_followers(user)
                for follower in followers:
                    if self.rg.containsUser(follower):
                        s = self.user_sim[user][follower]
                        ug = self.P[self.rg.user[follower]]
                        social_term_m += s * (p - ug)

                # update latent vectors
                self.P[u] += self.config.lr * (
                        error * q - self.config.alpha * (social_term_p + social_term_m) - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

                self.loss += 0.5 * self.config.alpha * social_term_loss

            self.loss += 0.5 * self.config.lambdaP * (self.P * self.P).sum() + 0.5 * self.config.lambdaQ * (
                    self.Q * self.Q).sum()

            iteration += 1
            if self.isConverged(iteration):
                break


if __name__ == '__main__':
    # srg = SocialReg()
    # srg.train_model(0)
    # coldrmse = srg.predict_model_cold_users()
    # print('cold start user rmse is :' + str(coldrmse))
    # srg.show_rmse()
    
    rmses = []
    maes = []
    tcsr = SocialReg_NV()
    # print(bmf.rg.trainSet_u[1])
    for i in range(tcsr.config.k_fold_num):
        print('the %dth cross validation training' % i)
        tcsr.train_model(i)
        rmse, mae = tcsr.predict_model()
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / 5
    mae_avg = sum(maes) / 5
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)
    print("the average of rmses is %s " % rmse_avg)
    print("the average of maes is %s " % mae_avg)







