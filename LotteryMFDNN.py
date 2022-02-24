
# coding: utf-8

# In[39]:


import os, sys, numpy as np, pandas as pd, tensorflow as tf, re, codecs, json, time
import pickle, collections, random, math, numbers, scipy.sparse as sp, itertools, shutil,pymysql
import datetime
import timeit
start = timeit.default_timer()
# 多核
from joblib import Parallel, delayed
import multiprocessing
#
import tensorflow as tf
def reload(mName):
    import importlib
    if mName in sys.modules:
        del sys.modules[mName]
    return importlib.import_module(mName)

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from collections import OrderedDict, Counter
os = __import__('os')


# In[18]:


dt = datetime.datetime.now()
def dateToNumeric(date):
    return date.year*10000+date.month*100+date.day


# In[41]:


class ModelMfDNN(object):
    def __init__(self,
                 n_items,
                 n_genres,
                 n_hall,
                 n_ag,
                 max_n_genres,
                 dim=32,
                 reg=0.05,
                 learning_rate=0.01,
                 dt = dt):
        """初始化 Tensorflow Graph"""
        
        self.dt = dt
        modelDir="./python-lottery-mfdnn-model-"+str(dateToNumeric(dt))+"/model_mf_with_dnn"
        self.java_model_path = './java-lottery-mfdnn-model-'+str(dateToNumeric(dt))
        self.max_n_genres = max_n_genres
        self.n_items = n_items
        self.n_genres = n_genres
        self.n_hall = n_hall
        self.n_ag = n_ag
        self.ftr_cols = OrderedDict()
        self.initial_learning_rate = learning_rate
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("decay_lr"):
                self.global_step = tf.Variable(0,trainable= False)
                self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,                                           global_step=self.global_step,                                           decay_steps=5,decay_rate=0.8)
                self.add_global = self.global_step.assign_add(1)
            # inputs/id_user:0
            with tf.variable_scope("inputs"):
                self.lr =tf.placeholder(tf.float32,None,name="lr")
                self.isTrain = tf.placeholder(tf.bool, None,name="isTrain")
                # user data
                self.hall_id = tf.placeholder(tf.int32,[None],name="hall_id")
                self.ag_id = tf.placeholder(tf.int32,[None],name="ag_id")
                self.user_id = tf.placeholder(tf.int32, [None],name="user_id")
                self.query_game_ids = tf.placeholder(tf.int32, [None, None],name="query_game_ids")
                self.query_game_ids_len = tf.placeholder(tf.int32, [None],name="query_game_ids_len")
              #  self.web_like_type = tf.placeholder(tf.int32,[None],name="web_like_type")
                # item data
                self.genres = tf.placeholder(tf.int32, [None, None],name="genres")
                self.genres_len = tf.placeholder(tf.int32, [None],name="genres_len")
                self.avg_rating = tf.placeholder(tf.float32, [None],name="avg_rating")
                
                self.candidate_game_id = tf.placeholder(tf.int32, [None],name="candidate_game_id")
                self.web_type_0 = tf.placeholder(tf.float32,[None],name="web_type_0")
                self.web_type_7 = tf.placeholder(tf.float32,[None],name="web_type_7")
                self.rating = tf.placeholder(tf.float32, [None],name="rating")

            init_fn = tf.glorot_normal_initializer()
            emb_init_fn = tf.glorot_uniform_initializer()
            self.b_global = tf.Variable(emb_init_fn(shape=[]), name="b_global")
            with tf.variable_scope("embedding"):
                self.w_query_game_ids = tf.Variable(emb_init_fn(shape=[self.n_items, dim]), name="w_query_game_ids")
                self.b_query_game_ids = tf.Variable(emb_init_fn(shape=[dim]), name="b_query_game_ids")
                self.w_candidate_game_id = tf.Variable(init_fn(shape=[self.n_items, dim]), name="w_candidate_game_id")
                self.b_candidate_game_id = tf.Variable(init_fn(shape=[dim + self.max_n_genres + 3]), name="b_candidate_game_id")
                self.w_genres = tf.Variable(emb_init_fn(shape=[self.n_genres, self.max_n_genres]), name="w_genres")
                self.w_hall = tf.Variable(emb_init_fn(shape=[self.n_hall,dim]),name="w_hall")
                self.w_ag =tf.Variable(emb_init_fn(shape=[self.n_ag,dim]),name="w_ag")
               # self.w_web_like_type = tf.Variable(emb_init_fn(shape = [2,dim]),name= "w_web_like_type")
                # query_game embedding
                '''sqrtn aggregation(pooling), X: data, W: weight
                       X_1*W_1 + X_2*W_2 + ... + X_n*W_n / sqrt(W_1**2 + W_2**2 + ... W_n**2)
                     = weighted sum of X and normalized W
                   here data = self.query_emb, weight = query_game_mask '''
                self.query_emb = tf.nn.embedding_lookup(self.w_query_game_ids, self.query_game_ids)
                query_game_mask = tf.expand_dims(tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.query_game_ids_len)), 1), -1)
                self.query_emb = tf.reduce_sum(self.query_emb * query_game_mask, 1)
                self.query_bias = tf.matmul(self.query_emb, self.b_query_game_ids[:, tf.newaxis])
                # hall,ag embedding
                self.hall_emb = tf.nn.embedding_lookup(self.w_hall,self.hall_id) 
                self.ag_emb = tf.nn.embedding_lookup(self.w_ag,self.ag_id)
                # web like type embedding
               # self.web_like_emb = tf.nn.embedding_lookup(self.w_web_like_type,self.web_like_type) 
                # candidate_game embedding
                self.candidate_emb = tf.nn.embedding_lookup(self.w_candidate_game_id, self.candidate_game_id)

                # genres embedding
                '''sqrtn aggregation(pooling), X: data, W: weight
                       X_1*W_1 + X_2*W_2 + ... + X_n*W_n / sqrt(W_1**2 + W_2**2 + ... W_n**2)
                     = weighted sum of X and normalized W
                   here data = self.genres_emb, weight = genres_mask '''
                self.genres_emb = tf.nn.embedding_lookup(self.w_genres, tf.to_int32(self.genres))
                genres_mask = tf.expand_dims( tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.genres_len)), 1), -1)
                self.genres_emb = tf.reduce_sum(self.genres_emb * genres_mask, 1)
            
            with tf.variable_scope("dnn"):
                # encode [item embedding + item metadata]
                self.item_repr = tf.concat([self.candidate_emb, self.genres_emb, self.avg_rating[:, tf.newaxis],                                            self.web_type_0[:,tf.newaxis],self.web_type_7[:,tf.newaxis]], 1)
                
                self.user_repr = tf.concat([self.query_emb,self.hall_emb,self.ag_emb],1)
                self.candidate_bias = tf.matmul(self.item_repr, self.b_candidate_game_id[:, tf.newaxis])
                
                dp_scale = 0.5
              #  regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
                self.item_repr = tf.layers.dense(self.item_repr, dim, kernel_initializer=init_fn,                                                  activation=tf.nn.relu)
                self.item_repr = tf.layers.dense(self.item_repr, dim, kernel_initializer=init_fn,                                                  activation=tf.nn.relu)
                self.item_repr = tf.layers.dropout(self.item_repr, dp_scale, training=self.isTrain)
#                 self.item_repr = tf.layers.dense(self.item_repr, dim, kernel_initializer=init_fn, activation=tf.nn.relu)
#                 self.item_repr = tf.layers.dropout(self.item_repr, dp_scale, training=self.isTrain)
                self.user_repr = tf.layers.dense(self.user_repr,dim,kernel_initializer=init_fn,activation=tf.nn.relu)
                self.user_repr = tf.layers.dense(self.user_repr,dim,kernel_initializer=init_fn,activation=tf.nn.relu)
                self.user_repr = tf.layers.dropout(self.user_repr, dp_scale, training=self.isTrain)
#                 self.item_repr = tf.layers.dense(self.item_repr, dim, kernel_initializer=init_fn, activation=tf.nn.relu)
#                 self.item_repr = tf.layers.dropout(self.item_repr, dp_scale, training=self.isTrain)
#                 self.item_repr = tf.layers.dense(self.item_repr, dim, kernel_initializer=init_fn, activation=tf.nn.relu)
#                 self.item_repr = tf.layers.dropout(self.item_repr, dp_scale, training=self.isTrain)
                
            with tf.variable_scope("computation"):
                infer = tf.reduce_sum(self.user_repr * self.item_repr, 1, keep_dims=True)
                infer = tf.add(infer, self.b_global)
                infer = tf.add(infer, self.query_bias)
                self.infer = tf.add(infer, self.candidate_bias, name="infer")

                # one query for all items
              #  self.pred = tf.matmul(self.user_repr, tf.transpose(self.item_repr)) + \
               #             tf.reshape(self.candidate_bias, (1, -1)) + self.query_bias + self.b_global
                
                self.pred = tf.add(tf.add(tf.add(tf.matmul(self.user_repr,tf.transpose(self.item_repr)), 
                                          tf.reshape(self.candidate_bias, (1, -1))),self.query_bias),self.b_global,name="pred")
                pass

            with tf.variable_scope("loss"):
              #  self.regularizer = reg * tf.add(tf.nn.l2_loss(self.query_emb), tf.nn.l2_loss(self.candidate_emb))
               # l2_layer_loss = tf.losses.get_regularization_loss()
                self.loss = tf.losses.mean_squared_error(labels=self.rating[:, tf.newaxis], predictions=self.infer)# + \
                           # self.regularizer + l2_layer_loss
             #   self.loss =tf.nn.softmax_cross_entropy_with_logits(labels=self.rating[:,tf.newaxis],logits=self.infer)
                # for eval
                self.rmse_loss = tf.sqrt(self.loss)
                self.mae_loss = tf.reduce_mean(tf.abs(self.infer - self.rating[:, tf.newaxis]))
                pass

            with tf.variable_scope("train"):
                #self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
                self.train_op = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)
                #self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
                #self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
                #self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                pass

            self.saver = tf.train.Saver(tf.global_variables())
            self.graph = graph
            self.modelDir = modelDir
            
    def resetModel(self, modelDir):
        
        shutil.rmtree(path=modelDir, ignore_errors=True)
        os.makedirs(modelDir)

    def feed_dict(self, data,lr, mode="train"):
        ret = {
            self.hall_id:data["hall_id_trans"],
            self.ag_id:data["ag_id_trans"],
            self.user_id: data["user_id_trans"],
            self.query_game_ids: data["query_game_ids"],
            self.query_game_ids_len: data["query_game_ids_len"],
            self.genres: data["genres"],
            self.genres_len: data["genres_len"],
            self.avg_rating: data["avg_rating"],
            #self.web_like_type: data["web_like_type"],
            self.web_type_0 : data["web_type_0"],
            self.web_type_7 : data["web_type_7"],
        #    self.year: data["year"],
            self.candidate_game_id: data["game_id_trans"]
        }
        ret[self.lr] = lr
        ret[self.isTrain] = False
        if mode != "infer":
            ret[self.rating] = data["rating"]
            
            if mode == "train":
                ret[self.isTrain] = True
            elif mode == "eval":
                pass
        return ret

    def fit(self, sess, trainGen, testGen, reset=False, nEpoch=50):
        sess.run(tf.global_variables_initializer())
        if reset:
            print("reset model: clean model dir: {} ...".format(self.modelDir))
            self.resetModel(self.modelDir)
        # try: 試著重上次儲存的model再次training
        self.ckpt(sess, self.modelDir)

        start = time.time()
        print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
        minLoss = 1e7
        for ep in range(1, nEpoch + 1):
            tr_loss, tr_total = 0, 0
            if ep % 5 == 0:
                _,lr = sess.run([self.add_global,self.learning_rate])
            else :
                lr = sess.run(self.learning_rate)
            for i, data in enumerate(trainGen(), 1):
                loss, _ = sess.run([self.rmse_loss, self.train_op], feed_dict=self.feed_dict(data,lr, mode="train"))
                tr_loss += loss ** 2 * len(data["query_game_ids"])
                tr_total += len(data["query_game_ids"])
               
                print("\rtrain loss: {:.3f} (lr : %.6f)".format(loss)%lr, end="")
            if testGen is not None:
                epochLoss = self.epochLoss(sess, testGen)

            tpl = "\r%02d\t%.3f\t\t%.3f\t\t%.3f secs"
            if minLoss > epochLoss:
                tpl += ", saving ..."
                self.saver.save(sess, os.path.join(self.modelDir, 'model'), global_step=ep)
                minLoss = epochLoss

            end = time.time()
            print(tpl % (ep, np.sqrt(tr_loss / tr_total), epochLoss, end - start))
            start = end
        return self

    def ckpt(self, sess, modelDir):
        """load latest saved model"""
        latestCkpt = tf.train.latest_checkpoint(modelDir)
        if latestCkpt:
            self.saver.restore(sess, latestCkpt)
        return latestCkpt

    def epochLoss(self, sess, dataGen, tpe="rmse"):
        totLoss, totCnt = 0, 0
        for data in dataGen():
            lossTensor = self.rmse_loss if tpe == "rmse" else self.mae_loss
            
            loss = sess.run(lossTensor, feed_dict=self.feed_dict(data,0, mode="eval"))
            #print(loss)
            totLoss += (loss ** 2 if tpe == "rmse" else loss) * len(data["query_game_ids"])
            totCnt += len(data["query_game_ids"])
        return np.sqrt(totLoss / totCnt) if tpe == "rmse" else totLoss / totCnt

    def predict(self, sess, user_queries, items):
        self.ckpt(sess, self.modelDir)
        return sess.run(self.pred, feed_dict={
            self.isTrain: False,
            self.hall_id:user_queries["hall_id_trans"],
            self.ag_id:user_queries["ag_id_trans"],
            self.user_id: user_queries["user_id_trans"],
            self.query_game_ids: user_queries["query_game_ids"],
            self.query_game_ids_len: user_queries["query_game_ids_len"],
         #   self.web_like_type: user_queries["web_like_type"],
            ##################################
            self.genres: items["genres"],
            self.genres_len: items["genres_len"],
            self.avg_rating: items["avg_rating"],
            self.web_type_0 : items["web_type_0"],
            self.web_type_7 : items["web_type_7"],
            #self.year: items["year"],
            self.candidate_game_id: items["candidate_game_id"]
        })

    def evaluateRMSE(self, sess, dataGen):
        self.ckpt(sess, self.modelDir)
        return self.epochLoss(sess, dataGen, tpe="rmse")

    def evaluateMAE(self, sess, dataGen):
        self.ckpt(sess, self.modelDir)
        return self.epochLoss(sess, dataGen, tpe="mae")
    
    def save_model(self,session):
        shutil.rmtree(path=self.java_model_path, ignore_errors=True)
        signature = tf.saved_model.signature_def_utils.build_signature_def( 
        inputs = { 
        # user data
        'input_lr': tf.saved_model.utils.build_tensor_info(self.lr),
        'input_isTrain': tf.saved_model.utils.build_tensor_info(self.isTrain),
        'input_hall_id': tf.saved_model.utils.build_tensor_info(self.hall_id),
        'input_ag_id': tf.saved_model.utils.build_tensor_info(self.ag_id),
        'input_query_game_ids': tf.saved_model.utils.build_tensor_info(self.query_game_ids),
        'input_query_game_ids_len': tf.saved_model.utils.build_tensor_info(self.query_game_ids_len),
       # 'input_web_like_type': tf.saved_model.utils.build_tensor_info(self.web_like_type),
        # item data
        'input_genres': tf.saved_model.utils.build_tensor_info(self.genres),
        'input_genres_len': tf.saved_model.utils.build_tensor_info(self.genres_len),
        'input_avg_rating': tf.saved_model.utils.build_tensor_info(self.avg_rating),
        'input_candidate_game_id': tf.saved_model.utils.build_tensor_info(self.candidate_game_id),
        'input_web_type_0': tf.saved_model.utils.build_tensor_info(self.web_type_0),
        'input_web_type_7': tf.saved_model.utils.build_tensor_info(self.web_type_7),
        },
        outputs = {'output': tf.saved_model.utils.build_tensor_info(self.infer)})
        b = tf.saved_model.builder.SavedModelBuilder(self.java_model_path)
        b.add_meta_graph_and_variables(session,[tf.saved_model.tag_constants.SERVING],signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
        b.save() 

