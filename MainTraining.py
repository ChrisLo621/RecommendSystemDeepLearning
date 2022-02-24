
# coding: utf-8

# In[5]:


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
utils = reload('utils.utils')
from LotteryMFDNN import ModelMfDNN


# In[6]:


def date_range(start, end, intv):
    
    start = datetime.datetime.strptime(start,"%Y%m%d")
    end = datetime.datetime.strptime(end,"%Y%m%d")
    diff = (end  - start ) / intv
    for i in range(intv):
        yield (start + diff * i).strftime("%Y%m%d")
    yield end.strftime("%Y%m%d")

def quantile(array,bottom,top):
    return np.percentile(array,bottom),np.percentile(array,top)

def ConnectMemSQL(query,columns):
    conn=pymysql.connect(host='10.28.1.12',port=3306,user='BI_hp',password='djo62u4fu6BI',db='OLAP',charset='utf8')
    cursor = conn.cursor()
    cursor.execute(query)
    conn.close()
    rows=cursor.fetchall()
    #data=pd.DataFrame([[ij for ij in i] for i in rows],columns=columns)
    data=pd.DataFrame.from_records(list(rows),columns=columns)
    return data

def wagers_df(i):
    global start_list
    global end_list
   # query='select User_Key,GameType_Key,WagersTotal from Fact_Aggr_Wagers_User \
   # where GameKind_Key in (121,122)  and RoundDate_Key between '+start_list[i]+' and '+end_list[i]
    query='select a.RoundDate_Key,a.Hall_Key,a.AG_Key,b.Payway_Key,a.User_Key,a.GameType_Key,a.WagersTotal            from Fact_Aggr_Wagers_User as a,Dim_Hall as b            where a.GameKind_Key  in (121,122) and a.Hall_Key=b.Hall_Key and  a.RoundDate_Key between '+start_list[i]+' and '+end_list[i] +' and b.API=0 and a.Test=0 and a.Hall_Test=0'
   # print(query)
    #print('==============================================================================================================')
    colsname=['RoundDate_Key','Hall_Key','AG_Key','Payway_Key','User_Key','GameType_Key','WagersTotal']
    df=ConnectMemSQL(query,colsname)
    return df
    #print(query)

def web_type_df(i):
    global start_list
    global end_list
    
    query='select RoundDate_Key,User_Key,GameType_Key,count(if(WebVersion=7,WagersID,NULL)) as wag_7 , count(if(WebVersion!=7,WagersID,NULL)) as wag_0            from Fact_Wagers_2_FastReport            where RoundDate_Key between '+start_list[i]+' and '+end_list[i]+' group by 1,2,3'
    
   # print(query)
    #print('==============================================================================================================')
    colsname=['RoundDate_Key','User_Key','GameType_Key','Wag_7','Wag_0']
    df=ConnectMemSQL(query,colsname)
    return df 
    
    
    
def rank_function(df):
    df['rating']=df.WagersTotal.rank(ascending=True).head(20)
    return df


def rating_function(df):
    
   # total_wagers = sum(np.log(df.WagersTotal))
    
   # df['rating']= df['WagersTotal'].apply(lambda x : np.log(x)/total_wagers if np.log(x)>=0 and total_wagers>0  else 0.0) #(np.log(df.WagersTotal)/)
    #df['data'].apply(lambda x: 'true' if x <= 2.5 else 'false')
    df['rating']=(df.WagersTotal/sum(df.WagersTotal))
    return df


def smaple_func(df):
    
    
    
    return df.groupby(['User_Key','GameType_Key','Hall_Key','AG_Key','Payway_Key'])['RoundDate_Key','WagersTotal'].max().reset_index().tail(10)
    

def avg_function(df):
    return np.mean(df.Rating.values)

def apply_parallel(df,func):
    ret=Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in df.groupby(['User_Key']))
    return pd.concat(ret)





def preprocess(u,df):  
    queue=[]
    columns=["hall_id_trans","ag_id_trans","payway","user_id","query_game_ids","game_id_trans"
             ,"web_like_type","genres","web_type_0","web_type_7"
             ,"rating","avg_rating","month","day","week_day"
             ,"round_time"
            ]

    df=df.sort_values("rating",ascending=True)
    for i,(_,r) in enumerate(df.iterrows()):
        
        queue.append([int(r.hall_id_trans),int(r.ag_id_trans),int(r.Payway_Key),int(r.User_Key),df.game_id_trans[:i].tolist()+df.game_id_trans[i+1:].tolist()
                      ,int(r.game_id_trans),int(r.web_like_type),r.genres,r.web_type_0
                      ,r.web_type_7,r.rating,r.avg_rating,int(r.month),int(r.day),int(r.week_day)
                      ,r.round_time
                     ])
        
        
    return pd.DataFrame(queue,columns=columns) 


def preprocess(u,df):  
    queue=[]
    columns=["hall_id_trans","ag_id_trans","payway","user_id"
             ,"query_game_ids","game_id_trans"
             ,"web_like_type","genres","web_type_0","web_type_7"
             ,"rating","avg_rating"
             ,"daily_count"
            ]

    df=df.sort_values("rating",ascending=True)
    for i,(_,r) in enumerate(df.iterrows()):
        
        queue.append([int(r.hall_id_trans),int(r.ag_id_trans),int(r.Payway_Key),int(r.User_Key)
                      ,df.game_id_trans[:i].tolist()+df.game_id_trans[i+1:].tolist()
                      ,int(r.game_id_trans),int(r.web_like_type),r.genres,r.web_type_0
                      ,r.web_type_7,r.rating,r.avg_rating
                      ,r.daily_count
                     ])
        
        
    return pd.DataFrame(queue,columns=columns) 



def preprocessTrain(u,df):  
    queue=[]
    columns=["hall_id_trans","ag_id_trans","user_id_trans"
             ,"query_game_ids","game_id_trans"
             ,"genres","web_type_0","web_type_7"
             ,"rating","avg_rating"
             ,"daily_count"
            
            ]
    
    df=df.sort_values("rating",ascending=True)  

    for i,(_,r) in enumerate(df.iterrows()):

        queue.append([int(r.hall_id_trans),int(r.ag_id_trans),int(r.user_id_trans)
                      ,df.game_id_trans[:i].tolist()+df.game_id_trans[i+1:].tolist()
                      ,int(r.game_id_trans),r.genres,r.web_type_0
                      ,r.web_type_7,r.rating,r.avg_rating
                      ,r.daily_count
                     
                     ])

        
    return pd.DataFrame(queue,columns=columns)




def preprocessTest(u,df,train_hist):  
    queue=[]
    columns=["hall_id_trans","ag_id_trans","user_id_trans"
             ,"query_game_ids","game_id_trans"
             ,"genres","web_type_0","web_type_7"
             ,"rating","avg_rating"
             ,"daily_count"
            
            ]
    
    df=df.sort_values("rating",ascending=True)
    
        #user_game_hist = train_hist.query("User_Key=={}".format(u)).game_id_trans
    user_game_hist = train_hist[train_hist.user_id_trans==u].game_id_trans
    #user_web_like_type = train_hist[train_hist.user_id_trans==u].web_like_type
    
    for i,(_,r) in enumerate(df.iterrows()):

            all_hist = set(user_game_hist.tolist())
                  
            queue.append([int(r.hall_id_trans),int(r.ag_id_trans),int(r.user_id_trans)
                      ,df.game_id_trans[:i].tolist()+df.game_id_trans[i+1:].tolist()
                      ,int(r.game_id_trans),r.genres,r.web_type_0
                      ,r.web_type_7,r.rating,r.avg_rating
                      ,r.daily_count
                     
                     ])
        
    return pd.DataFrame(queue,columns=columns)


def Date_To_Numeric(date):
    return date.year*10000+date.month*100+date.day

class recordDate:
    def __init__(self):
        self.run_date = datetime.datetime.now()        
        self.end_date = self.run_date - datetime.timedelta(days=2)
        self.start_date = self.end_date - datetime.timedelta(days=120)
        self.test_date = self.end_date - datetime.timedelta(days=30)
        self.date_num_dict = {}
        
        self.date_num_dict['run_date'] = self.dateToNumeric(self.run_date)
        self.date_num_dict['start_date'] = self.dateToNumeric(self.start_date)
        self.date_num_dict['end_date'] = self.dateToNumeric(self.end_date)
        self.date_num_dict['test_date'] = self.dateToNumeric(self.test_date)
        self.n_split_date = 60
    def dateToNumeric(self,date):
        
        return date.year*10000+date.month*100+date.day
    
    def splitDate(self):
        time_split=list(date_range(str(self.date_num_dict['start_date']),str(self.date_num_dict['end_date']),self.n_split_date))
        start=0
        start_list=[]
        end_list=[]
        for t in range(len(time_split)-1):
            if start ==0:
                start=time_split[t]
                end=time_split[t+1]
            else:
                start=datetime.datetime.strptime(end,"%Y%m%d")+datetime.timedelta(days=1)
                start=start.strftime("%Y%m%d")
                end=time_split[t+1]
        
            start_list.append(start)
            end_list.append(end)
        return start_list,end_list

class timetick:
    def __init__(self,start):
        self.start=start
    def tick(self,time):
        sec=time-self.start
        self.start=time
        return str(sec) + " sec"

def dateToNumeric(date):
    return date.year*10000+date.month*100+date.day
time_delta=timetick(timeit.default_timer())


# In[7]:


date = recordDate()
print('date information =>',date.date_num_dict)
start_list,end_list = date.splitDate()


# In[8]:


date.run_date


# In[9]:


web_df=web_type_df(0)
for d in range(len(start_list)-1):
    web_df=web_df.append(web_type_df(d+1), ignore_index=True)
    time.sleep(0.5)
print("get web data used time : ",time_delta.tick(timeit.default_timer()))


# In[14]:


query='select GameType_Key,GameType_Name from Dim_GameType where GameKind_Key in (122,121)'
colsname=['GameType_Key','GameType_Name']
Dim_GameType=ConnectMemSQL(query,colsname)


item=pd.read_csv("./ItemData/item_category.csv")
item=item.fillna(0)
item_dict={}
label_dict={}
key_dict={}

for _,row in Dim_GameType.iterrows():
    key_dict[row['GameType_Name']]=row['GameType_Key']

for index ,row in item.iteritems():
    label_dict[index]=[]

count=0
for k,v in label_dict.items():
    label_dict[k]=count
    count+=1

for index, row in item.iteritems():
   # print(index)
    #print(row.values)
    items_name=list(filter((0).__ne__, row.values.tolist()))
    for i in items_name:
        item_dict[key_dict[i.replace(" ","")]]=[]

for index,row in item.iteritems():
    items_name=list(filter((0).__ne__, row.values.tolist()))
    for i in items_name:
        item_dict[key_dict[i.replace(" ","")]]+=[label_dict[index]]
    #item_dict[]=list(filter((0).__ne__, index))
    
n_game=len(item_dict)
n_genres=len(label_dict)

item_data=pd.DataFrame(pd.Series(item_dict)).reset_index()
item_data.columns=["GameType_Key","genres"]

item_le=LabelEncoder()
item_le.fit(item_data.GameType_Key)
item_data['game_id_trans']=item_le.transform(item_data.GameType_Key)

max_n_genres=item_data.genres.map(len).max()
n_game = item_data.shape[0]


# In[15]:


item_detail=pd.read_csv("./ItemData/item_detail.csv")

queue=[]
cols_name=["GameType_Key","round_time","daily_count","game_result"]
for i,r in item_detail.iterrows():
    queue.append([int(key_dict[r.GameType_Name]),r.time,r.counts,r.result])
item_detail=pd.DataFrame(queue,columns=cols_name)
item_data=pd.merge(item_data,item_detail,on=["GameType_Key"])


# In[16]:


df=wagers_df(0)
for d in range(len(start_list)-1):
    df=df.append(wagers_df(d+1), ignore_index=True)
    time.sleep(0.5)
 #  plit_df(d+1) 
#print(df.head())
print("get wagers data used time : ",time_delta.tick(timeit.default_timer()))


# In[17]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    df_out = df_in.loc[df_in[col_name] > fence_low]
    return df_out
def web_preprocess(df):
    queue=[]
    columns=["game_id_trans","web_type_0","web_type_7"]
    for i in np.unique(df.game_id_trans.values):
        
        type0_percentage=df[(df.game_id_trans==i) & (df.web_like_type==0)]['count_percentage'].values.tolist()[0]
        type7_percentage=df[(df.game_id_trans==i) & (df.web_like_type==1)]['count_percentage'].values.tolist()[0]
        queue.append([int(i),type0_percentage,type7_percentage])
    return pd.DataFrame(queue,columns=columns)


# In[18]:


train_df = df[df.RoundDate_Key<date.date_num_dict['test_date']].reset_index()
test_df = df[df.RoundDate_Key>=date.date_num_dict['test_date']].reset_index()


# In[19]:


class generateData:
    def __init__(self):
        
        self.web_le = None
        self.hall_le = None
        self.ag_le = None
        self.payway_le = None


    def preprocData(self,df,web_df,item_data,train = True,avg_r = None):
          if train :
            self.web_le = LabelEncoder()
            self.hall_le = LabelEncoder()
            self.ag_le = LabelEncoder()
            self.payway_le = LabelEncoder()
            
          #filter days = 1
          user_days_count=pd.DataFrame(df.groupby(['User_Key'])['RoundDate_Key'].agg('count')).reset_index()
          user_days_count=user_days_count.rename(index=str,columns={"RoundDate_Key":"days"})
          #filter games = 1
          user_games_count=pd.DataFrame(df.groupby(['User_Key'])['GameType_Key'].nunique()).reset_index()
          user_games_count=user_games_count.rename(index=str,columns={"GameType_Key":"games"})
          user_days_count,user_games_count=user_days_count[user_days_count.days>1],user_games_count[user_games_count.games>1]
          # remove outlier days,games
          user_days_in_value = remove_outlier(user_days_count,'days')
          user_games_in_value = remove_outlier(user_games_count,'games')
          user_in_value = pd.merge(user_days_in_value,user_games_in_value,on = 'User_Key',how='inner')
          df_in_value = pd.merge(df,user_games_in_value,on='User_Key',how='inner')
          # group by users , items and sum of wagers
          df_in_value = df_in_value.groupby(['Hall_Key','AG_Key','User_Key','GameType_Key'])['WagersTotal'].sum().reset_index()
          # remove wagers outlier
          df_in_value = remove_outlier(df_in_value,'WagersTotal')
          # balance daily round and WagersTotal
        
        #  for i,j in df_in_value.iterrows():
         #       daily_round = item_data[item_data['GameType_Key']==j['GameType_Key']]['daily_count'].values[0]
          #      j['WagersTotal'] = j['WagersTotal'] / daily_round
          c = df_in_value.columns
          df_in_value = pd.merge(item_data,df_in_value,on="GameType_Key",how="left")
          df_in_value['WagersTotal'] = df_in_value['WagersTotal'] / df_in_value['daily_count']
          df_in_value = df_in_value[c]
          
          # rating 
          df_in_value = apply_parallel(df_in_value,rating_function)
    
          # Web
          web_df_ = pd.merge(web_df,user_in_value,on="User_Key",how="inner")
          web_df_ = web_df_.groupby(['User_Key','GameType_Key'])[['Wag_7','Wag_0']].sum().reset_index()
          web_df_['web_max']=web_df_[['Wag_7','Wag_0']].idxmax(axis=1)
          # merge web_df_ and rating df
          df_in_value = pd.merge(web_df_,df_in_value,on=['User_Key','GameType_Key'],how='left')
          df_in_value = df_in_value.dropna()
          # merge item_data and web + rating df 
          df_in_value=pd.merge(item_data,df_in_value,on="GameType_Key",how="left")
          df_in_value=df_in_value.dropna()
    
          if train :
            self.web_le.fit(df_in_value.web_max)
            
          df_in_value['web_like_type'] = self.web_le.transform(df_in_value.web_max)
          if train :
                
              # compute avg rating by items
              avg_rating=pd.DataFrame(df_in_value.groupby('game_id_trans').rating.mean()                                .fillna(df_in_value.rating.mean()))
    
              avg_rating=avg_rating.reset_index()
              avg_rating=avg_rating.rename(index=str,columns={'rating':'avg_rating'})
          else:
              avg_rating = avg_r
          # add avg rating on item + web + rating df 
          df_in_value=pd.merge(avg_rating,df_in_value,on=['game_id_trans'],how='left')
          # add avg rating on item data
          item_data=pd.merge(item_data,avg_rating,on=['game_id_trans'],how='outer')
          # = = = = = = = = = = = = = = = = == = = = = = = = = = = =
          # compute web like percentage
          game_web_count=pd.DataFrame(df_in_value.groupby(['game_id_trans','web_like_type'])
                 .web_like_type.count().reset_index(name='count'))
    
          for i in np.unique(game_web_count.game_id_trans.values):
              for j in [0,1]:
                  if game_web_count[(game_web_count.game_id_trans==i) & (game_web_count.web_like_type==j)].empty == True:
                      insert_row=pd.DataFrame([[i,j,0]],columns=['game_id_trans', 'web_like_type', 'count'])
                      game_web_count=game_web_count.append(insert_row)
          game_web_max_count=pd.DataFrame(game_web_count.groupby(['game_id_trans'])['count'].sum().reset_index(name='sum_count'))
    
    
          game_web_count=pd.merge(game_web_count,game_web_max_count,on=['game_id_trans'],how='outer')
          game_web_count['count_percentage']=game_web_count['count']/game_web_count['sum_count']
    
          game_web_percent=web_preprocess(game_web_count)
          df_in_value=pd.merge(df_in_value,game_web_percent,on="game_id_trans",how="outer")
    
          item_data=pd.merge(item_data,game_web_percent,on='game_id_trans',how='inner')
    
          return df_in_value,item_data,avg_rating,self.web_le


# In[20]:


gendata = generateData()
train_preproc,tr_item_data,avg_rating,web_le = gendata.preprocData(train_df,web_df,item_data)
print("preproc training data used time : ",time_delta.tick(timeit.default_timer()))
test_preproc,test_item_data, _,_ = gendata.preprocData(df = test_df,web_df=web_df,item_data=item_data,train = False,avg_r = avg_rating)
print("preproc testing data used time : ",time_delta.tick(timeit.default_timer()))


# In[21]:


train_preproc_ = train_preproc.copy()
test_preproc_ = test_preproc.copy()

train_users = train_preproc_.User_Key.unique()
test_users = test_preproc_.User_Key.unique()

train_test_users = np.intersect1d(train_users,test_users)

train_preproc_ = train_preproc_[train_preproc_.User_Key.isin(train_test_users)]
test_preproc_ = test_preproc_[test_preproc_.User_Key.isin(train_test_users)]

all_preproc = train_preproc_.append(test_preproc_)

hall_le = LabelEncoder()
ag_le = LabelEncoder()
payway_le = LabelEncoder()
user_le = LabelEncoder()

hall_le.fit(all_preproc['Hall_Key'])
ag_le.fit(all_preproc['AG_Key'])
#payway_le.fit(all_preproc['Payway_Key'])
user_le.fit(all_preproc['User_Key'])

train_preproc_['hall_id_trans'] = hall_le.transform(train_preproc_['Hall_Key'])
test_preproc_['hall_id_trans'] = hall_le.transform(test_preproc_['Hall_Key'])

train_preproc_['ag_id_trans'] = ag_le.transform(train_preproc_['AG_Key'])
test_preproc_['ag_id_trans'] = ag_le.transform(test_preproc_['AG_Key'])

#train_preproc_['payway_id_trans'] = payway_le.transform(train_preproc_['Payway_Key'])
#test_preproc_['payway_id_trans'] = payway_le.transform(test_preproc_['Payway_Key'])

train_preproc_['user_id_trans'] = user_le.transform(train_preproc_['User_Key'])
test_preproc_['user_id_trans'] = user_le.transform(test_preproc_['User_Key'])

all_preproc = train_preproc_.append(test_preproc_)


# In[22]:


trProcessed=Parallel(n_jobs=multiprocessing.cpu_count())(delayed(preprocessTrain)(u,d) for u,d in train_preproc_.groupby("user_id_trans"))
teProcessed=Parallel(n_jobs=multiprocessing.cpu_count())(delayed(preprocessTest)(u,d,train_preproc_) for u,d in test_preproc_.groupby("user_id_trans"))

trProcessed=pd.concat(trProcessed)
teProcessed=pd.concat(teProcessed)

print("leave one out  training and testing data used time : ",time_delta.tick(timeit.default_timer()))


# In[23]:


def do_multi(df, multi_cols):
    """對於multivalent的欄位, 需要增加一個column去描述該欄位的長度"""
    pad = tf.keras.preprocessing.sequence.pad_sequences
    ret = OrderedDict()
    for colname, col in df.iteritems():
        if colname in multi_cols:
            lens = col.map(len)
            ret[colname] = list(pad(col, padding="post", maxlen=lens.max()))
            ret[colname + "_len"] = lens.values
        else:
            ret[colname] = col.values
    return ret

def dataFn(data, n_batch=128, shuffle=False):
    pad = tf.keras.preprocessing.sequence.pad_sequences
    def fn():
        dataInner = data.copy()
        indices = utils.get_minibatches_idx(len(dataInner), n_batch, shuffle=shuffle)
        for ind in indices:
            yield do_multi(dataInner.iloc[ind], ["query_game_ids","genres"])
    return fn

for i, e in enumerate(dataFn(trProcessed, n_batch=5, shuffle=True)(), 1):
    # print(e)
    break
pd.DataFrame(e)


# In[24]:


learning_rate = 0.01
dim = 32
n_batch = 128
#modelDir = "./model-3.5.0/model_mf_with_dnn"
    
tf.reset_default_graph()
model = ModelMfDNN(
            n_items=n_game,
            n_genres=n_genres,
            n_hall=hall_le.classes_.shape[0],
            n_ag=ag_le.classes_.shape[0],
            max_n_genres = max_n_genres,
            dim=dim,
            reg=0.1,
            dt = date.run_date,
            learning_rate=learning_rate)


# In[44]:


with tf.Session(graph=model.graph) as sess:
    model.fit(sess, dataFn(trProcessed, n_batch=n_batch, shuffle=True), dataFn(teProcessed, n_batch=n_batch), nEpoch=100, reset=True)
    model.save_model(sess)
print("training used time : ",time_delta.tick(timeit.default_timer()))


# In[45]:


os.mkdir(os.path.join(model.java_model_path,'model'))

for f in os.listdir(model.java_model_path):
    shutil.move(os.path.join(model.java_model_path,f), os.path.join(model.java_model_path,'model'))#'./java-lottery-mfdnn-model-20190123/model')


# In[46]:


def labelEncToJson(item_class):
    item_dict = {}
    for i,c in enumerate(item_class):
        try:
            item_dict[int(c)] = int(i)
        except:
            item_dict[c] = int(i)
    return item_dict


# In[47]:


class genrateJson:
    
    def __init__(self,hall_le,ag_le,web_le):
        global model
        self.hall_le = hall_le
        self.ag_le = ag_le
        self.web_le = web_le
        self.dir = "java-lottery-mfdnn-model-"+str(dateToNumeric(model.dt))#'json-'+str(dateToNumeric(model.dt))
        #os.mkdir(self.dir)
    def userLabel(self):
        
        label_dict = {}
#label_dict['gameType'] = labelEncToJson(item_le.classes_)
        label_dict['hallId'] = labelEncToJson(hall_le.classes_)
        label_dict['agId'] = labelEncToJson(ag_le.classes_)
        label_dict['webId'] = labelEncToJson(web_le.classes_)
        
        with open(os.path.join(self.dir,'userLabel.json'), 'w') as fp:
            json.dump(label_dict,fp,indent=4,sort_keys=True)
    
    def game(self):
        
        itemDictToJson = {}
        pad = tf.keras.preprocessing.sequence.pad_sequences
        itemDictToJson["sequence"] = [str(i) for i in tr_item_data.game_id_trans.tolist()]#tr_item_data.game_id_trans.tolist()
        for _,i in tr_item_data.iterrows():
            itemDictToJson[i.game_id_trans] = {}
            itemDictToJson[i.game_id_trans]["dailyCount"] = i.daily_count
            itemDictToJson[i.game_id_trans]["gameType"] = int(item_le.inverse_transform([i.game_id_trans])[0])
            itemDictToJson[i.game_id_trans]["genres"] = pad([i.genres],padding="post",maxlen=tr_item_data.genres.map(len).max()).tolist()[0]#i.genres
            itemDictToJson[i.game_id_trans]["genresLength"] = len(i.genres)
            itemDictToJson[i.game_id_trans]["avgRating"] = i.avg_rating
            itemDictToJson[i.game_id_trans]["candidateGameId"] = i.game_id_trans
            itemDictToJson[i.game_id_trans]["webType0"] = i.web_type_0
            itemDictToJson[i.game_id_trans]["webType7"] = i.web_type_7
            
        with open(os.path.join(self.dir,'game.json'), 'w') as fp:
            json.dump(itemDictToJson,fp,indent=4)
    def delete(self):
        shutil.rmtree(self.dir,ignore_errors=True)


# In[48]:


gJ = genrateJson(hall_le,ag_le,web_le)
gJ.userLabel()
gJ.game()


# In[49]:


shutil.make_archive(model.java_model_path,'zip',model.java_model_path)
shutil.make_archive(os.path.split(model.modelDir)[0],'zip',os.path.split(model.modelDir)[0])


# In[50]:


shutil.move("java-lottery-mfdnn-model-"+str(dateToNumeric(model.dt))+".zip", "./java_model")
shutil.move("python-lottery-mfdnn-model-"+str(dateToNumeric(model.dt))+".zip", "./python_model")


# shutil.rmtree("java-lottery-mfdnn-model-"+str(dateToNumeric(model.dt)),ignore_errors=True)
# shutil.rmtree("python-lottery-mfdnn-model-"+str(dateToNumeric(model.dt)),ignore_errors=True)

# shutil.make_archive(os.path.join('json_file',gJ.dir),'zip',gJ.dir)

# In[52]:


gJ.delete()


# In[60]:


shutil.rmtree(model.modelDir.split("/")[1],ignore_errors=True)

