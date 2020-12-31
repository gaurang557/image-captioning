from keras.models import *
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
import os
from keras.preprocessing import image
import tensorflow as tf


global mod
mod=load_model('gaurangmodel.h5')
global graph1
graph1=tf.get_default_graph()
# mod._make_predict_function()



with open('fv.txt',"r") as f:
    fv=f.read().split()
word2idx={}
idx2word={}
for i,j in enumerate(fv):
    word2idx[j]=i+1
    idx2word[i+1]=j


# model_res._make_predict_function()
global model_res
model_res=ResNet50(weights='imagenet',input_shape=(224,224,3))
model_res=Model(model_res.input,model_res.layers[-2].output)
# this is key : save the graph after loading the model
global graph
graph = tf.get_default_graph()


def get(path):
    if not os.path.exists(path):
        return 0
    img=image.load_img(path,target_size=(224,224,3))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    print(img.shape)
    img=preprocess_input(img)
    with graph.as_default():
        img=model_res.predict(img)
    img=img.reshape((1,-1))
    return img

def predict(path):
    s='startseq';seq=""
    img=get(path)
    while True:        
        cap=[word2idx[c] for c in s.split() if c in word2idx.keys()]
        cap=pad_sequences([cap],maxlen=39,value=0,padding='post')[0].reshape(1,39)
        with graph1.as_default():       
            res=mod.predict([img,cap])
        res=np.argmax(res)
        s+=' '+idx2word[res]
        if idx2word[res]=='endseq':
            return seq
        seq+=idx2word[res]+" "