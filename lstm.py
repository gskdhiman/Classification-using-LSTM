# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense,Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from sklearn.metrics import confusion_matrix #,cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
lb = LabelBinarizer()

emb_dim = 300
seq_len =30
num_classes = 8
#num_classes autimatically selected based on training data if wrong
epoch_size = 40
validation_split = 0.2
batch_size = 4

#all the paths requored
tensorboard_dir =os.getcwd()
model_path = 'classify_lstm.h5'
train_path = os.path.join('..','Dataset','r8-train-stemmed.txt')
test_path = os.path.join('..','Dataset','r8-test-stemmed.txt')
word_emb_model_path = os.path.join('..','Word_embeddings','GoogleNews-vectors-negative300.bin')


def load_data(path,head_count = 0):
    """This function will load both trainig and test data"""
    global num_classes
    df = pd.read_csv(path, sep="\t",error_bad_lines = False)
    if head_count != 0:
        df = df.head(head_count)
    features = df.iloc[:,1]
    num_classes = len(df.iloc[:,0].value_counts())
    print("Data has only",num_classes,"labels")
    labels = df.iloc[:,0].values
    labels = labels.reshape(df.shape[0],1)
    df = df.drop_duplicates()
    print("Done with: ",path)
    return (features,labels)

def encode_labels(lb,labels,save =False):
    encoderpath = "labels.pickle"
    en_labels = lb.fit_transform(labels)
    if save == True:
        pickle.dump( lb.classes_, open(encoderpath, "wb" ))
    print("Done with label encoder")
    return (en_labels,encoderpath)

def load_word2vec(word_emb_model_path):
    print('Started loading Word2vec model')
    emb_model = KeyedVectors.load_word2vec_format(word_emb_model_path, binary=True)
    print('Word2vec model loaded')
    return emb_model

def get_emb(element):
    global seq_len,emb_dim
    emb = [emb_model[str(word).lower()] if word in emb_model  else np.zeros(emb_dim) for word in element.split()]
    return pad_sequences([emb],maxlen=seq_len,dtype='float32')[0]

#def getembedding(x):
#    global GloveEmbeddings
#    emb = [GloveEmbeddings[word] if word in GloveEmbeddings.keys() else GloveEmbeddings['zerovec'] for word in x.split()]
#    return pad_sequences([emb],maxlen=seq_len, dtype='float32',padding = 'post')


def features_to_emb(features):
    features = features.apply(lambda x : get_emb(x).astype('float64'))
    features = np.array(features.tolist())
    print("Done with the text to embedding conversion")
    return features

def model_design(model_path, seq_len=30,emb_dim = 300,tensorboard_dir = "."):
    inp = Input(shape=(seq_len,emb_dim))
    out = LSTM(300,return_sequences = True)(inp)
    out = Dropout(0.5)(out)
    out = LSTM(100,return_sequences = False)(inp)
    out = Dropout(0.5)(out)
    out = Dense(num_classes, activation='softmax')(out)
    model = Model(inputs = inp, outputs=out)
    checkpointer = ModelCheckpoint(filepath = model_path,
                               verbose=2,
                               save_best_only=True)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                              patience=10, min_lr=0.000001, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join(tensorboard_dir,'logs'),
                          histogram_freq=0,
                          write_graph=True)
    model.compile(optimizer = 'nadam', loss= 'categorical_crossentropy', metrics=['accuracy'])
    print("Model designed")
    return (model,checkpointer,tensorboard)

def fit_model(model,model_path,checkpointer,tensorboard,epoch_size =30,validation_split= 0.2,batch_size = 4):
    model.fit(train_features,train_labels,epochs=epoch_size, verbose=2,
          validation_split=validation_split,batch_size=batch_size,callbacks = [checkpointer,tensorboard])
    model.save(model_path)
    return model

def load_trained_model(model_path):
    return load_model(model_path)
    
def predict_batch_with_exp(predictions,lb_classes,test_feat,test_lab):
    """(batch mode) when the expected data is available"""
    def func(row):
        if row['actual'] == row['pred']:
            val = 'TP'
        elif row['actual'] != row['pred']:
            val = 'FP'
        return val
    pred = []
    actual = []
    
    for d in predictions:
        pred.append(str(lb_classes[np.argmax(d)]))
    for d in test_lab.tolist():
        actual.append(d[0])
    total =sum(1 for pred_v, actual_v in zip(pred, actual) if pred_v == actual_v)
    a = total/len(actual)*100
    print("accuracy: %.2f" % a,"% on the test set")
    pred_df = pd.DataFrame({'sentences':test_feat,'actual':actual,'pred':pred}) 
    pred_df['result'] = pred_df.apply(func, axis=1)
    c_matrix = confusion_matrix(actual,pred)
    return (pred_df,c_matrix)

def predict_batch_without_exp(model,lb_classes,sentences,output_path = 'output.csv'):
    """(batch mode) when you don't have the expected data"""
    sent=[]
    for s in sentences:
        sent.append(get_emb(s).astype('float64'))
    sent =  np.array(sent)
    predict = model.predict(sent)
    pred = []
    for d in predict:
        pred.append(str(lb_classes[np.argmax(d)]))
    df = pd.DataFrame({'sentences':sentences,'pred':pred}) 
    df.to_csv(output_path,sep=',',index = False)
    return df
        
def predict_class(model,lb_classes,sent):
    """(single)Can be used when you want to predict single sentence"""
    s = get_emb(sent).astype('float64')
    s = s.reshape(1,s.shape[0],s.shape[1])
    predict = model.predict(s)
    result = str(lb_classes[np.argmax(predict)])
    print(sent,'---->',result)
    return result


def plot_heatmap(c_matrix):
    sns.heatmap(c_matrix.T,annot = True,
            fmt = 'g',cbar = False, xticklabels = list(lb.classes_),yticklabels =list(lb.classes_) )
    plt.xlabel('TRUE VALUES')
    plt.ylabel('PREDICTED VALUES')
    plt.show()
     


#main code starts here   
if __name__=='__main__':
    """Training"""    
    train_features,train_labels =load_data(train_path)#,head_count = 1000)
    #give head_count as second parameter if you want to train on less sentences 
    train_labels,encoder_path = encode_labels(lb,train_labels,save =True)
    emb_model = load_word2vec(word_emb_model_path) #emb_model created as global used in train_features via get_emb
    train_features = features_to_emb(train_features)
    model,checkpointer,tensorboard = model_design(model_path,seq_len,emb_dim,tensorboard_dir)
    model = fit_model(model,model_path,checkpointer,tensorboard,epoch_size,validation_split,batch_size)


    """prediction"""
    test_feat,test_lab =load_data(test_path)#,head_count = 104)
    test_labels,encoder_path = encode_labels(lb, test_lab,save =False)

    test_features = features_to_emb(test_feat)
    model = load_trained_model(model_path)
    
    lb_classes = list(pickle.load(open(encoder_path, "rb" )))
    predictions = model.predict(test_features)
    """format1"""
    pred_df,c_matrix = predict_batch_with_exp(predictions,lb_classes,test_feat,test_lab)
    pred_df.to_csv('output.csv',index=False)
    plot_heatmap(c_matrix)
    #print(kappa_score)
    #annot to put numbers in each square

    """format2"""
#    pred_df2 = predict_batch_without_exp(model= model, lb_classes=lb_classes, sentences = test_feat.tolist())
    """format3"""
#    predict_class(model = model ,lb_classes = lb_classes,sent="Your test sentence")