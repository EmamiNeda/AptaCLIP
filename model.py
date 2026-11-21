"""Exact projection heads, interaction classifier and squared-Euclidean triplet loss from notebook cell 2."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def triplet_loss_matrix(margin=2.0):
    def loss(y_true,y_pred):
        dim=y_pred.shape[1]//3
        anchor=y_pred[:,:dim]; positive=y_pred[:,dim:2*dim]; negative=y_pred[:,2*dim:]
        pdist=tf.reduce_sum(tf.square(anchor-positive),axis=1)
        ndist=tf.reduce_sum(tf.square(anchor-negative),axis=1)
        return tf.reduce_mean(tf.maximum(pdist-ndist+margin,0.0))
    return loss

def protein_projection_matrix(x):
    x=layers.Dense(512,activation='relu')(x); x=layers.BatchNormalization()(x); x=layers.Dropout(.3)(x)
    x=layers.Dense(256,activation='relu')(x); x=layers.BatchNormalization()(x); x=layers.Dropout(.2)(x)
    x=layers.Dense(128,activation='linear')(x)
    return layers.Lambda(lambda t:tf.math.l2_normalize(t,axis=1))(x)

def aptamer_projection_matrix(d):
    inp=layers.Input((d,)); x=layers.Dense(512,activation='relu')(inp); x=layers.BatchNormalization()(x); x=layers.Dropout(.3)(x)
    x=layers.Dense(256,activation='relu')(x); x=layers.BatchNormalization()(x); x=layers.Dropout(.2)(x)
    x=layers.Dense(128,activation='linear')(x)
    return keras.Model(inp,layers.Lambda(lambda t:tf.math.l2_normalize(t,axis=1))(x),name='aptamer_projection')

def build_model_matrix(pd_,ad_):
    pi=layers.Input((pd_,),name='protein_input')
    ai=layers.Input((ad_,),name='positive_aptamer_input')
    bi=layers.Input((ad_,),name='negative_aptamer_input')
    ci=layers.Input((ad_,),name='classification_aptamer_input')
    fi=layers.Input((1,),name='rna_dna_input')
    ap=aptamer_projection_matrix(ad_)
    pe=layers.Lambda(lambda x:x,name='prot_emb')(protein_projection_matrix(pi))
    ae=layers.Lambda(lambda x:x,name='pos_emb')(ap(ai))
    be=layers.Lambda(lambda x:x,name='neg_emb')(ap(bi))
    ce=layers.Lambda(lambda x:x,name='cls_emb')(ap(ci))
    tri=layers.Concatenate(name='triplet_output')([pe,ae,be])
    x=layers.Concatenate(name='joint_input')([pe,ce,fi])
    x=layers.Dense(128,activation='relu',name='dense_128')(x); x=layers.Dropout(.3)(x)
    inter=layers.Dense(1,activation='sigmoid',name='interaction')(x)
    return keras.Model([pi,ai,bi,ci,fi],[inter,tri])

