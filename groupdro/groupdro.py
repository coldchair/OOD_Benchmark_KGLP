import numpy as np
import ampligraph
import tensorflow as tf
import pandas as pd
import os

from ampligraph.datasets import load_from_csv


from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.utils import save_model, restore_model

model = ScoringBasedEmbeddingModel(k=350,
                                   eta=30,
                                   scoring_type='TransE',
                                   seed=0)

from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=1.5e-4)


info_list=[125*2,529*2,2649*2,69*2]
#info_list=[3372*2]



#tf.config.experimental_run_functions_eagerly(True)



adj=2



self_step_size = 0.01
self_normalize_loss = False

self_n_groups = 4
#self_n_groups = 1
self_group_counts = tf.constant([125*2,529*2,2649*2,69*2])
#self_group_counts = tf.constant([3372*2])

if adj is not None:
    self_adj = tf.convert_to_tensor(adj, dtype=tf.float32)
else:
    self_adj = tf.zeros(self_n_groups, dtype=tf.float32)


# quantities maintained throughout training
self_adv_probs = tf.ones(self_n_groups, dtype=tf.float32) / tf.cast(self_n_groups, dtype=tf.float32)

#init
group_idx_list=[]
for i in range(len(info_list)):
    for j in range(info_list[i]):
        group_idx_list.append(i)
group_idx=tf.constant(group_idx_list)














# class NLLMulticlass(Loss):
#     r"""Multiclass Negative Log-Likelihood loss.

#     Introduced in :cite:`chen2015`, this loss can be used when both the subject and objects are corrupted
#     (to use it, pass ``corrupt_sides=['s,o']`` in the embedding model parameters).

#     This loss was re-engineered in :cite:`kadlecBK17` where only the object was corrupted to get improved
#     performance (to use it in this way pass ``corrupt_sides ='o'`` in the embedding model parameters).

#     .. math::

#         \mathcal{L(X)} = -\sum_{x_{e_1,e_2,r_k} \in X} log\,p(e_2|e_1,r_k)
#          -\sum_{x_{e_1,e_2,r_k} \in X} log\,p(e_1|r_k, e_2)

#     Example
#     -------
#     >>> import ampligraph.latent_features.loss_functions as lfs
#     >>> loss = lfs.NLLMulticlass({'reduction': 'mean'})
#     >>> isinstance(loss, lfs.NLLMulticlass)
#     True

#     >>> loss = lfs.get('multiclass_nll')
#     >>> isinstance(loss, lfs.NLLMulticlass)
#     True

#     """

#     def __init__(self, loss_params={}, verbose=False):
#         """Initialize the loss.

#         Parameters
#         ----------
#         loss_params : dict
#             Dictionary of loss-specific hyperparams:

#             - `"reduction"`: (str) - Specifies whether to `"sum"` or take the `"mean"` of loss per sample w.r.t. \
#              corruption (default: `"sum"`).

#         """
#         super().__init__(loss_params, verbose)

#     def _init_hyperparams(self, hyperparam_dict={}):
#         """Verifies and stores the hyperparameters needed by the algorithm.

#         Parameters
#         ----------
#         hyperparam_dict : dict
#             The Loss will check the keys to get the corresponding parameters.
#         """
#         pass

#     @tf.function(experimental_relax_shapes=True)
#     def _apply_loss(self, scores_pos, scores_neg):
#         print("hhhhh")
#         score_pos = tf.clip_by_value(score_pos, clip_value_min=-75.0, clip_value_max=75.0)
#         score_neg = tf.clip_by_value(score_neg, clip_value_min=-75.0, clip_value_max=75.0)

#         neg_exp = tf.exp(score_neg)
#         pos_exp = tf.exp(score_pos)
#         softmax_score = pos_exp / (tf.reduce_sum(neg_exp, 0) + pos_exp)
#         per_sample_losses = -tf.math.log(softmax_score)
#         #multiclass_nll        
        
        
        
#         group_map = tf.cast(tf.equal(group_idx, tf.expand_dims(tf.range(self_n_groups), axis=1)), dtype=tf.float32)#
#         print("gggg",group_map)
#         group_count = tf.reduce_sum(group_map, axis=1)
#         group_denom = group_count + tf.cast(group_count == 0, dtype=tf.float32)  # avoid nans
#         group_loss = tf.reduce_sum(tf.matmul(group_map, tf.reshape(per_sample_losses, (-1, 1))), axis=1) / group_denom#
        
        
        


#         # compute overall loss btl=false is_robust=1
#         #actual_loss, weights = self.compute_robust_loss(group_loss, group_count)


#         adjusted_loss = tf.identity(group_loss)#copy?
#         if tf.reduce_all(self_adj > 0):
#             print("ssss",self_adj.numpy())
#             adjusted_loss += tf.cast(self_adj, tf.float32) / tf.sqrt(tf.cast(self_group_counts, tf.float32))
#         if self_normalize_loss:
#             adjusted_loss = adjusted_loss / tf.reduce_sum(adjusted_loss)
#         self_adv_probs = self_adv_probs * tf.exp(self_step_size * adjusted_loss)
#         self_adv_probs = self_adv_probs / tf.reduce_sum(self_adv_probs)

#         robust_loss = tf.reduce_sum(group_loss * self_adv_probs)
#         #return robust_loss, self.adv_probs


#         return robust_loss

















def myloss(score_pos, score_neg):
    global self_adv_probs
    score_pos = tf.clip_by_value(score_pos, clip_value_min=-75.0, clip_value_max=75.0)
    score_neg = tf.clip_by_value(score_neg, clip_value_min=-75.0, clip_value_max=75.0)

    neg_exp = tf.exp(score_neg)
    pos_exp = tf.exp(score_pos)
    softmax_score = pos_exp / (tf.reduce_sum(neg_exp, 0) + pos_exp)
    per_sample_losses = -tf.math.log(softmax_score)
    #multiclass_nll        
    
    
    
    group_map = tf.cast(tf.equal(group_idx, tf.expand_dims(tf.range(self_n_groups), axis=1)), dtype=tf.float32)#
    group_count = tf.reduce_sum(group_map, axis=1)
    group_denom = group_count + tf.cast(group_count == 0, dtype=tf.float32)  # avoid nans
    #print(group_denom)
    group_loss = tf.reduce_sum(tf.matmul(group_map, tf.reshape(per_sample_losses, (-1, 1))), axis=1) / group_denom#
    
    
    


    # compute overall loss btl=false is_robust=1
    #actual_loss, weights = self.compute_robust_loss(group_loss, group_count)


    adjusted_loss = tf.identity(group_loss)#copy?
    if tf.reduce_all(self_adj > 0):
        adjusted_loss += tf.cast(self_adj, tf.float32) / tf.sqrt(tf.cast(self_group_counts, tf.float32))
    if self_normalize_loss:
        adjusted_loss = adjusted_loss / tf.reduce_sum(adjusted_loss)
    self_adv_probs = self_adv_probs * tf.exp(self_step_size * adjusted_loss)
    self_adv_probs = self_adv_probs / tf.reduce_sum(self_adv_probs)
    #hh=self_adv_probs.numpy()
    #print(hh)
    #print(group_loss)
    #print(self_adv_probs)
    robust_loss = tf.reduce_sum(group_loss * self_adv_probs)*1000
    #print("rrr",robust_loss)
    #return robust_loss, self.adv_probs

    return robust_loss



































# def myloss(score_pos, score_neg, margin=0.5, lambda_irm=0):
#     score_pos = tf.clip_by_value(score_pos, clip_value_min=-75.0, clip_value_max=75.0)
#     score_neg = tf.clip_by_value(score_neg, clip_value_min=-75.0, clip_value_max=75.0)

#     pos_exp = tf.exp(score_pos)
#     neg_exp = tf.exp(score_neg)

#     softmax_score = pos_exp / (tf.reduce_sum(neg_exp, 0) + pos_exp)

#     loss = -tf.math.log(softmax_score)
    
#     loss_value = tf.reduce_sum(loss, 0)

#     loss_split = tf.split(loss, num_or_size_splits=[1961, 2607, 2379, 3059, 177, 611], axis=0)

#     # loss_split = tf.reduce_mean(loss_split, 1)

#     # tf.print(loss.shape)

#     loss_split[0] = tf.reduce_mean(loss_split[0], 0)
#     loss_split[1] = tf.reduce_mean(loss_split[1], 0)
#     loss_split[2] = tf.reduce_mean(loss_split[2], 0)
#     loss_split[3] = tf.reduce_mean(loss_split[3], 0)
#     loss_split[4] = tf.reduce_mean(loss_split[4], 0)
#     loss_split[5] = tf.reduce_mean(loss_split[5], 0)
    
#     loss_split = tf.stack(loss_split, axis=0)

#     loss_value = 2000 * (tf.reduce_sum(loss_split, 0) + tf.math.reduce_variance(loss_split, 0) * lambda_irm)

#     # loss_value = 2000* loss_split[5]

#     return loss_value

import ampligraph.latent_features.loss_functions as lfs
# loss = lfs.NLLMulticlass({'reduction': 'mean'})

regularizer = get_regularizer('LP', {'p': 2, 'lambda': 1e-4})

initializer = tf.keras.initializers.GlorotNormal(seed=0)

model.compile(loss=myloss,
              optimizer=optimizer,
              entity_relation_regularizer=regularizer,
              entity_relation_initializer=initializer)

tf.keras.utils.get_custom_objects()['myloss'] = myloss

from tqdm import *

X_train_shuffled = []
batch_size_number=3372*2
number_groups=160
X_split=[]


X_train=pd.read_csv(
    'train.txt',
    sep='\t',
    header=None,
    names=None,
    dtype=str,
).values
print(len(X_train))
for i in range(4):
    name='split0.txt'.replace('0',str(i))
    X_splitnow=pd.read_csv(
        name,
        sep='\t',
        header=None,
        names=None,
        dtype=str,
    ).values
    X_split.append(np.array(X_splitnow, dtype=str).tolist() )
    print(len(X_split[i]))
import numpy as np
import random
def my_shuffle_dataset():
    global X_train_shuffled
    X_train_shuffled = []
    #print("st")
    #print(X_split[0][:10])
    random.shuffle(X_split[0])
    #print(X_split[0][:10])
    #print("ed")
    random.shuffle(X_split[1])
    random.shuffle(X_split[2])
    random.shuffle(X_split[3])
    #print(X_split[0][:10])
    info_list___=[125*2,529*2,2649*2,69*2]
    for i in range(number_groups):
        for t in range(4):
           for j in range(info_list___[t]): 
               X_train_shuffled.append(X_split[t][i * info_list___[t] + j])
        
        # for j in range(1961):
        #     X_train_shuffled.append(X_split1[i * 1961 + j])
        # for j in range(2607):
        #     X_train_shuffled.append(X_split2[i * 2607 + j])
        # for j in range(2379):
        #     X_train_shuffled.append(X_split3[i * 2379 + j])
        # for j in range(3059):
        #     X_train_shuffled.append(X_split4[i * 3059 + j])
        # for j in range(177):
        #     X_train_shuffled.append(X_split5[i * 177 + j])
        # for j in range(611):
        #     X_train_shuffled.append(X_split6[i * 611 + j])
    X_train_shuffled = np.array(X_train_shuffled, dtype=str)    
# example_name = "TransE_nostop_4.pkl"
#model = restore_model('/workspace/modelsave/TransE_test3_5.pkl')

from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
import json
for _ in tqdm(range(40)):
    print('Big Epoch %d/40\n' % (_ + 1))

    # print(X_train_shuffled[:10])
    # if _%100==0:
    #     my_shuffle_dataset()
    my_shuffle_dataset()
    #print(self_adv_probs.numpy())
    #print(len(X_train_shuffled))
    #print(X_train_shuffled)
    #print(X_train_shuffled[:10])
    model.fit(X_train_shuffled, batch_size=batch_size_number, epochs=100, verbose=True, shuffle=False)

    if _ % 1 == 0:
        example_name = "TransE_test3_%d.pkl" % ((_ + 1) / 1)
        save_model(model,model_name_path= '/workspace/modelsave2/'+example_name)

        print('Evaluating ...')

        test_file_list = ['test.txt']

        with open('result.csv', 'a') as w:

            w.write(str(_ + 1) + ' Epoch,')

            for name in test_file_list:

                X_test = pd.read_csv(
                        name,
                        sep='\t',
                        header=None,
                        names=None,
                        dtype=str,
                    ).values

                filter = {'test' : np.concatenate((X_train, X_test))}

                ranks = model.evaluate(X_test,
                                    use_filter=filter,
                                    corrupt_side='s,o',
                                    verbose=True)

                

                print('Evaluating ' + name)
                mrr = mrr_score(ranks)
                print("MRR: %.2f" % (mrr))

                hits_10 = hits_at_n_score(ranks, n=10)
                print("Hits@10: %.2f" % (hits_10))
                hits_3 = hits_at_n_score(ranks, n=3)
                print("Hits@3: %.2f" % (hits_3))
                hits_1 = hits_at_n_score(ranks, n=1)
                print("Hits@1: %.2f" % (hits_1))

                w.write(','.join([name, str(mrr), str(hits_10), str(hits_3), str(hits_1), '']))

            w.write('\n')

