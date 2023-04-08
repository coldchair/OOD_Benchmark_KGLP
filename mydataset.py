import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class MyDataset:
    def __init__(self, name="transE_NELL995"):
        NAME = name
        train_dataset = np.loadtxt(f"models/{NAME}/train.txt", delimiter='\t', dtype=str)
        valid_dataset = np.loadtxt(f"models/{NAME}/valid.txt", delimiter='\t', dtype=str)
        test_dataset = np.loadtxt(f"models/{NAME}/test.txt", delimiter='\t', dtype=str)
        result = np.loadtxt(f"models/{NAME}/ranks.csv", delimiter=',')
        entity = set()

        entity2degree_in_all = defaultdict(lambda: 0)
        entity2degree_out_all = defaultdict(lambda: 0)
        entity2degree_sum_all = defaultdict(lambda: 0)

        entity2degree_in_train = defaultdict(lambda: 0)
        entity2degree_out_train = defaultdict(lambda: 0)
        entity2degree_sum_train = defaultdict(lambda: 0)

        entity2degree_in_test = defaultdict(lambda: 0)
        entity2degree_out_test = defaultdict(lambda: 0)
        entity2degree_sum_test = defaultdict(lambda: 0)

        for item in train_dataset:
            entity2degree_in_all[item[2]] += 1
            entity2degree_out_all[item[0]] += 1
            entity2degree_sum_all[item[0]] += 1
            entity2degree_sum_all[item[2]] += 1

            entity2degree_in_train[item[2]] += 1
            entity2degree_out_train[item[0]] += 1
            entity2degree_sum_train[item[0]] += 1
            entity2degree_sum_train[item[2]] += 1

            entity.add(item[0])
            entity.add(item[2])

        for item in valid_dataset:
            entity2degree_in_all[item[2]] += 1
            entity2degree_out_all[item[0]] += 1
            entity2degree_sum_all[item[0]] += 1
            entity2degree_sum_all[item[2]] += 1

            entity.add(item[0])
            entity.add(item[2])

        for item in valid_dataset:
            entity2degree_in_all[item[2]] += 1
            entity2degree_out_all[item[0]] += 1
            entity2degree_sum_all[item[0]] += 1
            entity2degree_sum_all[item[2]] += 1

            entity2degree_in_test[item[2]] += 1
            entity2degree_out_test[item[0]] += 1
            entity2degree_sum_test[item[0]] += 1
            entity2degree_sum_test[item[2]] += 1
            
            entity.add(item[0])
            entity.add(item[2])

        self.entity2degree_in_train = entity2degree_in_train
        self.entity2degree_out_train = entity2degree_out_train
        self.entity2degree_sum_train = entity2degree_sum_train
        self.entity2degree_in_test = entity2degree_in_test
        self.entity2degree_out_test = entity2degree_out_test
        self.entity2degree_sum_test = entity2degree_sum_test

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.result = result
        self.entity = entity
    
    def get_degree(self, entity:str, degree_type: str):
        if degree_type == 'none':
            return 0
        if degree_type == 'in': 
            return self.entity2degree_in_train[entity]
        if degree_type == 'out':
            return self.entity2degree_out_train[entity]
        if degree_type == 'all':
            return self.entity2degree_sum_train[entity]

    @staticmethod
    def get_buckets_result(buckets):
        mrr_list = []
        hits1_list = []
        hits10_list = []
        hits100_list = []
        for bucket in buckets:
            n = len(bucket)
            if n == 0:
                mrr_list.append(np.nan)
                hits1_list.append(np.nan)
                hits10_list.append(np.nan)
                hits100_list.append(np.nan)
                continue

            mrr_sum = 0
            hits1_sum = 0
            hits10_sum = 0
            hits100_sum = 0
            for ret in bucket:
                mrr_sum += 1.0 / ret
                hits1_sum += (ret == 1)
                hits10_sum += (ret <= 10)
                hits100_sum += (ret <= 100)
            mrr_list.append(mrr_sum/n)
            hits1_list.append(hits1_sum/n)
            hits10_list.append(hits10_sum/n)
            hits100_list.append(hits100_sum/n)

        return mrr_list, hits1_list, hits10_list, hits100_list
    
    def plot_degree(
        self,
        result_type: str,  # head, tail, all
        head_degree_type: str, # none, in, out, all
        tail_degree_type: str, # none, in, out, all
        intervels: list = None, # intervel for division
        buckets: int = 100, # num of buckets for average division,
        mrr: bool = True,
        hits1: bool = True,
        hits10: bool = True,
        hits100: bool = True,
        save_path: str = 'fig.png'
    ):
        head_ret = False
        tail_ret = False
        if result_type == 'head':
            head_ret = True
        if result_type == 'tail':
            tail_ret = True
        if result_type == 'all':
            head_ret = True
            tail_ret = True

        ret_list = []
        for i, item in enumerate(self.test_dataset):
            if head_ret == True:
                ret_list.append((
                    self.result[i][0],
                    self.get_degree(item[0], head_degree_type) \
                        + self.get_degree(item[2], tail_degree_type)
                ))       
            if tail_ret == True:
                ret_list.append((
                    self.result[i][1],
                    self.get_degree(item[0], head_degree_type) \
                        + self.get_degree(item[2], tail_degree_type)
                ))
        
        ret_list.sort(key=lambda x: x[1])

        if intervels is None:
            intervels = [0, 1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

        intervel_cur = 0
        intervel_buckets = [[] for _ in range(len(intervels) + 1)]
        for item in ret_list:
            while intervel_cur < len(intervels) \
                and item[1] > intervels[intervel_cur]:
                    intervel_cur += 1
            intervel_buckets[intervel_cur].append(item[0])

        plt.figure(figsize=(6,20))
        xs = intervels + [intervels[-1] + 10]
        mrr_list, hits_1_list, hits_10_list, hits_100_list = self.get_buckets_result(intervel_buckets)
        plt.subplot(3, 1, 1)
        plt.plot(xs, mrr_list, color='red', label='mrr')
        plt.plot(xs, hits_1_list, color='blue', label='hits1')
        plt.plot(xs, hits_10_list, color='green', label='hits10')
        plt.plot(xs, hits_100_list, color='yellow', label='hits100')
        plt.legend()
        plt.title('intervel bucket interpolation')

        xs = range(len(intervel_buckets))
        plt.subplot(3, 1, 2)
        plt.plot(xs, mrr_list, color='red', label='mrr')
        plt.plot(xs, hits_1_list, color='blue', label='hits1')
        plt.plot(xs, hits_10_list, color='green', label='hits10')
        plt.plot(xs, hits_100_list, color='yellow', label='hits100')
        plt.legend()
        plt.title('intervel bucket idx')

        N = buckets
        average_buckets = [[] for _ in range(N)]
        bucket_size = len(ret_list) // N
        for i in range(N):
            st = i * bucket_size
            ed = (i+1) * bucket_size
            if i == N - 1:
                ed = len(ret_list)
            for j in range(st, ed):
                average_buckets[i].append(ret_list[j][0])
        
        mrr_list, hits_1_list, hits_10_list, hits_100_list = self.get_buckets_result(average_buckets)
        plt.subplot(3, 1, 3)
        xs = list(range(N))
        plt.plot(xs, mrr_list, color='red', label='mrr')
        plt.plot(xs, hits_1_list, color='blue', label='hits1')
        plt.plot(xs, hits_10_list, color='green', label='hits10')
        plt.plot(xs, hits_100_list, color='yellow', label='hits100')
        plt.legend()
        plt.title('average bucket')
        plt.subplots_adjust(hspace=0.5, top=0.9, bottom=0.1)
        plt.savefig(save_path)
        plt.show()
        plt.clf()
        
    def get_degree_dist(self):
        train_entity_degree = []
        for entity in self.entity:
            train_entity_degree.append(
                (entity, self.entity2degree_sum_train[entity])
            )
        
        train_entity_degree.sort(key=lambda x: x[1])
        ret = []
        ratio = []
        for entity, degree in train_entity_degree:
            ret.append(
                self.entity2degree_sum_test[entity]
            )
            if degree == 0:
                ratio.append(np.nan)
            else:
                ratio.append(
                    self.entity2degree_sum_test[entity] / degree
                )
        xs = list(range(len(ret)))
        plt.plot(xs, ret)
        plt.show()
        plt.clf()
        plt.plot(xs, ratio)
        plt.show()