import os
from utils.dir import make_dir
from utils.evalution import mrr_score
from degree.get_degree import get_degree
from matplotlib import pyplot as plt

K = 100

def draw(dataset, ranks, img_dir):
    arr = []
    entity_in_degrees, entity_out_degrees, entity_degrees = get_degree(dataset)
    n = 0
    print(len(ranks))
    for (head, relation, tail) in dataset.test_triples:
        arr.append([
            entity_out_degrees[head] + entity_in_degrees[tail],
            [ranks[n, 0], ranks[n, 1]]
        ])
        n += 1
    def myKey(elem):
        return elem[0]
    arr.sort(reverse=False, key=myKey)
    plt_x = []
    plt_y = []
    blo = int(n / K)
    for i in range(0, K):
        l = blo * i
        r = blo * (i + 1) - 1
        if (i == K - 1):
            r = n - 1

        now_ranks = []
        for j in range(l, r + 1):
            now_ranks.append(arr[j][1])
        mrr = mrr_score(now_ranks)

        plt_x.append(i)
        plt_y.append(mrr)

    plt.clf()
    plt.scatter(plt_x, plt_y)
    plt.xlabel("index")
    plt.ylabel("mrr")
    make_dir(img_dir)
    plt.savefig(os.path.join(img_dir, "s_plus_o_100_buckets.svg"))
