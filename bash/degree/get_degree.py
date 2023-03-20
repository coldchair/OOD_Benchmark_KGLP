from collections import defaultdict

def get_degree(dataset):
    # {dataset} : <class 'datasets.Dataset'>
    # Return 3 kinds of lists of degrees
    print("Computing the mappings <entity name -> degree> (for in, out and overall degree) in %s training set..." % dataset.name)

    entity_2_in_degree = defaultdict(lambda: 0)
    entity_2_out_degree = defaultdict(lambda: 0)
    entity_2_degree = defaultdict(lambda: 0)

    for (head, relation, tail) in dataset.train_triples:
        entity_2_out_degree[head] += 1
        entity_2_in_degree[tail] += 1
        entity_2_degree[head] += 1
        entity_2_degree[tail] += 1

    return entity_2_in_degree, entity_2_out_degree, entity_2_degree