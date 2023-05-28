import numpy as np
import torch

class KGDataset(object):
    def __init__(
        self, 
        triples: np.ndarray,
        entity_to_id: dict = None,
        relation_to_id: dict = None
    ):
        self._raw_triples = triples
        if entity_to_id is None:
            entity_to_id = {}
            num_entities = 0
            for triple in triples:
                h, _, t = triple
                if h not in entity_to_id.keys():
                    entity_to_id[h] = num_entities
                    num_entities += 1
                if t not in entity_to_id.keys():
                    entity_to_id[t] = num_entities
                    num_entities += 1
        else:
            num_entities = len(entity_to_id)
                
        if relation_to_id is None:
            relation_to_id = {}
            num_relations = 0
            for triple in triples:
                _, r, _ = triple
                if r not in relation_to_id.keys():
                    relation_to_id[r] = num_relations
                    num_relations += 1

        else:
            num_relations = len(relation_to_id)

        ret = []
        for triple in triples:
            h, r, t = triple
            try:
                h = entity_to_id[h]
                t = entity_to_id[t]
                r = relation_to_id[r]
                ret.append([h, r, t])
            except Exception:
                ret.append([-1, -1, -1])

        self._num_triples = np.array(ret, dtype=np.int32)

        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.num_entities = num_entities
        self.num_relations = num_relations

        id_to_entity = {}
        id_to_relation = {}
        for k, v in entity_to_id.items():
            id_to_entity[v] = k
        
        for k, v in relation_to_id.items():
            id_to_relation[v] = k


    @classmethod
    def load_csv(cls, file_path: str, entity_to_id: dict = None, relation_to_id: dict = None):
        triples = np.loadtxt(file_path, delimiter="\t", dtype=str)
        return cls(triples, entity_to_id=entity_to_id, relation_to_id=relation_to_id)

    def __len__(self):
        return len(self._num_triples)

    def __getitem__(self, key):
        return self._num_triples[key]

    def to_triples_factory(self):
        from pykeen.triples import TriplesFactory
        return TriplesFactory(
            torch.tensor(self._num_triples),
            self.entity_to_id,
            self.relation_to_id,
            create_inverse_triples=False,
            num_entities=self.num_entities,
            num_relations=self.num_relations
        )
    