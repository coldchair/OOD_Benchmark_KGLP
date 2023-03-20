# This file defines class Dataset to save train, valid, test

import os

class Dataset:
    def __init__(self, path, name, separator="\t"):
        # {path} is the path of "models" folder
        # {name} is the folder name under the models directory
        # The default {separator} is "\t"

        self.path = path
        self.name = name
        self.home = os.path.join(path, self.name)

        if not os.path.isdir(self.home):
            raise Exception("Folder %s does not exist" % self.home)

        self.train_path = os.path.join(self.home, "train.txt")
        self.valid_path = os.path.join(self.home, "valid.txt")
        self.test_path = os.path.join(self.home, "test.txt")

        self.entities = set()
        self.relationships = set()

        print("Reading train triples for %s..." % self.name)
        self.train_triples = self._read_triples(self.train_path, separator)
        print("Reading validation triples for %s..." % self.name)
        self.valid_triples = self._read_triples(self.valid_path, separator)
        print("Reading test triples for %s..." % self.name)
        self.test_triples = self._read_triples(self.test_path, separator)

    def _read_triples(self, triples_path, separator="\t"):
        triples = []
        with open(triples_path, "r") as triples_file:
            lines = triples_file.readlines()
            for line in lines:
                #line = html.unescape(line)
                head, relationship, tail = line.strip().split(separator)
                triples.append((head, relationship, tail))
                self.entities.add(head)
                self.entities.add(tail)
                self.relationships.add(relationship)
        return triples