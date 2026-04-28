import os
import math
import json
from typing import List, Optional
from collections import defaultdict
import random


class LabelTree:

    def __init__(self, tree_instance_dir, internal_node_strategy='echo') -> None:
        # get corresponding decorator
        self.internal_node_decorators = {
            'echo': self._echo_decorator, 
            'special_token': self._special_token_decorator
        }
        assert internal_node_strategy in self.internal_node_decorators
        self.internal_node_strategy = internal_node_strategy
        self.internal_node_decorator = self.internal_node_decorators[internal_node_strategy]

        # instantiate tree instance
        self.tree_instance = json.load(open(os.path.join(tree_instance_dir, 'relation_tree_name.json')))

        # quick access for some properties
        self.label2paths   = defaultdict(list)
        self.path2children = defaultdict(set)
        self.internal_nodes = []
        self.root_nodes = []
        self.max_depth = -math.inf

        if not isinstance(self.tree_instance, (list, tuple)):
            self.tree_instance = [ self.tree_instance ]
        for tree_ins in self.tree_instance:
            self.root_nodes.append(tree_ins['name'])
            # enrich the content of properties above
            self._find_all_path(tree_ins, [])
        
        self.root_nodes = sorted(self.root_nodes, key=lambda x: x in ['no valid relation', 'not available', 'invalid relation', 'no'])
        self.pos_start_node, self.na_node = self.root_nodes

    def _echo_decorator(self, node_name):
        return node_name

    def _special_token_decorator(self, node_name):
        return '<|' + node_name + '|>'

    def _find_all_path(self, root, path):
        assert isinstance(root, dict)
        self.max_depth = max(self.max_depth, len(path) + 1)

        if 'children' not in root:
            self.path2children[tuple(path)].add(root['name'])
            path.append(root['name'])
            self.label2paths[root['name']].append(path)
            return 
        
        internal_node = self.internal_node_decorator(root['name'])
        self.path2children[tuple(path)].add(internal_node)
        path.append(internal_node)
        self.internal_nodes.append(internal_node)

        for child in root['children']:
            self._find_all_path(child, path.copy())
        
    # def get_path_by_label(self, label: str, return_all: bool = False) -> None | List[str] | List[List[str]]:
    def get_path_by_label(self, label: str, return_all: bool = False):
        assert self.label2paths, 'LabelTree instance is not initialized. '
        if label not in self.label2paths:
            print(f'OOD {label = }')
            return None
        paths = []
        for p in self.label2paths[label]: 
            tuple_p = tuple(p)
            if tuple_p not in paths:
                paths.append(tuple_p)
        paths = list(map(list, paths))  # tuple -> list
        if return_all:
            return paths
        path_idx = random.randint(0, len(paths) - 1)
        return paths[path_idx]
    
    @property
    def root_special_token(self) -> str:
        assert self.internal_nodes, 'LabelTree instance is not initialized. '
        assert self.internal_node_strategy == 'special_token'
        return self.internal_nodes[0]

    @property
    def all_special_tokens(self) -> List[str]:
        assert self.internal_nodes, 'LabelTree instance is not initialized. '
        assert self.internal_node_strategy == 'special_token'
        return self.internal_nodes

    # def children_of(self, path: list | tuple) -> List[str]:
    def children_of(self, path) -> List[str]:
        assert self.path2children, 'LabelTree instance is not initialized. '
        assert isinstance(path, (list, tuple)), f'{type(path) = }'
        if isinstance(path, list):
            path = tuple(path)
        if path not in self.path2children:
            if path[-1] not in self.label2paths:  # intermediate nodes
                print(f'OOD {path = }')
            return []
        return list(self.path2children[path])
        
