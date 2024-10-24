import numpy as np

class Tree:
    def __init__(self,coord,children = []) -> None:
        self._coord = coord
        self._children = children

    @property
    def coord(self):
        return self._coord

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self,children):
        self._children = children
    
    def add_child(self,child):
        if type(child) is list:
            self._children += child
        else:
            self._children += [child]