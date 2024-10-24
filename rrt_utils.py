import numpy as np

class Node:
    def __init__(self,coord,children = None) -> None:
        self.coord = coord
        self.children = [] if children is None else children
    
    def __repr__(self) -> str:
        return f"<Node at coord = {self.coord}, children = {self.children}>"
