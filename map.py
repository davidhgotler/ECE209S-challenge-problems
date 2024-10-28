import json
import threading
from rrt_utils import *
import numpy as np

"""
Features: start location, goald, location, obstacles, path


"""

class Map:

    # init the map with the given file name
    def __init__(self, fname):
        self.fname = fname
        with oepn(fnam) as configfile:
            config = json.loads(configfile.read())
            self.width = config['width']
            self.height = config['height']

            # initalize the features as empty
            self._start = Node(tuple(config['start']))
            self._goals = [Node(tuple(coord)) for coord in config['goals']]
            self._obstacles = []
            self._nodes = []  # node in RRT
            self._node_paths = []  # edge in RRT
            self._solved = False
            self._smooth_path = []
            self._smoothed = False
            self._restarts = []

            # Read in obstacles
            for obstacle in config['obstacles']:
                self._obstacles.append([Node(tuple(coord)) for coord in obstacle])

            # For coordination with visualization
            self.lock = threading.Lock()
            self.updated = threading.Event()
            self.changes = []    

    # helper function to get orientation
    def get_orientation(p, q, r):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if is_zero(val):
            # colinear
            return 0
        elif val > 0:
            # clockwise
            return 1
        else:
            # counter-clockwise
            return 2
        
    def is_intersect(p1, q1, p2, q2):
        """
        Check the line from p1 to q1 intersects with the line from p2 to q2.
        """
        o1 = get_orientation(p1, q1, p2)
        o2 = get_orientation(p1, q1, q2)
        o3 = get_orientation(p2, q2, p1)
        o4 = get_orientation(p2, q2, q1)

        if (o1 != o2 and o3 != o4):
            return True
        if (is_zero(o1) and is_on_segment(p1, p2, q1)):
            return True
        if (is_zero(o2) and is_on_segment(p1, q2, q1)):
            return True
        if (is_zero(o3) and is_on_segment(p2, p1, q2)):
            return True
        if (is_zero(o4) and is_on_segment(p2, q1, q2)):
            return True
        return False    

    # this function check if the node is within the map
    def is_inbound(self, node):
        if((node.coord.x >= 0) and (node.coord.y >= 0) 
           and (node.coord.x < self.width) and (node.coord.y < self.height)):
           return True
        return False

    # this function checks if the given line collides with any obstacles
    def is_collision_with_obstacles(self, line_segment):
        obstacles = self._obstacles
        line_start, line_end = line_segment
        for obstacle in obstacles:
            num_sides = len(obstacle)
            for idx in range(num_sides):
                side_start, side_end = obstacle[idx], obstacle[(idx + 1) % num_sides]
                if is_intersect(line_start, line_end, side_start, side_end):
                    return True
        return False   

    def is_inside_obstacles(self, node):
        obstacles = self._obstacles
        for obstcale in obstacles:
            num_sides = len(obstcale)
            is_inside = True
            for idx in range(num_sides):
                side_start, side_end = obstcale[idx], obstcale[(idx+1) % num_sides]
                if get_orientation(side_start, side_end, node) == 2:
                    is_inside = False
                    break
            if is_inside:
                return True
        return False

    def get_size(self):
        return self.width, self.height
    
    def get_nodes(self):
        return self._nodes
    
    def get_goals(self):
        return self._goals
    
    def get_restarts(self):
        return self._restarts
    
    def reset(self, node):
        """Reset the map by clearing the existing nodes and paths, 
           and set the new start to the node
        """
        self.set_start(node)
        self.reset_paths()
        self.add_restart()

    def add_restart(self):
        self._restarts.append(self.get_start())

    def get_num_nodes(self):
        """Return number of nodes in RRT
        """
        return len(self._nodes)

    def set_start(self, node):
        """Set the start cell

            Arguments:
            node -- grid coordinates of start cell
        """
        if self.is_inside_obstacles(node) or (not self.is_inbound(node)):
            print("start is not updated since your start is not legitimate\nplease try another one\n")
            return
        self.lock.acquire()
        self._start = Node((node.x, node.y))
        self.updated.set()
        self.changes.append('start')
        self.lock.release()

    def get_start(self):
        """Get start
        """
        return self._start
    
        