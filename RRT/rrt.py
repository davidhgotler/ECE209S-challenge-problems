import random
from RRT.rrt_utils import *
from RRT.map import *

def fft(map: Map, start_point, goal, step_distance, obstacles, dest_thr):
    map_xsize = 1000
    map_ysize = 1000 # TBD


    vertices = []
    # vertices.append(Node(start_point[0], start_point[1]))
    while not map._solved:
        rand_x = random.random() * map_xsize
        rand_y = random.random() * map_ysize
        s_rand = Node((rand_x, rand_y))

        # find the nearest node in the tree to s_rand
        nearest_vertex = find_nearest_vertex(vertices, s_rand)

        # move from nearest_vertex towards s_rand
        sin = (s_rand.coord[0] - nearest_vertex.coord[0]) / distance(nearest_vertex, s_rand)
        cos = (s_rand.coord[1] - nearest_vertex.coord[1]) / distance(nearest_vertex, s_rand)
        delta_x = step_distance * sin
        delta_y = step_distance * cos
        s_new = (nearest_vertex[0] + delta_x, nearest_vertex[1] + delta_y)
        nearest_vertex.children.append(s_new)
        
        # check if new_vertex is valid
        if map.is_inside_obstacles(s_new): # inside obstacles
            continue
        elif map.is_collision_with_obstacles([nearest_vertex, s_new]): # collision with obstacles
            continue
        elif map.is_inside_obstacles(s_new): # inside obstacles
            vertices.append(s_new)
            if distance(s_new, goal) < dest_thr:
                map._solved = True
        

def find_nearest_vertex(vertices:list[Node], s_rand:Node) -> Node:
    nearest = s_rand
    min_dist = float('inf')
    for vertex in vertices:
        dist = distance(vertex, s_rand)
        if dist < min_dist:
            nearest = vertex
            min_dist = dist
    return nearest

def distance(v1:Node, v2:Node):  
    return ((v1.coord[0] - v2.coord[0])**2 + (v1.coord[1] - v2.coord[1])**2)**0.5

