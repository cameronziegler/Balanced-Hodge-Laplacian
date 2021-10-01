import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import linalg
from tempfile import NamedTemporaryFile
import matplotlib.patches as mpatches
from matplotlib.offsetbox import (DrawingArea, OffsetImage,AnnotationBbox)
import math
import os

def sortSimplicies(s):
    '''Given a list of simplicies of a given dimension, returns the sorted list
    of sorted simplicies.  This ensures all simplex labels are standardized
    as ascending.'''
    s2 = s.copy()
    for i in range(len(s)):
        s2[i] = sorted(s[i])
    return sorted(s2)

def calcB1(nodes, edges):
    '''Calculates the first boundary matrix for a simplicial complex.  
    Edges should be sorted with ascending labels.'''
    B1 = np.zeros((nodes, len(edges)))
    i = 0
    for e in edges:
        B1[e[0]-1,i] = -1
        B1[e[1]-1, i] = 1
        i = i+1
    return B1

def calcB2(edges, triangles):
    '''Calculates the second boundary matrix for a simplicial complex.  
    Edges and triangles should be sorted with ascending labels.'''
    B2 = np.zeros((len(edges), len(triangles)))
    i = 0
    for t in triangles:
        a = [t[0], t[1]]
        b = [t[1], t[2]]
        c = [t[0], t[2]]
        ai = edges.index(a)
        bi = edges.index(b)
        ci = edges.index(c)
        B2[ai, i] = 1
        B2[bi, i] = 1
        B2[ci, i] = -1
        i = i+1
    return B2

def triangleLocation(triangles, pos):
    '''Given a list of triangles and the (x, y) positions of nodes, this 
    finds the relevant upper and lower edges of the triangles in order to 
    plot them.'''
    x = []
    yLower = []
    yUpper = []
    j = 0
    for i in triangles:
        a = pos[i[0]]
        b = pos[i[1]]
        c = pos[i[2]]
        (a, b, c) = sorted((a, b, c))
        
        #Set x values with spacing of 0.01
        xMax = c[0]
        xMin = a[0]
        xVals = np.linspace(xMin, xMax, int((xMax-xMin)*100))
        line = np.linspace(0, xMax-xMin, int((xMax-xMin)*100))
        x.append(xVals)
        
        #Set y values
        if a[0] == b[0]: #Left edge is vertical
            uSlope = (c[1]-b[1])/(c[0]-b[0])
            lSlope = (c[1]-a[1])/(c[0]-a[0])
            y1 = a[1]+lSlope*line[:int(100*(c[0]-a[0]))]
            y2 = b[1]+uSlope*line[:int(100*(c[0]-b[0]))]
        elif b[0] == c[0]: #Right edge is vertical
            lSlope = (b[1]-a[1])/(b[0]-a[0])
            uSlope = (c[1]-a[1])/(c[0]-a[0])
            y1 = a[1]+lSlope*line[:int(100*(b[0]-a[0]))]
            y2 = a[1]+uSlope*line[:int(100*(c[0]-a[0]))]
        else: #no vertical edges, so we have to split either the upper or lower values into two line segments
            acSlope = (c[1]-a[1])/(c[0]-a[0])
            abSlope = (b[1]-a[1])/(b[0]-a[0])
            bcSlope = (c[1]-b[1])/(c[0]-b[0])
            if abSlope > acSlope: #the upper values are split
                y1 = a[1]+acSlope*line[:int(100*(c[0]-a[0]))]
                y21 = a[1]+abSlope*line[:int(100*(b[0]-a[0]))]
                y22 = b[1]+bcSlope*line[:(int(100*(c[0]-a[0]))-int(100*(b[0]-a[0])))]
                y2 = np.concatenate((y21, y22))
            else: #the lower values are split
                y11 = a[1]+abSlope*line[:int(100*(b[0]-a[0]))]
                y12 = b[1]+bcSlope*line[:(int(100*(c[0]-a[0]))-int(100*(b[0]-a[0])))]
                y1 = np.concatenate((y11, y12)) 
                y2 = a[1]+acSlope*line[:int(100*(c[0]-a[0]))]
        yLower.append(y1)
        yUpper.append(y2)
    return x, yLower, yUpper
    
class simplicialComplex:
    def __init__(self, nodes, edges, triangles, pos):
        self.nodes = nodes
        self.pos = pos
        
        #Sorts edges and triangles with ascending labels
        self.edges = sortSimplicies(edges)
        self.triangles = sortSimplicies(triangles)
        
        #Create graph
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(1, nodes))
        self.G.add_edges_from(edges)
        
        #Create boundary matricies
        self.B1 = calcB1(self.nodes, self.edges)
        self.B2 = calcB2(self.edges, self.triangles)
        
        #Calculate Hodge Laplacians
        self.L0 = np.dot(self.B1, self.B1.T)
        self.L1 = np.dot(self.B1.T, self.B1) + np.dot(self.B2, self.B2.T)
    
    def plotTriangles(self, tri_color='blue', alpha=.2, ax=None):
        '''Plots the triangles of the simplicial complex'''
        if ax is None:
            ax = plt.gca()
        
        x, y1, y2 = triangleLocation(self.triangles, self.pos)
        for i in range(len(x)):
            ax.fill_between(x[i], y1[i], y2[i], facecolor = tri_color, alpha = alpha)
            
    def plotGraph(self, ax=None, **kwargs):
        '''Plots the nodes and edges of the simplicial complex.  Keyword arguments are passed to nx.draw'''
        if ax is None:
            ax = plt.gca()
        
        kwargs.setdefault('width', 4)
        kwargs.setdefault('node_color', 'red')
        kwargs.setdefault('with_labels', True)
        nx.draw(self.G, self.pos, ax=ax, **kwargs)
        #Want to see if some keywords can have defaults such as width=4, node_color='red' and with_labels=True
        
    def plotSimplicialComplex(self, tri_color='blue', alpha=.2, ax = None, **kwargs):
        '''Plots the simplicial complex.  Keyword arguments are passed to nx.draw'''
        self.plotGraph(ax, **kwargs)
        self.plotTriangles(tri_color, alpha, ax)
        
    def balancedHodge(self, alpha=.5):
        '''Returns the 1-balanced Hodge Laplacian for the simplicial complex
        for the chosen alpha value.'''
        return alpha*np.dot(self.B1.T, self.B1) + (1-alpha)*np.dot(self.B2, self.B2.T)

def presetSC(index):
    '''Loads in a premade simplicial complex.  Inputting an invalid index
    will return an empty simplicial complex and print a description of the
    options.'''
    
    if index == 1:
        #simple example
        nodes  = 7
        edges = [(1, 2), (1, 3), (2, 3), (2, 4), (2, 6), (3, 4), (4, 5), (4, 7), (5, 6), (5, 7) ]
        triangles = [(1, 2, 3), (2, 3, 4)]
        pos = {1: (0, 2),
               2: (1, 2),
               3: (.5, 1),
               4: (1.5, 1), 
               5: (2.5, 1), 
               6: (2.5, 2), 
               7: (2, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 2:
        #Hexagon with all triangles
        nodes = 7
        edges = [(1, 2), (1, 3),(1, 4), (2, 4), (2, 5),  (3, 4), (3, 6),(4, 5), (4, 6),(4, 7), (5, 7), (6, 7) ]
        triangles = [(1, 2, 4), (1, 3, 4), (2, 4, 5), (3, 4, 6), (4, 5, 7), (4, 6, 7)]
        pos = {1: (.5, 2),
               2: (1.5, 2),
               3: (0, 1),
               4: (1, 1), 
               5: (2, 1), 
               6: (0.5, 0), 
               7: (1.5, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 3:
        #Hexagon with only 4 triangles
        nodes = 7
        edges = [(1, 2), (1, 3),(1, 4), (2, 4), (2, 5),  (3, 4), (3, 6),(4, 5), (4, 6),(4, 7), (5, 7), (6, 7) ]
        triangles = [(1, 3, 4), (2, 4, 5), (3, 4, 6), (4, 5, 7)]
        pos = {1: (.5, 2),
               2: (1.5, 2),
               3: (0, 1),
               4: (1, 1), 
               5: (2, 1), 
               6: (0.5, 0), 
               7: (1.5, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 4:
        #Hexagon with 3 triangles
        nodes = 7
        edges = [(1, 2), (1, 3),(1, 4), (2, 4), (2, 5),  (3, 4), (3, 6),(4, 5), (4, 6),(4, 7), (5, 7), (6, 7) ]
        triangles = [(2, 4, 5), (3, 4, 6), (4, 5, 7)]
        pos = {1: (.5, 2),
               2: (1.5, 2),
               3: (0, 1),
               4: (1, 1), 
               5: (2, 1), 
               6: (0.5, 0), 
               7: (1.5, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 5:
        #Graph with long middle
        nodes = 12
        edges = [(1, 2), (1, 3), (2, 3),  (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12) ]
        triangles = [(1, 2, 3)]
        pos = {1: (0, 2),
               2: (0, 0),
               3: (1, 1),
               4: (2.5, 1.5), 
               5: (3.5, .5), 
               6: (4.5, 1.5), 
               7: (6, 1),
               8: (5.5, 2),
               9: (6.5, 2),
               10: (7, 1),
               11: (6.5, 0),
               12: (5.5, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 6:
        #3/4 triangles of a subdivided triangle
        nodes = 6
        edges = [(1, 2), (1, 3), (2, 3),  (2, 4), (2, 5), (3, 5), (3, 6), (4, 5), (5, 6)  ]
        triangles = [(2, 3, 5), (2, 4, 5), (3, 5, 6)]
        pos = {1: (1, 2),
               2: (.5, 1),
               3: (1.5, 1),
               4: (0, 0), 
               5: (1, 0), 
               6: (2, 0), }
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 7:
        #4/4 triangles of a subdivided triangle
        nodes = 6
        edges = [(1, 2), (1, 3), (2, 3),  (2, 4), (2, 5), (3, 5), (3, 6), (4, 5), (5, 6)  ]
        triangles = [(1, 2, 3),(2, 3, 5), (2, 4, 5), (3, 5, 6)]
        pos = {1: (1, 2),
               2: (.5, 1),
               3: (1.5, 1),
               4: (0, 0), 
               5: (1, 0), 
               6: (2, 0), }
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 8:
        #square within square with 2 triangles
        nodes = 8
        edges = [(1, 2), (1, 3), (2, 3),  (2, 4), (2, 5), (3, 6), (3, 7), (4, 5), (5, 7), (5, 8), (6, 7), (7, 8) ]
        triangles = [(1, 2, 3), (2, 4, 5)]
        pos = {1: (0, 2),
               2: (1, 2),
               3: (0, 1),
               4: (2, 2), 
               5: (2, 1), 
               6: (0, 0),
               7: (1, 0), 
               8: (2, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 9:
        #square within square, 4 triangles
        nodes = 8
        edges = [(1, 2), (1, 3), (2, 3),  (2, 4), (2, 5), (3, 6), (3, 7), (4, 5), (5, 7), (5, 8), (6, 7), (7, 8) ]
        triangles = [(1, 2, 3), (2, 4, 5), (3, 6, 7), (5, 7, 8)]
        pos = {1: (0, 2),
               2: (1, 2),
               3: (0, 1),
               4: (2, 2), 
               5: (2, 1), 
               6: (0, 0),
               7: (1, 0), 
               8: (2, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 10:
        #simple example 2
        nodes  = 7
        edges = [(1, 2), (1, 3), (2, 3), (2, 4), (2, 6), (3, 4), (4, 5), (4, 7), (5, 6), (5, 7) ]
        triangles = [(1, 2, 3), (4, 5, 7)]
        pos = {1: (0, 2),
               2: (1, 2),
               3: (.5, 1),
               4: (1.5, 1), 
               5: (2.5, 1), 
               6: (2.5, 2), 
               7: (2, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 11:
        #Larger Hexagon
        nodes = 19
        edges = [(1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (2, 6), (3, 6),
        (3, 7), (4, 5), (4, 8), (4, 9), (5, 6), (5, 9), (5, 10), 
        (6, 7), (6, 10), (6, 11), (7, 11), (7, 12), (8, 9), (8, 13), 
        (9, 10), (9, 13),(9, 14), (10, 11), (10, 14), (10, 15), 
        (11, 12), (11, 15), (11, 16), (12, 16), (13, 14), (13, 17), 
        (14, 15), (14, 17), (14, 18), (15, 16), (15, 18), (15, 19),
        (16, 19), (17, 18), (18, 19)]
        pos = {1: (1, 4),
        2: (2, 4),
        3: (3, 4),
        4: (.5, 3), 
        5: (1.5, 3), 
        6: (2.5, 3), 
        7: (3.5, 3),
        8: (0, 2),
        9: (1, 2),
        10: (2, 2),
        11: (3, 2), 
        12: (4, 2),
        13: (.5, 1), 
        14: (1.5, 1),
        15: (2.5, 1), 
        16: (3.5, 1),
        17: (1, 0),
        18: (2, 0), 
        19: (3, 0)}
        triangles = [[1, 2, 5], [1, 4, 5], [2, 3, 6], [2, 5, 6], [3, 6, 7],
                [4, 5, 9], [4, 8, 9], [5, 6, 10], [5, 9, 10],
                [6, 7, 11], [6, 10, 11], [7, 11, 12], [8, 9, 13],
                [9, 10, 14], [9, 13, 14], [10, 11, 15], [10, 14, 15],
                [11, 12, 16], [11, 15, 16], [13, 14, 17], 
                [14, 15, 18], [14, 17, 18], [15, 16, 19], [15, 18, 19]]
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 12:
        #Another example
        nodes  = 8
        edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (2, 6), (3, 4),
                 (3, 7), (4, 5), (4, 8), (5, 6), (5, 8), (7, 8)]
        triangles = [(1, 2, 5), (1, 4, 5), (4, 5, 8)]
        pos = {1: (1.5, 2),
               2: (2.5, 2),
               3: (0, 1),
               4: (1, 1), 
               5: (2, 1), 
               6: (3, 1), 
               7: (0,0),
               8: (1.5, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif index == 13:
        #Example simplicial complex from our paper
        nodes  = 6
        edges = [(1, 2), (1, 3), (1, 4), (2, 3),
                 (2, 5), (3, 4), (3,6), (4, 6), (5, 6)]
        triangles = [(1, 3, 4), (3, 4, 6)]
        pos = {1: (1, 2),
               2: (0, 1),
               3: (1, 1), 
               4: (2, 1), 
               5: (0,0),
               6: (1, 0)}
        return simplicialComplex(nodes, edges, triangles, pos)
    
    elif 100<index<111:
        #Larger Hexagon with varying triangles
        nodes = 19
        edges = [(1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (2, 6), (3, 6),
        (3, 7), (4, 5), (4, 8), (4, 9), (5, 6), (5, 9), (5, 10), 
        (6, 7), (6, 10), (6, 11), (7, 11), (7, 12), (8, 9), (8, 13), 
        (9, 10), (9, 13),(9, 14), (10, 11), (10, 14), (10, 15), 
        (11, 12), (11, 15), (11, 16), (12, 16), (13, 14), (13, 17), 
        (14, 15), (14, 17), (14, 18), (15, 16), (15, 18), (15, 19),
        (16, 19), (17, 18), (18, 19)]
        pos = {1: (1, 4),
        2: (2, 4),
        3: (3, 4),
        4: (.5, 3), 
        5: (1.5, 3), 
        6: (2.5, 3), 
        7: (3.5, 3),
        8: (0, 2),
        9: (1, 2),
        10: (2, 2),
        11: (3, 2), 
        12: (4, 2),
        13: (.5, 1), 
        14: (1.5, 1),
        15: (2.5, 1), 
        16: (3.5, 1),
        17: (1, 0),
        18: (2, 0), 
        19: (3, 0)}
        
        if index == 100:
            triangles = [[1, 2, 5], [1, 4, 5], [2, 3, 6], [2, 5, 6], [3, 6, 7],
                [4, 5, 9], [4, 8, 9], [5, 6, 10], [5, 9, 10],
                [6, 7, 11], [6, 10, 11], [7, 11, 12], [8, 9, 13],
                [9, 10, 14], [9, 13, 14], [10, 11, 15], [10, 14, 15],
                [11, 12, 16], [11, 15, 16], [13, 14, 17], 
                [14, 15, 18], [14, 17, 18], [15, 16, 19], [15, 18, 19]]
        elif index == 101:
            triangles = [[1, 2, 5], 
                        [1, 4, 5],
                        [2, 5, 6], 
                        [4, 5, 9],
                        [5, 9, 10], 
                        [5, 6, 10]]
        elif index == 102:
            triangles = [[1, 4, 5], 
                        [4, 5, 9],
                        [5, 9, 10], 
                        [9, 10, 14],
                        [10, 14, 15], 
                        [14, 15, 18]]
        elif index == 103:
            triangles = [[4, 5, 9], 
                        [5, 9, 10],
                        [11, 12, 16], 
                        [11, 15, 16],
                        [13, 14, 17], 
                        [14, 17, 18]]

        elif index == 104:
            #checkerboard pattern
            triangles = [[1, 2, 5],
                         [2, 3, 6],
                         [4, 5, 9],
                         [5, 6, 10],
                         [6, 7, 11],
                         [8, 9, 13],
                         [9, 10, 14],
                         [10, 11, 15],
                         [11, 12, 16],
                         [13, 14, 17],
                         [14, 15, 18],
                         [15, 16, 19]]
        elif index == 105:
            #Small ring with middle triangle missing
            triangles = [[4, 5, 9],
                         [4, 8, 9],
                         [5, 6, 10],
                         [5, 9, 10],
                         [6, 10, 11],
                         [8, 9, 13],
                         [9, 13, 14],
                         [10, 11, 15],
                         [10, 14, 15],
                         [13, 14, 17],
                         [14, 15, 18],
                         [14, 17, 18]]
        elif index == 106:
            triangles = [[1,4,5], 
                         [4, 5, 9],
                         [4, 8, 9],
                         [5, 9, 10],
                         [8, 9, 13],
                         [9, 10, 14],
                         [9, 13, 14],
                         [10, 14, 15],
                         [13, 14, 17],
                         [14, 15, 18],
                         [14, 17, 18],
                         [15, 18, 19]]
        elif index == 107:
            triangles =[[1, 2, 5],
                         [1, 4, 5],
                         [2, 3, 6],
                         [2, 5, 6],
                         [3, 6, 7],
                         [4, 5, 9],
                         [4, 8, 9],
                         [6, 7, 11],
                         [7, 11, 12],
                         [8, 9, 13],
                         [9, 13, 14],
                         [11, 12, 16],
                         [11, 15, 16],
                         [13, 14, 17],
                         [14, 15, 18],
                         [14, 17, 18],
                         [15, 16, 19],
                         [15, 18, 19]]
        elif index == 108:
            triangles = [[1, 4, 5],
                        [4, 5, 9],
                        [4, 8, 9],
                        [5, 9, 10],
                        [5, 6, 10],
                        [6, 7, 11], 
                        [8, 9, 13], 
                        [9, 13, 14], 
                        [9, 10, 14],
                        [10, 14, 15], 
                        [10, 11, 15],
                        [13, 14, 17], 
                        [14, 17, 18], 
                        [14, 15, 18], 
                        [15, 18 ,19],
                        [6, 10, 11],
                        [1, 2, 5], 
                        [15, 16, 19]]
        elif index == 109:
            triangles =[[1, 2, 5], 
                        [1, 4, 5],
                        [2, 5, 6],
                        [6, 10, 11],
                        [3, 6, 7],
                        [6, 7, 11],
                        [7, 11, 12],
                        [11, 12, 16],
                        [11, 15, 16],
                        [15, 16, 19],
                        [15, 18, 19],
                        [14, 15, 18],
                        [9, 10, 14],
                        [13, 14, 17],
                        [9, 13, 14],
                        [8, 9, 13],
                        [4, 8, 9],
                        [4, 5, 9]]

        elif index == 110:
            triangles = [[1, 2, 5],
                        [1, 4, 5],
                        [2, 3, 6],
                        [3, 6, 7],
                        [4, 8, 9],
                        [5, 9, 10], 
                        [5, 6, 10], 
                        [6, 10, 11], 
                        [7, 11, 12],
                        [8, 9, 13], 
                        [9, 10, 14],
                        [10, 14, 15], 
                        [10, 11, 15], 
                        [11, 12, 16], 
                        [13, 14 ,17],
                        [14, 17, 18],
                        [15, 18, 19], 
                        [15, 16, 19]]

        return simplicialComplex(nodes, edges, triangles, pos)
    
    else: #Invalid index gives a list of valid indices
        print("Invalid index for presetGraph \n" + "Index 1: Simple example \n" +
                "Index 2: Hexagon with all triangles \n"
             +"Index 3: Hexagon with only 4 triangles\n"
             +"Index 4: Hexagon with 3 triangles \n"
             +"Index 5: Simplicial complex with long middle \n"
             +"Index 6: 3/4 triangles of a subdivided triangle \n"
             +"Index 7: 4/4 triangles of a subdivided triangle \n"
             +"Index 8: square within square with 2 triangles \n"
             +"Index 9: square within square, 4 triangles \n"
             +"Index 10: Simple example 2 \n"
             +"Index 11: Larger Hexagon \n"
             +"Index 12: Simple example 3\n"
             +"Index 13: Example simplicial complex from our paper\n"
             +"Index 101-110: Simplicial complexes with the same underlying 2-skeleton")
        nodes = 0
        edges = []
        triangles = []
        pos = {}
        return simplicialComplex(nodes, edges, triangles, pos)
        
def balancedHodge(B1, B2, alpha=.5): 
    '''1st balanced Hodge Laplacians of the simplex given by boundary
    matrices B1 and B2'''
    L1 = alpha*np.dot(B1.T, B1) + (1-alpha)*np.dot(B2, B2.T)
    return L1

def hodgeDecomp(c, B1, B2, alpha=.5):
    #Takes in an edge flow vector c as well as the matrices B1 and B2 and
    # outputs the vector decomposed into it's projections in gradient
    # flow, curl flow, and harmonic flow. 
    F = alpha*B1.T
    G = (1-alpha)*B2
            
    g = np.dot(F,linalg.lsqr(F, c)[0])
    r = np.dot(G,linalg.lsqr(G, c)[0])
    h = c - g - r
    
    return g,r,h
    
def BHL1(sc, alpha=.5, deltat=.01, T=2000, seed=1):
    '''Takes in a simplicial complex and calculates the Hodge Laplacian weighted by alpha (lower weight). 
    Starts with a random initial condition with edge values between 0 and 1, summing to 1.
    Then runs a forward Euler simulation where dx/dt is -Lx (the BHL-1 dynamics).  deltat is the size 
    of the time steps and T is the number of time steps.  Returns edge flows, gradient, curl, and 
    harmonic components. The seed for the random initial conditions can be set, but the initial 
    conditions will be the same without manually changing this seed. Outputs edge flow, gradient, 
    curl, and harmonic components for each time step.'''
    
    L = sc.balancedHodge(alpha)

    #Number of edges n:
    (n, m) = np.shape(L)

    #Random Initial Condition
    np.random.seed(seed)
    x = np.random.uniform(0, 1, n)
    x = x/np.sum(x)

    #Initialize hodge decomposition
    g = np.zeros((T, n))
    c = np.zeros((T, n))
    h = np.zeros((T, n))

    #edge flow e saved along time steps
    e = np.zeros((T+1, n))
    e[0, :] = x

    #Actual calculation (forward Euler simulation)
    for t in range(T):
        e[t+1, :] = e[t, :] -deltat*np.dot(L, e[t, :])  #recursive forward euler calculation

        #calculates hodge decomposition at each time step
        (g[t, :], c[t, :], h[t, :]) = hodgeDecomp(e[t+1, :], sc.B1, sc.B2, alpha=alpha);
    
    return e, g, c, h

def subplotSC(sc, weights, ax=None, xlabel='', ylabel = '', color='k', tri_color='k', **kwargs):
    '''Plots the simplicial complex with the weights labeling the edges.'''
    kwargs.setdefault('node_color', color)
    kwargs.setdefault('edge_color', color)
    
    if ax is None:
        ax = plt.gca()
    
    weights = np.around(weights, 3) #rounds data to fit in image
    
    edges = list()
    for i in sc.edges:
        edges.append(tuple(i))
    elabels = dict(zip(edges, weights)) #label edges with weights
    
    #Plot the thing
    sc.plotSimplicialComplex(tri_color=tri_color, ax=ax, **kwargs)
    nx.draw_networkx_edge_labels(sc.G, sc.pos, ax=ax, edge_labels = elabels) #Includes edge labels

    #labels plot, but removes axes
    ax.axis('on') 
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
def consensusOnSCPlot(sc, alpha=.5, times=(0, 50, 500), T=2000):
    '''Runs consensus simulation using BHL-1 and the given balancing parameter alpha.
    Then plots the Hodge decomposition projections onto the simplicial complex for the
    given time increments. T is the total number of time steps.'''
    #Runs consensus algorithm.  
    e, g, c, h = BHL1(sc,alpha=alpha, T=T)

    fig, axes = plt.subplots(len(times), 4, figsize=(12, len(times)*3))

    #intensity of edges is based on the weights from the Hodge decomposition
    weights = np.zeros((len(times), 4, 9))
    weights[:, 0, :] = e[times, :]
    weights[:, 1, :] = g[times, :]
    weights[:, 2, :] = c[times, :]
    weights[:, 3, :] = h[times, :]

    node_colors = ['k', 'tomato', 'fuchsia', 'springgreen']
    edge_colors = ['k', 'tomato', 'violet', 'palegreen']
    tri_colors = ['k', 'coral', 'violet', 'palegreen']
    xlabels=['total', 'gradient', 'curl', 'harmonic']

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            subplotSC(sc,weights[i, j,:], ax=ax, node_color=node_colors[j], 
                      edge_color=edge_colors[j], tri_color=tri_colors[j])
            if j==0:
                ax.set_ylabel(f'time = {times[i]}')
            if i==len(times)-1:
                ax.set_xlabel(xlabels[j])
    plt.tight_layout()
    
def timeSeriesPlot(ax,e, g, c, h,alpha, xrange=np.arange(1,1000), fsize=12):
    '''Plots the BHL-1 consensus under the Hodge decomposition.'''
    ax[0].plot(e[xrange])
    ax[1].plot(g[xrange])
    ax[2].plot(c[xrange])
    ax[3].plot(h[xrange])
    
def timeSeriesFigure(sc, alphas = .5, xrange=np.arange(1,500)):
    '''Creates a timeSeriesPlot for each value of alpha, and puts them
    nicely into subplots.'''
    fig, axes = plt.subplots(len(alphas),4,figsize=(12, 2.5*len(alphas)), sharex=True, sharey=True)
    titles=['total', 'gradient', 'curl', 'harmonic']

    #Plotting the time series
    for i, a in enumerate(alphas):
        e, g, c, h = BHL1(sc, alpha=a)
        timeSeriesPlot(axes[i],e, g, c, h,a, xrange)
        axes[i,0].set_ylabel(f"$\\alpha = {alphas[i]}$")

    #Labels
    for j in range(4):
        axes[0, j].set_title(titles[j])
        axes[-1, j].set_xlabel(r'time, $t$')

    plt.tight_layout()
    
def subspaceConvergencePlot(sc, N=20):
    '''Plots the convergence rates for the gradient and curl subspaces in a BHL-1 simulation for N 
    equally spaced balancing parameters from 0 to 1.  Also show the theoretical convergence rates 
    based on the the eigenvectors of the compontents of the Balanced Hodge Laplacian.  sc is the 
    relevant simplicial complex.'''

    #Values of α that we test
    alphaVals = np.arange(1, N+1)/(N+1)

    #Size of time steps and number of time steps
    deltat = .01

    #Slopes of the log error for total edge flow, gradient, and curl
    eErrorSlope = np.zeros(N)
    gErrorSlope = np.zeros(N)
    cErrorSlope = np.zeros(N)

    #These vectors record the values of -α λ21 and -(1-α)λ22, respectively
    lowerEig = np.zeros(N)
    upperEig = np.zeros(N)

    for i in range(N):
        alpha = alphaVals[i]

        #Finds Laplacian and runs simulation
        L = sc.balancedHodge(alpha)
        e, g, c, h = BHL1(sc, alpha=alpha, deltat = deltat)

        #Gets time steps and number of edges
        (T, n) = np.shape(g)

        #Using the last edge flow as the "true" value, we calculate error and its norm at each time step
        # We also calculate the error in each of the domains: gradient, curl, and harmonic
        error = [e[t,:]-e[-1,:] for t in range(T-1)]
        normErr = [np.linalg.norm(error[t]) for t in range(T-1)]
        gError = [g[t,:]-g[-1,:] for t in range(T-1)]
        gNormErr = [np.linalg.norm(gError[t]) for t in range(T-1)]
        cError = [c[t,:]-c[-1,:] for t in range(T-1)]
        cNormErr = [np.linalg.norm(cError[t]) for t in range(T-1)]

        #We remove any values of 0 error, where the error is too small for python to handle
        #IMPORTANT: If too many are deleted, we need to ensure that the lower calculations don't get the wrong data.
        normErr = np.delete(normErr, np.where(normErr == 0.0))
        gNormErr = np.delete(gNormErr, np.where(gNormErr == 0.0))
        cNormErr = np.delete(cNormErr, np.where(cNormErr == 0.0))

        #We get the new lengths of these vectors, as well as how many elements were deleted
        eLen = len(normErr)
        eDel = T-eLen
        gLen = len(gNormErr)
        gDel = T-gLen
        cLen = len(cNormErr)
        cDel = T-cLen

        #We want to find the slope of the error on a log scale
        #Note that if we generalize this to a function, the parameters of xmin and xmax will need to be changed
        #These were chosen as that range is most linear in the log error plot
        logSlope = [(np.log(normErr[t+1]) - np.log(normErr[t]))/deltat for t in range(eLen-2)]
        gLogSlope= [(np.log(gNormErr[t+1]) - np.log(gNormErr[t]))/deltat for t in range(eLen-2)]
        cLogSlope = [(np.log(cNormErr[t+1]) - np.log(cNormErr[t]))/deltat for t in range(eLen-2)]

        #We then find the average of these slopes in the range between xmin and xmin+xrange 
        # where the slope should be stable
        xmin = 500
        xrange = 1000
        xmax = xmin+xrange

        #We use these to compute the minimum and maximum values of our "stable range."
        #This is necessary to eliminate errors that may occur if many elements are deleted above.
        emin = max(0, xmin-eDel)
        emax = min(emin+xrange, eLen)
        gmin = max(0, xmin-gDel)
        gmax = min(emin+xrange, gLen)
        cmin = max(0, xmin-cDel)
        cmax = min(emin+xrange, cLen)

        eErrorSlope[i] = np.mean(logSlope[emin:emax])
        gErrorSlope[i] = np.mean(gLogSlope[gmin:gmax])
        cErrorSlope[i] = np.mean(cLogSlope[cmin:cmax])

        #We compute the relevant eigenvalues.
        lambda21 = (min(np.linalg.eig(np.dot(sc.B1,sc.B1.T))[0][np.linalg.eig(np.dot(sc.B1,sc.B1.T))[0]>1e-14]));
        lowerEig[i]=-lambda21*alpha
        lambda22 = (min(np.linalg.eig(np.dot(sc.B2,sc.B2.T))[0][np.linalg.eig(np.dot(sc.B2,sc.B2.T))[0]>1e-14]));
        upperEig[i]=-lambda22*(1-alpha)
        
        #Makes the plot 

    plt.plot(alphaVals, -eErrorSlope, ".",  color='k', zorder=2, ms=8)
    plt.plot(alphaVals, -gErrorSlope, "^", color = 'tomato', fillstyle="none",zorder=1, ms=9)
    plt.plot(alphaVals, -cErrorSlope, "s", color = 'fuchsia', fillstyle="none",zorder=1, ms=8)
    plt.plot(alphaVals, -lowerEig, color = 'tomato', zorder=1, linestyle='-')
    plt.plot(alphaVals, -upperEig, color = 'fuchsia', zorder=1, linestyle='--')
    plt.title('Subspace Convergence Rates')
    plt.xlabel(r'balancing parameter, $\alpha$')
    plt.ylabel(r'convergence rate, $\lambda_2$')
    plt.legend(('total', 'gradient subspace', 'curl subspace', '$\\alpha \\lambda_2^1$', '$(1-\\alpha)\\lambda_2^2$'))

    # alphas = [.3, .54, .8]
    # for a_star in alphas:
    #     plt.plot([a_star]*2,[-.2,2],'k:',alpha=.4)

    plt.ylim(0)    
    plt.xlim(0, 1)
    plt.tight_layout()
    
def convergenceEigenvalues(sc):
    '''Returns the two eigenvalues used to calculate the convergence rates of the subspaces with BHL-1 consensus.'''
    lambda21 = (min(np.linalg.eig(np.dot(sc.B1,sc.B1.T))[0][np.linalg.eig(np.dot(sc.B1,sc.B1.T))[0]>1e-14]));
    lambda22 = (min(np.linalg.eig(np.dot(sc.B2,sc.B2.T))[0][np.linalg.eig(np.dot(sc.B2,sc.B2.T))[0]>1e-14]));
    return lambda21.real, lambda22.real
    
def optimalAlpha(sc):
    '''Returns the balancing parameter alpha that optimizes convergence.'''
    lambda21 = (min(np.linalg.eig(np.dot(sc.B1,sc.B1.T))[0][np.linalg.eig(np.dot(sc.B1,sc.B1.T))[0]>1e-14]));
    lambda22 = (min(np.linalg.eig(np.dot(sc.B2,sc.B2.T))[0][np.linalg.eig(np.dot(sc.B2,sc.B2.T))[0]>1e-14]));
    return (lambda22/(lambda21+lambda22)).real
    
def optimalRate(sc):
    '''Returns the convergence rate with BHL-1 consensus when the balancing parameter is optimized.'''
    l1, l2 = convergenceEigenvalues(sc)
    a = optimalAlpha(sc)
    return l1*a
    
def subspaceConvergencePlotWithExamples(sc, N=20, alphas=[]):
    '''Creates the same plot as subspaceConvergencePlot, but also includes subplots of the
    log error during the simulation for different values of alpha.  These alpha values
    can be specified, or will default to 0.2, 0.8, and the value for optimal convergence.'''

    if alphas == []:
        alphas = [.2, np.around(optimalAlpha(sc), 2), .8]
    a = len(alphas)

    fsize=12
    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot2grid((3, a), (0, 0), colspan=a, rowspan=2)

    subspaceConvergencePlot(sc, N)
    #Plots a dotted line in the subspaceConvergencePlot for each specified value of alpha
    for a_star in alphas:
        plt.plot([a_star]*2,[-.2,2],'k:',alpha=.4)

    ##############################################
    deltat = .01
    axes = np.zeros(a, dtype='object')

    for i, alpha in enumerate(alphas):
        #Runs simulation
        e, g, c, h = BHL1(sc, alpha=alpha, deltat = deltat)

        #Gets time steps and number of edges
        (T, n) = np.shape(g)

        #Using the last edge flow as the "true" value, we calculate error and its norm at each time step
        # We also calculate the error in each of the domains: gradient, curl, and harmonic
        error = np.zeros((T, n))
        gError = np.zeros((T, n))
        cError = np.zeros((T, n))

        normErr = np.zeros(T)
        gNormErr = np.zeros(T)
        cNormErr = np.zeros(T)
        #This could probably be vectorized.  This fills in the lists with the error at each time step (difference from final step)
        # This also fills in the normed error lists for each time step
        error = [e[t,:]-e[-1,:] for t in range(T-1)]
        normErr = [np.linalg.norm(error[t]) for t in range(T-1)]
        gError = [g[t,:]-g[-1,:] for t in range(T-1)]
        gNormErr = [np.linalg.norm(gError[t]) for t in range(T-1)]
        cError = [c[t,:]-c[-1,:] for t in range(T-1)]
        cNormErr = [np.linalg.norm(cError[t]) for t in range(T-1)]

        #We remove any values of 0 error, where the error is too small for python to handle
        normErr = np.delete(normErr, np.where(normErr == 0.0))
        gNormErr = np.delete(gNormErr, np.where(gNormErr == 0.0))
        cNormErr = np.delete(cNormErr, np.where(cNormErr == 0.0))

        if i==0:
            axes[i] = plt.subplot2grid((3,a),(2,i))
        else:
            axes[i] = plt.subplot2grid((3,a),(2,i), sharey=axes[0])
            axes[i].set_yticks([])
        iterator = 70
        plt.plot(deltat*np.arange(0,T-1)[::iterator], np.log(normErr)[::iterator],  " .",   color='k', zorder=3, ms=5)
        plt.plot(deltat*np.arange(0,T-1)[::iterator],np.log(gNormErr)[::iterator], "^", color = 'tomato', fillstyle="none",zorder=2, ms=7)
        plt.plot(deltat*np.arange(0,T-1)[::iterator],np.log(cNormErr)[::iterator], "s", color = 'fuchsia', fillstyle="none",zorder=1, ms=5)
        plt.ylim(-16,0) #Don't show past machine precision

        plt.xlabel(r'time, $t$', fontsize = fsize)
        if i==0:
            plt.ylabel(r'log error', fontsize = fsize)
        plt.xlim((0, T*deltat/2))
        plt.ylim((-12, 0))

        axes[i].text(.6, .8, r"$\alpha= " + str(alphas[i]) + '$', fontsize = fsize+9-2*a, horizontalalignment='center', transform=axes[i].transAxes)
    ##############################################
    plt.tight_layout()
    plt.show()
    
def similarSCPlot(scs, colors=['green', 'blue', 'red', 'orange', 'purple']):
    '''Plots the given simplicial complexes in different colors, as well as showing the theoretical convergence
    rates of their gradient and curl subspaces.  This only works when all the simplicial complexes share the
    same underlying 2-skeleton (the same nodes and edges), but have different triangles.'''
    
    if len(scs) > len(colors):
        colors = colors*(math.ceil(len(scs)/len(colors)))

    plt.figure(figsize=(3*(len(scs)+1), 3))

    for i in range(len(scs)):
        plt.subplot(1, len(scs)+1, i+1)
        scs[i].plotSimplicialComplex(node_color=colors[i], edge_color=colors[i], tri_color=colors[i], 
                                     with_labels=False, node_size=80, arrows=False)
        plt.title(f'SC {i+1}', fontsize=15)

    lambda21 = [convergenceEigenvalues(sc)[0] for sc in scs]
    lambda22 = [convergenceEigenvalues(sc)[1] for sc in scs]

    plt.subplot(1, len(scs)+1, len(scs)+1)
    plt.plot([0, 1], [0, lambda21[0]], color = 'black')
    for i in range(len(scs)):
        plt.plot([0, 1], [lambda22[i], 0], color = colors[i], linestyle='--')
    for i, sc in enumerate(scs):
        plt.plot(optimalAlpha(sc), optimalAlpha(sc)*lambda21[i], c=colors[i], marker='o')
    plt.title(r'Subspace convergence', fontsize=15)
    plt.xlabel(r'balancing parameter, $\alpha$', fontsize=12)
    plt.ylabel(r'convergence rate, $\lambda_2$', fontsize=12)
    plt.ylim(0, 2)
    plt.legend(('gradient subspace', 'curl subspace, SC 1', 'curl subspace, SC 2', 'curl subspace, SC 3'), fontsize=10)
    plt.tight_layout()
    
def optAlphaPlot(scs, colors = ['black']):
    '''Plots the optimal balancing parameter (alpha) and the optimal convergence rate for the input for the
    inputted simplicial complexes.  Those simplicial complexes are also drawn on the same plot.  The 
    drawings can be colored if desired.'''
    N = len(scs)
    
    #font size
    fsize=12

    if len(scs) > len(colors):
        colors = colors*(math.ceil(len(scs)/len(colors)))
    
    #Create temporary images of all simplicial complexes.
    images = np.zeros(N, dtype='object')

    size = 4
    for i in range(N):
        fig = plt.figure(figsize=(size, size))
        scs[i].plotSimplicialComplex(node_color=colors[i], edge_color=colors[i], tri_color=colors[i], 
                                         with_labels=False, node_size=20, arrows=False, width=6)
        images[i] = NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(images[i].name)
        plt.close(fig)

    #Plot of optimal alpha values and convergence rates for these different simplicial complexes
    optAlpha = np.zeros(N)
    optConvergence = np.zeros(N)

    #Calculate data
    for i, sc in enumerate(scs):
        if not sc.triangles:
            optAlpha[i] = -.1
            optConvergence[i] = -.1
        else:
            optAlpha[i] = optimalAlpha(sc)
            optConvergence[i] = optimalRate(sc)

    #Sort by optimal alpha
    ind = np.argsort(optAlpha)
    optAlpha = optAlpha[ind]
    optConvergence = optConvergence[ind]  

    #Plots the optimal alpha and convergence rate
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(optAlpha, '.', ms=40)
    plt.plot(optConvergence, 's', ms=20)

    for i in range(N):
        #Displays images of the graphs on the plot
        arr_img = mpimg.imread(images[ind[i]].name, format='png')

        imagebox = OffsetImage(arr_img, zoom=0.3)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (i,0.1),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(.5, 1),
                            bboxprops={"edgecolor" : "none"})
        ax.add_artist(ab)

    plt.legend(['optimal balancing parameter, $\\alpha\'$', 'optimal convergence rate'], fontsize=2*(fsize-2))
    plt.rc('ytick', labelsize= 2*fsize)
    plt.ylim(-.1)
    ax.set_xticklabels([])
    plt.grid()
    ax.spines['bottom'].set_visible(False)
    plt.title("Impact of 2-simplex arrangement on BHL-1 consensus", fontsize=2*fsize)
    plt.tight_layout()

    #Delete temporary images
    for img in images:
        img.close()
        os.remove(img.name)
        
