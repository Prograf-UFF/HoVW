import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class SpacePoint():
    def __init__(self, coordinates=None, index=None, label=None):
        if coordinates == None:
            self.coordinates = self._np_coordinates((-1,-1,-1))
        else:
            self.coordinates = self._np_coordinates(coordinates)
        
        if index == None:
            self.index = -1
        else:
            self.index = index

        if label == None:
            self.label = -1
        else:
            self.label = label

    def _np_coordinates(self,c):
        return np.array([c[0],c[1],c[2]], dtype=np.float_)

    def set_coordinates(self, c):
        self.coordinates = c

    def set_label(self, l):
        self.label = l

    def pprint(self):
        print("Point {} ; Label {}: \n\t- x: {}\n\t- y: {}\n\t- z: {}"
        .format(self.index, self.label, self.coordinates[0], self.coordinates[1], self.coordinates[2]))


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines
    
def d2c(D):
    visited = []
    all_points = []
    #points are tuple of: ndarray for position and distance's matrix index
    p1 = SpacePoint((0,0,0), 0)
    visited.append(0)

    p1maxdist = np.amax(D[0])
    p2index = np.argwhere(D == p1maxdist)[0][1]

    p2 = SpacePoint((p1maxdist,0,0), p2index)
    visited.append(p2index)

    all_points.append(p1)
    all_points.append(p2)

    for i in range(D[0].shape[0]):
        if i not in visited:
            p3 = SpacePoint((-1,-1,-1), i)
            visited.append(i)
            break
    
    p3x = D[p1.index][p2.index] ** 2 + D[p1.index][p3.index] ** 2 - D[p2.index][p3.index] ** 2
    p3x = p3x/(2*D[p1.index][p2.index])
    p3y = np.sqrt(D[p1.index][p3.index] ** 2 - p3x ** 2)

    p3.set_coordinates((p3x, p3y, 0))
    
    all_points.append(p3)
    
    for i in range(D[0].shape[0]):
        if i not in visited:
            p4 = SpacePoint((-1,-1,-1), i)
            visited.append(i)
            
            p4x = D[p1.index][p2.index] ** 2 + D[p1.index][p4.index] ** 2 - D[p2.index][p4.index] ** 2
            p4x = p4x/(2*D[p1.index][p2.index])

            p4y = p3.coordinates[0] ** 2 + p3.coordinates[1] ** 2 - D[p1.index][p4.index] ** 2
            p4y = p4y - 2 * p3.coordinates[0] * p4x - D[p3.index][p4.index] ** 2
            p4y = p4y / (2 * p3.coordinates[1])

            p4z = np.sqrt(D[p1.index][p4.index] ** 2 - p4x - p4y)

            p4.set_coordinates((p4x, p4y, p4z))
            
            all_points.append(p4)

    return all_points

if __name__ == '__main__':
    colors183 = ["#ff0000", "#ffc480", "#00b330", "#266399", "#621d73", "#f20000", "#8c6c46", "#004011", "#80c4ff", "#2e1a33", "#660000", "#59442d", "#264d30", "#001433", "#c299cc", "#330000", "#a6927c", "#608068", "#295ba6", "#47004d", "#994d4d", "#594f43", "#008c38", "#bfd9ff", "#d900ca", "#4c2626", "#ffaa00", "#1a3324", "#8698b3", "#ff80f6", "#331a1a", "#7f5500", "#a3d9b8", "#003de6", "#bf60b9", "#cc9999", "#593c00", "#00ff88", "#001b66", "#4d264a", "#b21800", "#bf8f30", "#33cc85", "#3662d9", "#4d394b", "#ff5940", "#a68a53", "#1a6642", "#1a2e66", "#b3008f", "#591f16", "#d9c7a3", "#29a67c", "#6c89d9", "#73005c", "#e58273", "#e5b800", "#73e6bf", "#2d3959", "#401036", "#663a33", "#403610", "#86b3a4", "#565e73", "#ffbff2", "#ffc8bf", "#f2da79", "#30403a", "#393e4d", "#997391", "#997873", "#736739", "#00ffcc", "#23318c", "#ff00aa", "#4d3c39", "#999173", "#bffff2", "#8091ff", "#994d80", "#731f00", "#bfb300", "#00f2e2", "#202440", "#33262f", "#cc5c33", "#8c8300", "#00998f", "#b6bef2", "#d90074", "#8c3f23", "#665f00", "#00403c", "#0000b3", "#660036", "#ffa280", "#a6a053", "#1d736d", "#0000a6", "#e5005c", "#b27159", "#333226", "#00eeff", "#000040", "#8c234d", "#7f5140", "#dae639", "#005359", "#737399", "#401023", "#d9b1a3", "#494d13", "#0d3033", "#262633", "#ff80b3", "#ff6600", "#d5d9a3", "#6cd2d9", "#3a29a6", "#592d3e", "#b24700", "#526600", "#4d9499", "#4100f2", "#d9a3b8", "#662900", "#b6f23d", "#00ccff", "#1b0066", "#735662", "#331400", "#9fbf60", "#23778c", "#170d33", "#ff0044", "#d97736", "#74b32d", "#bff2ff", "#896cd9", "#b20030", "#7f4620", "#57d900", "#566d73", "#7736d9", "#b22d50", "#ffb380", "#315916", "#0099e6", "#473366", "#8c4659", "#bf8660", "#1c330d", "#003c59", "#440080", "#cc001b", "#402d20", "#598040", "#23698c", "#754d99", "#990014", "#ffd9bf", "#4c5943", "#0d2633", "#655673", "#4c000a", "#ff8800", "#87a67c", "#609fbf", "#6f00a6", "#731d28", "#995200", "#0e6600", "#335566", "#2b0040", "#d96c7b", "#593000", "#90ff80", "#7c98a6", "#ca79f2", "#331b00", "#00ff00", "#262f33", "#b800e6", "#e59539", "#bfffbf", "#0088ff"]
    distances = np.genfromtxt("Master/data/tree_distances_matrix-train.npy")
    labels = np.load("Master/data/trees_clustered_vq-train.npy")
    
    points = d2c(distances)

    #assign labels to points
    for p in points:
        p.set_label(labels[p.index])
        p.pprint()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.savefig("3d-tree-distance-view.png")

    for p in points:
        x, y, z = p.coordinates
        ax.scatter(x, y, z, c=colors183[p.label])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
  
    ax.set_title('Trees distance view')

    plt.show()