#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Homography, Fundamental Matrix and Two View SfM
#
# Date: 5 September 2024
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 2.0
#
#####################################################################################
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvLibrary import *

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2



# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)


def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)


def plot2DPoints(x, strColor='r', marker='x', markersize=10):
    """
    Plot 2D points on the current image/figure.
     -input:
         x: (2xN or 3xN) points in image coordinates.
         strColor: point color.
         marker: matplotlib marker.
         markersize: marker size.
     -output: None
    """
    plt.plot(x[0, :], x[1, :], marker, color=strColor, markersize=markersize, linestyle='None')


def plotImageScene(img, x, title, figNum=None, strColor='r', numbered=True, offset=(10, 0), wait=True):
    """
    Show an image with its 2D points overlaid.
     -input:
         img: RGB image.
         x: (2xN or 3xN) points in image coordinates.
         title: figure title.
         figNum: matplotlib figure number (None creates a new one).
         strColor: color of the points and their indexes.
         numbered: if True, write the index of each point next to it.
         offset: offset from the point to its index text.
         wait: if True, block until the user clicks on the figure.
     -output: None
    """
    plt.figure(figNum)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plot2DPoints(x, strColor)
    if numbered:
        plotNumberedImagePoints(x, strColor, offset)
    plt.title(title)
    plt.draw()  # We update the figure display

    if wait:
        print('Click in the image to continue...')
        plt.waitforbuttonpress()


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

def plot3DPoints(ax, X_w, strColor='b', marker='.'):
    """
    Scatter 3D points on a 3D axis.
     -input:
         ax: axis handle (projection='3d')
         X_w: (3xN or 4xN) points in world coordinates.
         strColor: point color.
         marker: matplotlib marker.
     -output: None
    """
    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker=marker, color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

def plot3DScene(X_w, cameras, ax=None, strColor='b', bbox=(0, 4), show=True):
    """
    Plot reference systems and 3D points in a 3D axis.
     -input:
         X_w: (3xN or 4xN) points in world coordinates.
         cameras: dict {name: T_w_c (4x4)} of reference systems to draw.
         ax: axis handle (projection='3d'). If None, a new figure and axis are created.
         strColor: color of the points.
         bbox: (min, max) of the fake bounding box to emulate axis('equal').
         show: if True, call plt.show() at the end.
     -output:
         ax: the axis used, so more elements can be added when show=False.
    """
    if ax is None:
        plt.figure()
        ax = plt.axes(projection='3d', adjustable='box')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for name, T in cameras.items():
        drawRefSystem(ax, T, '-', name)

    plot3DPoints(ax, X_w, strColor)

    # Matplotlib does not correctly manage axis('equal')
    fake = np.linspace(bbox[0], bbox[1], 2)
    ax.plot(fake, fake, fake, 'w.')

    if show:
        print('Close the figure to continue. Left button for orbit, right button for zoom.')
        plt.show(block=True)    

    return ax

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_w_c1 = np.loadtxt(DATA_LAB + 'T_w_c1.txt')
    T_w_c2 = np.loadtxt(DATA_LAB +'T_w_c2.txt')

    K_c = np.loadtxt(DATA_LAB + 'K_c.txt')
    X_w = np.loadtxt(DATA_LAB + 'X_w.txt')

    x1 = np.loadtxt(DATA_LAB + 'x1Data.txt')
    x2 = np.loadtxt(DATA_LAB + 'x2Data.txt')


    ##Plot the 3D cameras and the 3D points
    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    cameras = {'W': np.eye(4), 'C1': T_w_c1, 'C2': T_w_c2}
    plot3DScene(X_w, cameras, ax=ax)

    # ## 2D plotting example
    # img1 = cv2.cvtColor(cv2.imread(DATA_LAB + 'image1.png'), cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(cv2.imread(DATA_LAB + 'image2.png'), cv2.COLOR_BGR2RGB)


    # plotImageScene(img1, x1, 'Image 1', figNum=1)
    # plotImageScene(img2, x2, 'Image 2', figNum=2)