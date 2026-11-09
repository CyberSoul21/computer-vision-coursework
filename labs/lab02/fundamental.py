import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvLibrary import *
from plotData_v2 import *

IMG1 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image1.png"), cv2.COLOR_BGR2RGB)
IMG2 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image2.png"), cv2.COLOR_BGR2RGB)

def plotEpipolarLinesInteractive(img1, img2, F_21, colors=('r', 'g', 'b', 'c', 'm', 'y')):
    """
    Click points in image 1 and draw their epipolar lines in image 2.
    Press Enter (or right-click) to finish.
     -input:
         img1, img2: images.
         F_21: (3x3) fundamental matrix mapping points in image 1 to lines in image 2 (l2 = F_21 @ x1).
         colors: colors cycled for each point/line pair.
     -output:
         x1: (3xN) clicked points in homogeneous coordinates.
         l2: (3xN) corresponding epipolar lines in image 2.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(img1); ax1.set_title("Click points in image 1 (Enter to finish)")
    ax2.imshow(img2); ax2.set_title("Epipolar lines in image 2")

    points, lines = [], []
    k = 0
    while True:
        plt.sca(ax1)
        pts = plt.ginput(1, timeout=0)
        if not pts:
            break
        u1, v1 = pts[0]
        c = colors[k % len(colors)]

        x1 = np.array([u1, v1, 1.0])
        l2 = F_21 @ x1

        ax1.plot(u1, v1, 'x', color=c, markersize=10)
        plt.sca(ax2)
        drawLine(l2,c)
        ax2.set_xlim(0, img2.shape[1]); ax2.set_ylim(img2.shape[0], 0)
        plt.draw()

        points.append(x1)
        lines.append(l2)
        k += 1

    plt.show(block=True)
    return np.array(points).T, np.array(lines).T



if __name__ == '__main__':
    
    print("Fundamental...")

    F_21_test = np.loadtxt(DATA_LAB + 'F_21_test.txt')

    # # Display image 1
    # plt.figure(1)
    # plt.imshow(IMG1)
    # plt.title("Click a point in image 1")
    # # Wait for one mouse click
    # points = plt.ginput(1, timeout=0)
    # #plt.close()
    # # Extract coordinates
    # u1, v1 = points[0]
    # # Convert to homogeneous coordinates
    # x1 = np.array([u1, v1, 1.0])
    # l2 = F_21_test @ x1
    # # Display image 2
    # plt.figure(2)
    # plt.imshow(IMG2)
    # plt.title("Epipolar line in Image 2")
    # drawLine(l2)

    plotEpipolarLinesInteractive(IMG1,IMG2,F_21_test)


    