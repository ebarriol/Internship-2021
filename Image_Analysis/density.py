import numpy as np
import cv2
import os
import glob
import fnmatch
import copy
import matplotlib.pyplot as plt


folder = 'finalpredict'
data_type = '*.png'


def get_images_pre(path, extension, recursive):
    if not recursive:
        img_paths = glob.glob(path + extension)
    else:
        img_paths = []
        for root, directories, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, extension):
                img_paths.append(os.path.join(root, filename))

    img_paths.sort()
    return img_paths


def pretreatment(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED) # for a binary image
    image_clr = cv2.imread(filename) # for a colored image
    gray = cv2.cvtColor(image_clr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    adapt_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    return image, thresh, adapt_thresh


def explode_xy(xy):
    xl=[]
    yl=[]
    for i in range(len(xy)):
        xl.append(xy[i][0][0])
        yl.append(xy[i][0][1])
    return xl,yl


def shoelace_area(x_list,y_list):
    a1, a2=0, 0
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    for j in range(len(x_list)-1):
        a1 += x_list[j]*y_list[j+1]
        a2 += y_list[j]*x_list[j+1]
    l = abs(a1-a2)/2
    return l


def measure_density(filename, mask, hull):
    hull = hull[0]
    hull_list = hull.tolist()
    points = explode_xy(hull_list)
    area = shoelace_area(points[0], points[1])
    image = cv2.imread(filename)
    selection = np.zeros_like(image)
    selection[mask] = image[mask]
    rootPixels = np.count_nonzero(mask)
    #w = mask.shape[1]
    #h = mask.shape[0]
    density = rootPixels / (area)
    return density


def draw_contours(mask):
    mask1 = copy.deepcopy(mask)
    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # detecting contours
    hull = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], True))

    # create an empty black image
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0)  # green - color for contours
        color = (255, 0, 0)  # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    return drawing, hull, contours


def draw_largest_contour(mask):
    mask1 = copy.deepcopy(mask)
    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.vstack(contours)
    hull = [cv2.convexHull(contours)]
    drawing = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    color_contours = (0, 255, 0)
    color = (255, 0, 0)
    cv2.drawContours(drawing, contours, -1, color_contours, 1, 8)
    cv2.drawContours(drawing, hull, -1, color, 1, 8)
    return drawing, hull, contours


def main():
    d_list = []
    img_paths = get_images_pre(folder, extension=data_type, recursive=True)
    for i, img_path in enumerate(img_paths):
        step1 = pretreatment(img_path)

        contours = draw_contours(step1[0])
        largest_contour = draw_largest_contour(step1[0])

        step2 = measure_density(img_path, step1[0], largest_contour[1])
        cv2.imwrite("mask" + str(i) + ".png", step1[0])
        cv2.imwrite('output_contour_merge' + str(i) + '.png', largest_contour[0])
        cv2.imwrite('output_contours' + str(i) + '.png', contours[0])
        print("density for file ", img_path, ":", step2)
        d_list += [step2]
    return d_list


def plot(list):
    x_list = [i for i in range(len(list))]
    plt.plot(x_list, list)
    plt.xlabel("image number")
    plt.ylabel("density")
    plt.show()


if __name__ == "__main__":
    list = main() #[0.1957526507165401, 0.08011678239828607, 0.147725084275233, 0.20080462768032117, 0.18101561806740946, 0.2116557227819696, 0.10536776350549695, 0.22698522680080574, 0.1606405101697311, 0.18601560204800613, 0.2157426157975562, 0.08220008320570783, 0.233439767057401, 0.08806061443624448, 0.09721117228604499, 0.1383959213343065, 0.1327142649172342, 0.18154662765449758, 0.2043579555137361, 0.18542503173292413]
    print(list)
    plot(list)