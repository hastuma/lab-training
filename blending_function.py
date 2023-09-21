import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp
from matplotlib.path import Path
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from numpy.linalg import inv, det
def bilinear_interpolation(image, x, y):
    if len(image.shape) == 2 :
        channel = 1
    else :
        channel = 3
    height, width = image.shape[0],image.shape[1]
     # Check if indices are within valid range
    if x < 0 or x >= width - 1  :
        return 0  
    elif y < 0 or y >= height - 1: 
        return 0 
    x_floor = int(np.floor(x))
    y_floor = int(np.floor(y))
    x_ceil = min(x_floor + 1, width - 1)
    y_ceil = min(y_floor + 1, height - 1)
    


    x_fraction = x - x_floor
    y_fraction = y - y_floor
    
    top_left = image[y_floor, x_floor]
    top_right = image[y_floor, x_ceil]
    bottom_left = image[y_ceil, x_floor]
    bottom_right = image[y_ceil, x_ceil]
    
    interpolated_value = (1 - x_fraction) * (1 - y_fraction) * top_left + x_fraction * (1 - y_fraction) * top_right + \
                         (1 - x_fraction) * y_fraction * bottom_left + x_fraction * y_fraction * bottom_right
    if channel == 1 :
        interpolated_value=int(interpolated_value)
        if interpolated_value>255:
            interpolated_value=255
        elif interpolated_value<0:
            interpolated_value=0
        return interpolated_value
    for i in range(channel):
        interpolated_value[i]=int(interpolated_value[i])
        if interpolated_value[i]>255:
            interpolated_value[i]=255
        elif interpolated_value[i]<0:
            interpolated_value[i]=0
    return interpolated_value
def show_triangle_image(img,set):
    points = np.array(set)
    triangulation = sp.Delaunay(points)
    # Create a plot
    fig, ax = plt.subplots()
    # Plot the image as background
    ax.imshow(img)
    # Plot Delaunay triangles
    for simplex in triangulation.simplices:
        vertices = points[simplex]
        polygon = Polygon(vertices, edgecolor='blue', fill=None)
        ax.add_patch(polygon)
    plt.title('Delaunay Triangulation on Image')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def points_in_triangle(dst_tri_pts): # give three point of triangle returns a list containing all points contained 
    p0=dst_tri_pts[0]
    p1=dst_tri_pts[1]
    p2=dst_tri_pts[2]
    triangle = Path([p0, p1, p2])
    bounding_box = np.array([[
        min(p0[0], p1[0], p2[0]),
        min(p0[1], p1[1], p2[1])
    ], [
        max(p0[0], p1[0], p2[0]),
        max(p0[1], p1[1], p2[1])
    ]])

    points = []
    for x in range(int(bounding_box[0, 0]), int(bounding_box[1, 0]) + 1):
        for y in range(int(bounding_box[0, 1]), int(bounding_box[1, 1]) + 1):
            if triangle.contains_point((x, y)):
                points.append((x, y))

    return points

def inverse_warp_triangle(src_image, affine_matrix, src_tri_pts, dst_tri_pts,canva):

    # Compute inverse affine matrix
    inverse_affine_matrix = np.linalg.inv(affine_matrix)

    target=points_in_triangle(dst_tri_pts)
    
    for point in target :
        x,y = point
        dst = np.array([x,y,1])
        src = np.matmul(inverse_affine_matrix,dst)
        val = bilinear_interpolation(src_image,src[0],src[1])
        #print(val/total_population)
        canva[y][x]=(val)                
        #canva[y][x]+=(val) ######### += 是給大眾平均臉的
    return canva 

def plot_points_on_image(image, points, title=None):
    points = np.array(points)
    plt.imshow(image, cmap='gray')  # Display the image
    plt.scatter(points[:, 0], points[:, 1], c='red', marker='o')  # Overlay points on the image
    plt.title(title)  # Set the title of the plot
    plt.show() 
def plot_delaunay_triangles(image, points, title=None):
    points = np.array(points)
    triangulation = Delaunay(points)
    plt.imshow(image, cmap='gray')  # Display the image
    for simplex in triangulation.simplices:
        for i in range(3):
            plt.plot(points[simplex[[i, (i + 1) % 3]], 0], 
                     points[simplex[[i, (i + 1) % 3]], 1], 'r-')
    plt.plot(points[:, 0], points[:, 1], 'o', markersize=3, color='blue')  # Plot points
    plt.title(title)  
    plt.show()  

def morph(im1, im2, im1_pts, im2_pts, warp_frac,dissolve_frac):
    a=np.float32(np.array(im1_pts))
    b=np.float32(np.array(im2_pts))
    a_frac=np.multiply(a,warp_frac)
    b_frac=np.multiply(b,1-warp_frac)
    set_mean = np.uint16(np.add(a_frac,b_frac))
    points = np.array(set_mean)
    triangulation = sp.Delaunay(points)
    # run through every triangle , simplex 存了每個座標在原本list裡的idx
    transform_1=[]
    transform_2=[]
    canva=np.copy(im1)

    for simplex in triangulation.simplices: 
        original = np.array(a[simplex]).T
        newrow = [1,1,1]
        original = np.vstack([original, newrow])
        vertices = np.vstack([np.array(points[simplex]).T,newrow])
        matrix= np.matmul(vertices,np.linalg.inv(original))
        transform_1.append(matrix)
        canva = inverse_warp_triangle(im1, matrix, a[simplex], points[simplex],canva)
    canva_1=np.copy(canva)

    canva=np.copy(im2)

    for simplex in triangulation.simplices: 
        original = np.array(b[simplex]).T
        newrow = [1,1,1]
        original = np.vstack([original, newrow])
        vertices = np.vstack([np.array(points[simplex]).T,newrow])
        matrix= np.matmul(vertices,np.linalg.inv(original))
        transform_2.append(matrix)
        canva = inverse_warp_triangle(im2, matrix, a[simplex], points[simplex],canva)
    canva_2 = np.copy(canva)
    canva_1 = np.multiply(canva_1,dissolve_frac)
    canva_2 = np.multiply(canva_2,1-dissolve_frac)
    canva = np.add(np.uint16(canva_1),np.uint16(canva_2))
    canva = np.clip(canva,0,255)
    canva = canva.astype('uint8')
    # plt.imshow(canva)  
    # plt.title(str(dissolve_frac))            
    # plt.savefig(str(dissolve_frac)+"_morph.png")
    # plt.show()      
    return canva


def empty_triangle_area(image, vertices):
    # Compute Delaunay triangulation
    triangulation = Delaunay(vertices)
    
    # Get the bounding box of the triangle
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    
    # Iterate through bounding box and set pixel values to zero inside the triangle
    for x in range(int(min_x), int(max_x) + 1):
        for y in range(int(min_y), int(max_y) + 1):
            if triangulation.find_simplex((x, y)) >= 0:
                image[y, x] = 0
    return image

def population_morph(im1, im2, im1_pts, im2_pts, total_population):
    im1 = im1.astype('uint32')
    a=np.float32(np.array(im1_pts))
    b=np.float32(np.array(im2_pts))
    points = np.array(im1_pts)
    triangulation = sp.Delaunay(points)
    # run through every triangle , simplex 存了每個座標在原本list裡的idx
    for simplex in triangulation.simplices: 
        original = np.array(b[simplex]).T
        newrow = [1,1,1]
        original = np.vstack([original, newrow])
        vertices = np.vstack([np.array(points[simplex]).T,newrow])
        # print("det = ", det(original))
        if  int(det(original))== 0:
            return (np.zeros_like(im1))
        mat=np.linalg.inv(original)
        matrix= np.matmul(vertices,mat)
        im1 = inverse_warp_triangle(im2, matrix, b[simplex], points[simplex],im1)  
    im1 = im1.astype('uint32')
    return im1

def read_pts_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('{') and not line.startswith('}') and not line.startswith('version:')and not line.startswith('n_points:'):
                x, y = map(float, line.split())
                points.append((int(x), int(y)))
    return points


# img = plt.imread('./frontalimages_spatiallynormalized/1a.jpg')
# print(img.shape)
# img = plt.imread('./frontalimages_spatiallynormalized/2a.jpg')
# print(img.shape)
"""
set1=[ (232,132),(275,103),(323,101),(368,101),(403,132),(325, 142), (244, 178), (218, 179), (303, 175), (208, 226), (278, 194), (326, 195), (220, 279), (245, 329), (292, 244), (282, 288), (312, 295), (325, 236), (323, 273), (323, 296), (322, 353), (403, 181), (423, 178), (345, 173), (425, 217), (365, 190), (420, 275), (400, 323), (355, 243), (364, 288), (335, 295)]
set2=[ (226,100),(271,62),(316,62),(364,79),(399,103),(316, 121), (242, 162), (218, 164), (290, 166), (216, 187), (273, 187), (313, 191), (223, 247), (227, 295), (290, 252), (279, 288), (307, 307), (313, 243), (316, 283), (317, 304), (314, 348), (392, 164), (407, 161), (342, 162), (405, 190), (360, 190), (396, 251), (388, 300), (338, 247), (347, 287), (329, 305)]
set_mean=[ (229, 116) ,(273,82),(319,  81) ,(366 ,  90 ), (401 , 117),(320, 131), (243, 170), (218, 171), (296, 170), (212, 206), (275, 190), (319, 193), (221, 263), (236, 312), (291, 248), (280, 288), (309, 301), (319, 239), (319, 278), (320, 300), (318, 350), (397, 172), (415, 169), (343, 167), (415, 203), (362, 190), (408, 263), (394, 311), (346, 245), (355, 287), (332, 300)]
num=1
im1 = plt.imread('./data/'+str(num)+'-11.jpg')
im2 = plt.imread('./data/'+str(num+1)+'-11.jpg')


########################################################### part 2 "Mid-way Face" ###########################################################
morph(im1, im2, set1, set2, 0.5)"""
