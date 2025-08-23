import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import pickle
import os, subprocess
def svg_to_emf(svg_figpath):
    if True:
        cwd_path = os.getcwd()
        svg_figpath = os.path.join(cwd_path, svg_figpath)
        print(svg_figpath)
    inkscape_path = 'D:\\Pycharm\\Inkscape\\bin\\inkscape.exe' 
    if svg_figpath is not None:
        path, svgfigname = os.path.split(svg_figpath)
        figname, figform = os.path.splitext(svgfigname)
        emf_figpath = os.path.join(path, figname + '.emf')
        subprocess.call("{} {} -T -o {}".format(inkscape_path, svg_figpath, emf_figpath), shell=True)
        # os.remove(svg_figpath)

def getRotation_RotVecDegree(vec0, vec1):
    # as the same with the Matlab function: vrrotvec.m
    if type(vec0) == 'list':
        vec0 = np.array(vec0)
    if type(vec1) == 'list':
        vec1 = np.array(vec1)
    vec0_norm = vec0 / np.linalg.norm(vec0)
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vecAx = np.cross(vec0, vec1)
    vecAx_norm = vecAx / np.linalg.norm(vecAx)
    angle = np.arccos(np.dot(vec0_norm, vec1_norm))
    return vecAx_norm, angle

def getRotationMatrix(vec0=None, vec1=None, vecAx=None, angle=None):
    # as the same with the Matlab function: vrrotvec2mat.m
    # https://en.wikipedia.org/wiki/Rotation_matrix
    if vecAx==None and angle==None:
        vecAx, angle = getRotation_RotVecDegree(vec0, vec1)
    if type(vecAx) == 'list':
        vecAx = np.array(vecAx)
    s = np.sin(angle)
    c = np.cos(angle)
    t = 1 - c
    vecAx = vecAx / np.linalg.norm(vecAx)
    x, y, z = vecAx 
    rotMatrix = np.array([[t*x*x + c,    t*x*y - s*z,  t*x*z + s*y], 
                          [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
                          [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]])
    return rotMatrix
    
def calc_theta_and_phi_byEvectorAtOrigin(EvecOrigin):
    # Compute the polar angle θ (0 ~ 180°)
    theta_radians = np.arccos(EvecOrigin[2] / np.linalg.norm(EvecOrigin)) # cos_theta
    theta_degrees = np.degrees(theta_radians)
    # Compute the azimuthal φ (-180 ~ 180°)
    phi_radians = np.arctan2(EvecOrigin[1], EvecOrigin[0])
    phi_degrees = np.degrees(phi_radians)
    return theta_degrees, phi_degrees
    
def calc_ROISurfEvec_toOriginEfieldDirection(elmEvecs, elmNormals):
    num_elem     = elmEvecs.shape[0]
    thetaElement = np.zeros(num_elem)
    phiElement   = np.zeros(num_elem)
    for i in range(num_elem):
        Evec, normal = elmEvecs[i,:], elmNormals[i,:]
        rotMatrix_normal = getRotationMatrix(vec0=normal, vec1=[0, 0, 1])
        Evec_origin = np.dot(rotMatrix_normal, elmEvecs[i, :])
        thetaElement[i], phiElement[i] = calc_theta_and_phi_byEvectorAtOrigin(Evec_origin)
    return thetaElement, phiElement

def rayleigh_test(x, y=None):
    if y is None:
        y = np.ones_like(x)
    C = np.sum(y * np.cos(x))
    S = np.sum(y * np.sin(x))
    R = np.sqrt(C**2 + S**2)
    n = len(x)
    Z = R**2 / n
    p_value = np.exp(-Z)
    return R, Z, p_value

# def calc_list_color(value_data=np.zeros(10), value_min=-1, value_max=1, color_name='jet', color_numPoint=30000, method='nonsymmetry'):
#     value_array = np.linspace(value_min, value_max, color_numPoint)
#     cmap        = plt.get_cmap(color_name, lut=color_numPoint)
#     list_color = []
#     if method == 'nonsymmetry':
#         value_array_negative = np.linspace(value_min, 0.0, int(color_numPoint/2))
#         value_array_positive = np.linspace(0.0, value_max, int(color_numPoint/2))
#         for i in range(len(value_data)):
#             if value_data[i] >= value_max:
#                 list_color.append(cmap(color_numPoint-1))
#             elif value_data[i] <= value_min:
#                 list_color.append(cmap(0))
#             elif value_data[i] < 0.0:
#                 closest_indices = np.argmin(np.abs(value_array_negative - value_data[i]))
#                 list_color.append(cmap(closest_indices))
#             else:
#                 closest_indices = np.argmin(np.abs(value_array_positive - value_data[i]))
#                 list_color.append(cmap(closest_indices + int(color_numPoint/2)))
#     if method == 'symmetry':
#         for i in range(len(value_data)):
#             if value_data[i] >= value_max:
#                 list_color.append(cmap(color_numPoint-1))
#             elif value_data[i] <= value_min:
#                 list_color.append(cmap(0))
#             else:    
#                 closest_indices = np.argmin(np.abs(value_array - value_data[i]))
#                 list_color.append(cmap(closest_indices))
#     return list_color


def calc_list_color(value_data=np.zeros(10), value_min=-1, value_max=1, 
                    color_name='jet', color_numPoint=30000, method='symmetry', cmap=None):

    value_data = np.asarray(value_data)
    
    if cmap is None:
        cmap = plt.get_cmap(color_name, lut=color_numPoint)

    if method == 'nonsymmetry':
        # 将数据分为负和正两部分，分别归一化到 [0, 1)
        half = color_numPoint // 2
        list_color = np.empty((len(value_data), 4))  # RGBA 格式
        negative_mask = value_data < 0
        positive_mask = ~negative_mask

        # 负值部分映射
        neg_vals = value_data[negative_mask]
        neg_indices = np.clip(((neg_vals - value_min) / (0.0 - value_min) * (half - 1)).astype(int), 0, half - 1)
        list_color[negative_mask] = cmap(neg_indices)

        # 正值部分映射
        pos_vals = value_data[positive_mask]
        pos_indices = np.clip(((pos_vals - 0.0) / (value_max - 0.0) * (half - 1)).astype(int) + half, half, color_numPoint - 1)
        list_color[positive_mask] = cmap(pos_indices)

    elif method == 'symmetry':
        # 所有值线性映射到 color_numPoint
        indices = np.clip(((value_data - value_min) / (value_max - value_min) * (color_numPoint - 1)).astype(int), 0, color_numPoint - 1)
        list_color = cmap(indices)

    else:
        raise ValueError("method must be 'symmetry' or 'nonsymmetry'")

    return list_color.tolist()




def fit_gauss(x=np.zeros(10), paras=[[0.,0.,0.], ], order=2):
    y = np.zeros_like(x)
    if len(paras) == order:
        for i in range(order):
            a, b, c = paras[i]
            y += a*np.exp(-((x-b)/c)**2)
    return y


def calc_cell_effective_polarization_length_at_given_direction(coords, theta, phi, angle='degree'):
    # Compute the cell projection length at a given E-field direction
    if angle == 'degree':
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    # Convert the spherical coordinate system to Cartesian coordinate system
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    direction = np.array([x, y, z]) # unit vector
    # Compute the projection using dot product
    projections = np.dot(coords, direction)
    effective_polarization_length = np.max(projections) - np.min(projections)
    return effective_polarization_length