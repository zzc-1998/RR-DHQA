import numpy as np
import trimesh
from nss_functions import * 
import cv2

def mesh_or_scene(obj):
    try:
        # only trimesh object has vertices attribute. if the obj contains multiple meshes, it is loaded as scene obj.
        vertices = obj.vertices
        return obj
    except:
        print("The object is a scene with mulitple meshes, the main mesh is used.")
        meshes = list(obj.geometry.values())
        max = 0
        for i in range(len(meshes)):
            vertices = len(meshes[i].vertices)
            if vertices > max:
                max = vertices
                max_i = i
        return list(obj.geometry.values())[max_i]
        
    

def get_3d_nss_features(objpath,imgpath):
    print("Loading obj mesh..")
    mesh = trimesh.load(objpath)
    mesh = mesh_or_scene(mesh)
    # extract curvature features
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    #print("Mean curvature")
    r = max(extents) * 0.003
    #mean_curvature = np.array(trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, r))[0]
    gaussian_curvature = np.array(trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, r))
    # random sampling for mean_curvature_mesh
    vertices_array = np.array(mesh.vertices)
    if len(vertices_array) > 4000:
        idx = 4000
    else:
        idx = len(vertices_array)
    np.random.shuffle(vertices_array)
    # Assign it back to mesh.vertices if applicable
    mesh.vertices = vertices_array
    mean_curvature = np.array(trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices[:idx], r))
    defects = trimesh.curvature.vertex_defects(mesh)
    dihedral_angle = mesh.face_adjacency_angles


    print("Loading texture..")
    # read the texture
    image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

    # extract texture features
    gray = image.ravel()
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient = magnitude.ravel()
    nss_params = []

    for tmp in [mean_curvature,gaussian_curvature,dihedral_angle,defects,gray,gradient]:
        params = get_nss_param(tmp)
        #flatten the feature vector
        nss_params = nss_params + [i for item in params for i in item]
    
    type = ['mean_curvature','gaussian_curvature','dihedral_angle','defect','gray','gradient']
    for j in range(len(type)):
        for i in range(3+2+4):
            print(type[j]+str(i) + ": " + str(nss_params[j*9 + i]))
    return nss_params


objpath = "sample_dh/00005_Sam006_Basics_006_simp_rate_0.40.obj"
imgpath = "sample_dh/00005_Sam006_Diffuse_simp_rate_0.40.jpg"
get_3d_nss_features(objpath,imgpath)