import os
import sys
import numpy as np
import h5py
import trimesh
from sklearn import decomposition


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
#DATA_DIR = os.path.join(BASE_DIR, 'data')
#if not os.path.exists(DATA_DIR):
#    os.mkdir(DATA_DIR)
#if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#    zipfile = os.path.basename(www)
#    os.system('wget %s; unzip %s' % (www, zipfile))
#    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h5_resampled(h5_filename,num_points):
    f = h5py.File(h5_filename)
    vertices = f['vertices'][:]
    faces = f['faces'][:]
    
    vertices = np.array(vertices, dtype=np.float)
    faces = np.array(faces, dtype=np.int) 
    numvertices = len(vertices)
    
    if (numvertices >= 0):
        m = trimesh.Trimesh(vertices,faces)
        cm = np.mean(m.vertices,axis=0)
        
        m.vertices = m.vertices-cm
        vertices, fi = trimesh.sample.sample_surface(m,num_points)

        vertices = np.expand_dims(vertices, axis=0)
        labels = np.ndarray([1])
        return (vertices,labels)
    else:
        return None,None
    
def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


def loadOffDataFile(filename,num_points):
    """Reads a mesh's vertices and faces from an off file"""
    print("This is filename: ")
    print(filename)
    assert os.path.isfile(filename)
    with open(filename) as fname:
    	content = fname.readlines()
    nums = content[1].split() # 0 - numvert, 1 - numfaces, 2 - numedges
    v = content[2:2+int(nums[0])]
    f = content[2+int(nums[0]): 2+int(nums[0])+int(nums[1])]

    vertices = [[float(i) for i in x.split()][:3] for x in v]
    faces = [[int(i) for i in x.split()][1:] for x in f]

    vertices = np.array(vertices, dtype=np.float)
    faces = np.array(faces, dtype=np.int) 
    numvertices = len(vertices)
    #print(numvertices)
    
    if (numvertices >= 0):
        m = trimesh.Trimesh(vertices,faces)
        

        #cm = m.center_mass
        cm = np.mean(m.vertices,axis=0)
        
        m.vertices = m.vertices-cm
        vertices, fi = trimesh.sample.sample_surface(m,num_points)

        
        #print(vertices.shape)
        #print(cm.shape)
        #exit(0)
        
        vertices = np.expand_dims(vertices, axis=0)
        labels = np.ndarray([1])
        return (vertices,labels)
    else:
        return None,None
    
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotate_to_principal_component(vertices):
    pca = decomposition.PCA(n_components=3)
    pca.fit(vertices)
    mainaxis = pca.components_[0]
    yaxis = np.asarray([1,0,0])
    rotation_matrix = rotation_matrix_from_vectors(yaxis,mainaxis)
    rotated_data = np.dot(vertices, rotation_matrix)
    return rotated_data
    
def loadAllOffDataFile(filename,num_points):
    """Reads a mesh's vertices and faces from an off file"""
    assert os.path.isfile(filename)
    with open(filename) as fname:
    	content = fname.readlines()
        
    print("This is off file: ", filename)
    nums = content[1].split() # 0 - numvert, 1 - numfaces, 2 - numedges
    v = content[2:2+int(nums[0])]
    f = content[2+int(nums[0]): 2+int(nums[0])+int(nums[1])]

    vertices = [[float(i) for i in x.split()][:3] for x in v]
    faces = [[int(i) for i in x.split()][1:] for x in f]

    vertices = np.array(vertices, dtype=np.float)
    faces = np.array(faces, dtype=np.int) 
    numvertices = len(vertices)
    #print(numvertices)
    
    if (numvertices > -100):
        m = trimesh.Trimesh(vertices,faces)
        

        #cm = m.center_mass
        cm = np.mean(m.vertices,axis=0)
        
        m.vertices = m.vertices-cm
        
        m.vertices = rotate_to_principal_component(m.vertices)
        
        vertices, fi = trimesh.sample.sample_surface(m,num_points)

        
        #print(vertices.shape)
        #print(cm.shape)
        #exit(0)
        
        vertices = np.expand_dims(vertices, axis=0)
        labels = np.ndarray([1])
        return (vertices,labels)
    else:
        return None,None
    
    
    