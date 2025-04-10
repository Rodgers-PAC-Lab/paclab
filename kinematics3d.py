## Helper functions for working with 3D kinematics
import numpy as np
import scipy.io

## Loading functions
def load_COM(filepath):
    '''
    Load 3D center of mass predictions from a COM predictions file or com3d_used file.

    filepath: path to COM file
    
    Returns com: T x 3 numpy array containing COM predictions for every frame
    '''
    com = scipy.io.loadmat(filepath)['com']
    return com

def load_DANNCE(filepath):
    '''
    Load 3D keypoint predictions from a DANNCE predictions file
    
    filepath: path to DANNCE predictions file

    Returns pred: TxNx3xK numpy array, where T is the number of frames, N is the number of animals
             (always 1 for single-animal experiments), and K is the number of keypoints
    '''
    pred = scipy.io.loadmat(filepath)['pred']
    return pred

def load_skeleton(filepath):
    '''
    Load skeleton graph containing keypoint names as a python list and the edge indices as an E x 2 numpy array

    filepath: path to Label3D skeleton file. Note: a copy can be found at cuttlefish/lucas/prelim_dannce/mouse22.mat
    
    Returns parsed_joint_names: a python list with each index matching the corresponding joint name in the skeleton
            edges: an E x 2 numpy array containing the indices of each edge endpoint, where E is the total number of edges in the skeleton
    '''
    full_skeleton_data = scipy.io.loadmat(filepath)
    joint_names = full_skeleton_data['joint_names'][0]
    parsed_joint_names = []
    
    for joint in joint_names:
        parsed_joint_names.append(str(joint[0]))

    edges = full_skeleton_data['joints_idx']

    return parsed_joint_names, edges

## Angle functions

def compute_joint_angle(pred, k1, kv, k2, degrees = True, index_base = 1):
    '''
    Compute the angle between keypoints across time
    
    pred: DANNCE prediction data. Expected to be shape Tx1x3xK or Tx3xK
    k1: keypoint to use as first end point
    kv: keypoint to use as vertex point
    k2: kepoint to use as second end point
    degrees: boolean flag that returns angle in degrees if true and in radians if false. Default: true
    index_base: specifies whether joint indices are specified as indexed from 0 or 1. Default: 1
    
    Returns angle: a T x 1 numpy array containing the angle between keypoints in degrees at each time point
    '''
    if len(pred.shape) > 3:
        pred = np.squeeze(pred)

    if index_base == 1:
        k1 -= 1
        kb -= 1
        k2 -= 1
    elif index_base != 0:
        raise ValueError
    
    v1 = pred[:,:,k1] - pred[:,:,kb]
    v2 = pred[:,:,k2] - pred[:,:,kb]
    
    v1mag = np.sqrt(v1[:,0]**2 + v1[:,1]**2 + v1[:,2]**2)
    v2mag = np.sqrt(v2[:,0]**2 + v2[:,1]**2 + v2[:,2]**2)
    
    v1norm = v1 / np.column_stack((v1mag, v1mag, v1mag))
    v2norm = v2 / np.column_stack((v2mag, v2mag, v2mag))
    
    dot = np.sum(v1norm * v2norm, axis=1)

    angle = np.arccos(dot)
    
    if degrees:
        angle *= (180/np.pi)
        
    return angle

def compute_joint_angles(pred, parsed_joint_names, edges, angles_to_compute = 'all', degrees = True, index_base = 1):
    '''
    Compute specified joint angles between keypoints across time. Iterates over each keypoint specified in "angles_to_compute"
    as the vertex and computes all possible angles with pairs of connected nodes. This measures how open the joint is, but does not measure
    the orientation of the joint.

    pred: DANNCE prediction data. Expected to be shape Tx1x3xK or Tx3xK
    parsed_joint_names: a python list with each index matching the corresponding joint name in the skeleton
    edges: an E x 2 numpy array containing the indices of each edge endpoint, where E is the total number of edges in the skeleton
    angles_to_compute: a list of joint names whose angles will be computed, or 'all' for computing all joint angles
    degrees: boolean flag that returns angle in degrees if true and in radians if false 
    index_base: specifies whether joint indices are specified as indexed from 0 or 1. Only needed if angles_to_compute is given
                as list of keypoint indices. Default: 1
    
    Returns angles: a T x M numpy array containing the angle at each specified joint for each time point, where M is the total number of 
                    computed joint angles
            angle_points: a M x 3 numpy array containing the computed joint angle indices
    '''
    if len(pred.shape) > 3:
        pred = np.squeeze(pred)
        
    if angles_to_compute == 'all':
        joint_idx = range(len(parsed_joint_names)) # 0-indexed
                          
    else:
        # find all joints that match supplied names/idx, then compute those angles only
        if isinstance(angles_to_compute[0], str):  
            joint_idx = np.where(np.isin(joint_names, angles_to_compute))[0] # 0-indexed
        elif isinstance(angles_to_compute[0], int):
            if index_base == 1:
                joint_idx = angles_to_compute - 1
            elif index_base == 0:
                joint_idx = angles_to_compute
            else:
                raise ValueError
            
    angles = np.zeros((pred.shape[0], 0)) # Difficult to compute how many total angles will be computed, so will grow numpy array from this
    angle_points = np.zeros((0, 3))
            
    for i, joint in zip(joint_idx, parsed_joint_names[joint_idx]):
        # Find all nodes connected to this joint
        connected_nodes = np.where(edges == i + 1)[0]
        n_angles_this_joint = len(connected_nodes) * (len(connected_nodes) - 1) / 2 # given a vertex, we want to compute the angle between
                                                                                    # all pairs of connected nodes. This becomes a pairwise
                                                                                    # comparison, which means n(n-1)/2 angles for n connected
                                                                                    # nodes
        angles_this_joint = np.zeros((pred.shape[0], n_angles_this_joint))
        points_this_joint = np.zeros((n_angles_this_joint, 3)    
        # Iterate over each pair of connected nodes and compute joint angle. Will skip over joints with fewer than 2 connections
        l = 0
        for j in range(len(connected_nodes) - 1): # no need to consider the last connected node
            for k in connected_nodes[j+1:]: # compute angle between j and all other nodes connected to i
                angles_this_joint[:, l] = compute_joint_angle(pred, connected_nodes[j], i + 1, k, degrees = degrees)
                points_this_joint[l, :] = [connected_nodes[j], i + 1, k]
                l += 1
        angles = np.concatenate((angles, angles_this_joint), axis = 1)
        angle_points = np.concatenate((angle_points, points_this_joint), axis = 0)

    return angles, angle_points

## Distance functions
def compute_distance(a, b):
    '''
    Compute Euclidean distance between two points across time.

    a: a T x 3 array containing the 3D position of a point across time
    b: a T x 3 array containing the 3D position of a second point across time

    returns distance: a T x 1 array containing the distance between the points at each time
    '''
    distance = np.sqrt((a[:, 0] - b[:, 0])**2 + (a[:, 1] - b[:, 1])**2 + (a[:, 2] - b[:, 2])**2)
    return distance

def compute_all2all_distances(pred, edges = None, index_base = 1):
    '''
    Compute the 3D Euclidean distance between keypoints in the skeleton.
    pred: DANNCE prediction data. Expected to be shape Tx1x3xK or Tx3xK
    edges: None, or an E x 2 numpy array containing the joint indices whose distance between are to be computed
           If None, then the function will compute pairwise distance between all tracked keypoints (which is 231 distances for 22 keypoints)
    index_base: specifies whether joint indices are specified as indexed from 0 or 1. Only needed if angles_to_compute is given
                as list of keypoint indices. Default: 1

    Returns distances, a T x D numpy array containing the distances between keypoints at each time point, where D is the number of computed
            distances
    '''
    if len(pred.shape) > 3:
        pred = np.squeeze(pred)
        
    if edges is not None:
        if index_base == 1:
            edges -= 1
        elif index_base != 0:
            raise ValueError
        distances = np.zeros((pred.shape[0], len(edges)))
        
        for i, edge in zip(range(edges), edges):
            distances[:, i] = compute_distance(pred[:, :, edge[0]], pred[:, :, edge[1]])
            
    else:
        ## Compute true all2all distance matrix
        distances = np.zeros((pred.shape[0], pred.shape[2]*(pred.shape[2] - 1) / 2)) # n(n-1) / 2 pairwise distances between n keypoints
        k = 0
        for i in range(pred.shape[2] - 1):
            for j in range(i, pred.shape[2]):
                distances[:, k] = compute_distance(pred[:, :, i], pred[:, :, j])
                k += 1
    
    return distances
                





    