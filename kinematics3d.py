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
                
## Egocentering and alignment
def egocenter(pred, bindcenter = 5, align = '3d', b1 = 3, b2 = 6, index_base = 1, keep_bindcenter = False):
    '''
    Egocenter data so that the bindcenter keypoint is situated at the origin. Optionally, this function
    also rotates (aligns) the skeleton so that the vector from b2 to b1 always points east.
    Adapted from Kanishk Jain's demoutils.egoh5 function in the mmpy repo

    pred: DANNCE prediction data. Expected to be shape Tx1x3xK or Tx3xK
    bindcenter: the index of the keypoint at which the coordinate frame will be centered. Default: 5, which corresponds to 'SpineM' in
                mouse22 assuming 1-indexing
    align: a string which specifies the alignment mode or None. Default: '3d'. Possible values: 
            - '3d': apply 3D alignment so that the alignment vector points to [1, 0, 0]
            - '2d': apply 2D alignment so that the alignment vector points to [1, 0, z], where z is the pre-alignment z coordinate of the alignment vector
            - None: don't apply any alignment
    b1: the index of the keypoint that acts as the endpoint of the alignment vector. Default: 3, which corresponds to 'Snout' in mouse22
        assuming 1-indexing
    b2: the index of the keypoint that acts as the start point of the alignment vector. Default: 6, which corresponds to 'Tail(base)' in
        mouse22 assuming 1-indexing (Note: bindcenter is assumed to be indexed between b1 and b2)
    index_base: specifies whether joint indices are specified as indexed from 0 or 1. Only needed if angles_to_compute is given
                as list of keypoint indices. Default: 1
    keep_bindcenter: boolean flag which determines whether bindcenter will be retained in the output. Default: False

    returns ego: if keep_bindcenter is False, a T x 3 x K-1 array containing egocentered coordinates for each keypoint excluding bindcenter across time
                 if keep_bindcenter is True, then an array with same shape as pred
    '''
    if len(pred.shape) > 3:
        pred = np.squeeze(pred)

    if index_base == 1:
        bindcenter -= 1
        b1 -= 1
        b2 -= 1
    elif index_base !=0:
        raise ValueError
    ## First do the egocentering


    # Apply egocentering by subtracing bindcenter position
    ego = pred - pred[:, [bindcenter for i in range(pred.shape[2]], :]

    if not keep_bindcenter:
        # Get indices of all keypoints that aren't the bindcenter
        idx_to_keep = np.setdiff1d(np.arange(pred.shape[2]), bindcenter)
        
        # Only keep the non-bindcenter points (since bindcenter is always at 0,0,0)
        ego = ego[:, :, idx_to_keep]
    
    if align is None:
        return ego
    
    ## Now do the alignment
    if align == '2d':
        # TODO: implement 2D alignment
        raise NotImplementedError
        
    # Grab the alignment vector
    if not keep_bindcenter:
        alignment_vector = ego[:, :, b1] - ego[:, b2 - 1] # subtract 1 to account for bindcenter removal
    else:
        alignment_vector = ego[:, :, b1] - ego[:, :, b2]

    # Normalize alignment vector
    alignment_vector /= np.linalg.norm(alignment_vector, axis=1)[:, np.newaxis]

    ## Compute and apply rotation matrix to each timepoint
    # Target is east
    target = np.array([1.0, 0.0, 0.0])
    
    for t in range(ego.shape[0]):
        v = alignment_vector[t]
        
        # Need an axis of rotation orthogonal to the alignment plane
        axis = np.cross(v, target)
        axis /= np.linalg.norm(axis)
        # Also need an angle of rotation
        angle = np.arccos(np.clip(np.dot(v, target), -1.0, 1.0))
        
        ## Compute rotation using Rodrigues' rotation formula
        # First set up the cross-product matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # Then apply the formula: I + sin(angle)K + (1 - cos(angle))K^2
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Finally apply the rotation
        ego[t] = R @ ego[t]
    return ego



    