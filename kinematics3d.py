## Helper functions for working with 3D kinematics
import numpy as np
import scipy.io


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
    
def compute_joint_angle(pred, k1, kv, k2 degrees = True):
    '''
    Compute the angle between keypoints across time
    
    pred: DANNCE prediction data. Expected to be shape Tx1x3xK or Tx3xK
    k1: keypoint to use as first end point
    kv: keypoint to use as vertex point
    k2: kepoint to use as second end point
    degrees: boolean flag that returns angle in degrees if true and in radians if false 
    
    Returns angle: a T x 1 numpy array containing the angle between keypoints in degrees at each time point
    '''
    if len(pred.shape) > 3:
        pred = np.squeeze(pred)
    
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

def compute_joint_angles(pred, parsed_joint_names, edges, angles_to_compute = 'all', degrees = True):
    '''
    Compute specified joint angles between keypoints across time

    pred: DANNCE prediction data. Expected to be shape Tx1x3xK or Tx3xK
    parsed_joint_names: a python list with each index matching the corresponding joint name in the skeleton
    edges: an E x 2 numpy array containing the indices of each edge endpoint, where E is the total number of edges in the skeleton
    angles_to_compute: a list of joint names whose angles will be computed, or 'all' for computing all joint angles
    degrees: boolean flag that returns angle in degrees if true and in radians if false 

    Returns angles: a T x M numpy array containing the angle at each specified joint for each time point, where M is the total number of 
                    computed joint angles
            angle_points: a M x 3 numpy array containing the computed joint angle indices
    '''
    if len(pred.shape) > 3:
        pred = np.squeeze(pred)
        
    if angles_to_compute == 'all':
        joint_idx = range(1, len(parsed_joint_names + 1)
                          
    else:
        # find all joints that match supplied names/idx, then compute those angles only
        if isinstance(angles_to_compute[0], str):  
            joint_idx = np.where(np.isin(joint_names, angles_to_compute))[0] + 1
        elif isinstance(angles_to_compute[0], int):
            joint_idx = angles_to_compute + 1
            
    angles = np.zeros((pred.shape[0], 0)) # Difficult to compute how many total angles will be computed, so will grow numpy array from this
    angle_points = np.zeros((0, 3))
            
    for i, joint in zip(joint_idx, parsed_joint_names[joint_idx - 1]):
        # Find all nodes connected to this joint
        connected_nodes = np.where(edges == i)[0]
        n_angles_this_joint = len(connected_nodes) * (len(connected_nodes) - 1) / 2
        angles_this_joint = np.zeros((pred.shape[0], n_angles_this_joint))
        points_this_joint = np.zeros((n_angles_this_joint, 3)    
        # Iterate over each pair of connected nodes and compute joint angle. Will skip over joints with fewer than 2 connections
        l = 0
        for j in range(len(connected_nodes) - 1):
            for k in connected_nodes[j+1:]:
                angles_this_joint[:, l] = compute_joint_angle(pred, connected_nodes[j], i, connected_nodes[k], degrees = degrees)
                points_this_joint[l, :] = [connected_nodes[j], i, connected_joints[k]]
                l += 1
        angles = np.concatenate((angles, angles_this_joint), axis = 1)
        angle_points = np.concatenate((angle_points, points_this_joint), axis = 0)

    return angles, angle_points