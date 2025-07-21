## Helper functions for working with 3D kinematics
import os
import numpy as np
import scipy.io
from scipy.spatial.transform import Rotation as R
import pandas
import pickle
import ffmpeg

## Loading + misc helperfunctions
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

def frame2samp_id(frames):
    '''
    Convert video frame number to DANNCE sample ID value
    frames: Tx1, 1xT, or T, numpy array containing frame number
    
    returns sampleID, the DANNCE-generated sample ID value
    '''
    frames = np.squeeze(frames)
    sampleID = 20 * frames + 1
    return sampleID
    
def samp_id2frame(sampleID):
    '''
    Convert DANNCE sample ID value to video frame number
    sampleID: Tx1, 1xT or T, numpy array containing DANNCE-generated sample ID
    
    returns frames, the corresponding video frame numbers
    '''
    sampleID = np.squeeze(sampleID)
    frames = (sampleID - 1) // 20
    
    return frames

def extract_samples_by_exp(path_to_sample_pkl):
    '''
    Extract list of sampleIDs for each frame in a sample pickle file aggregated
    in lists by experiment ID (in order of label3d_file in io.yaml exp block.
    
    path_to_sample_pkl: path pointing to a training or validation sample pickle
    returns an aggregated pandas dataframe groupby with the sampleIDs in lists
    '''
    f = open(path_to_sample_pkl, 'rb')
    data = pickle.load(f)
    df =  pandas.DataFrame(np.int64(np.array(list(np.char.split(data, '_')))), columns = ['exp', 'sample'])
    return df.groupby('exp')['sample'].agg(list)    

def parse_exp(path_to_yaml, mode = 'dannce'):
    '''
    Parse io.yaml file exp block to recover session filepaths
    path_to_yaml: string filepath to io.yaml file
    mode: if 'com', grab the paths in the com_exp block
          if 'dannce', grab the paths in the exp block (default dannce)
          
    returns exp, a list of string filepaths to session dannce.mat files, and
            com_files, a list of string filepaths to session com file ('dannce'
            mode only)
    '''
    f = open(path_to_yaml, 'r')
    exp = []
    if mode == "com":
        # Iterate until com_exp block
        while True:
            line = f.readline()
            if not line:
                raise EOFError
            elif 'com_exp' in line:
                break

        
        # Iterate over all lines in com_exp block and append paths to exp
        while True:
            line = f.readline()
            if not line:
                raise EOFError
                
            elif '- label3d_file: ' in line:
                line = line.strip()
                line = line.split()
                if '../' in line[-1]:
                    # Convert relative path to absolute. TODO: make this more robust
                    exp.append(os.path.join(os.path.dirname(path_to_yaml), line[-1].strip("'").strip('"')))
                else:
                    exp.append(line[-1].strip("'").strip('"'))
            
            elif len(line.strip()) > 0:
                return exp
            
        
    elif mode == 'dannce':
        com_files = []
        # Iterate until exp block
        while True:
            line = f.readline()
            if not line:
                raise EOFError
                
            elif 'exp:' == line.strip():
                break
        
        # Iterate over all lines in exp block
        while True:
            line = f.readline()
            if not line:
                return exp, com_files
            
            elif "- label3d_file: " in line:
                line = line.strip()
                line = line.split()
                if '../' in line[-1]:
                    # Convert relative path to absolute. TODO: make this more robust
                    exp.append(os.path.join(os.path.dirname(path_to_yaml), line[-1].strip("'").strip('"')))
                else:
                    exp.append(line[-1].strip("'").strip('"'))
            elif "com_file: " in line:
                line = line.strip()
                line = line.split()
                if '../' in line[-1]:
                    # Convert relative path to absolute. TODO: make this more robust
                    com_files.append(os.path.join(os.path.dirname(path_to_yaml), line[-1].strip("'").strip('"')))
                else:
                    com_files.append(line[-1].strip("'").strip('"'))
    else:
        raise ValueError

def extract_selected_frames(input_video_path, output_video_path, frame_list):
    '''
    Write a video with select frames from an input video.
    
    input_file_path: string path to input video
    output_file_path: string path to new video file that will be written
    frame_list: a list of frame numbers from the input video to write
                to the output video (0-indexed)
    '''
    ## AI generated
    # FFmpeg select filter expression
    select_expr = '+'.join([f'eq(n,{f})' for f in sorted(frame_list)])

    stream = (
        ffmpeg
        .input(input_video_path)
        .filter_('select', select_expr)
        .filter_('setpts', 'N/FRAME_RATE/TB')  # Reset timestamps to avoid stretching
        .output(output_video_path, vcodec='libx264',r = 50, vsync='0')
    )

    ffmpeg.run(stream)


## Angle functions

def __compute_joint_angle(pred, k1, kv, k2, degrees = True, index_base = 1):
    '''
    OLD- do not use
    Compute the 3 point angle between keypoints across time
    
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
        kv -= 1
        k2 -= 1
    elif index_base != 0:
        raise ValueError
    
    v1 = pred[:,:,k1] - pred[:,:,kv]
    v2 = pred[:,:,k2] - pred[:,:,kv]
    
    v1mag = np.sqrt(v1[:,0]**2 + v1[:,1]**2 + v1[:,2]**2)
    v2mag = np.sqrt(v2[:,0]**2 + v2[:,1]**2 + v2[:,2]**2)
    
    v1norm = v1 / np.column_stack((v1mag, v1mag, v1mag))
    v2norm = v2 / np.column_stack((v2mag, v2mag, v2mag))
    
    dot = np.sum(v1norm * v2norm, axis=1)

    angle = np.arccos(dot)
    
    if degrees:
        angle = np.degrees(angle)
        
    return angle


def _compute_rotation_matrix(z, x_ref):
    z_mag = np.sqrt(z[:, 0]**2 + z[:, 1]**2 + z[:, 2]**2)
    x_ref_mag = np.sqrt(x_ref[:, 0]**2 + x_ref[:, 1]**2 + x_ref[:, 2]**2)

    z_norm = z / np.column_stack((z_mag, z_mag, z_mag))
    x_ref_norm = x_ref / np.column_stack((x_ref_mag, x_ref_mag, x_ref_mag))
    
    y = np.cross(z_norm, x_ref_norm)
    y_mag = np.sqrt(y[:, 0]**2 + y[:, 1]**2 + y[:, 2]**2)
    y_norm = y / np.column_stack((y_mag, y_mag, y_mag))

    x = np.cross(y_norm, z_norm)
    x_mag = np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)
    x_norm = x / np.column_stack((x_mag, x_mag, x_mag))

    z_norm = np.expand_dims(z_norm, axis = 2)
    y_norm = np.expand_dims(y_norm, axis = 2)
    x_norm = np.expand_dims(x_norm, axis = 2)

    if np.any(z_mag < 1):
        print("Warning: bone is very small!")
    if np.any(y_mag < 1e-1):
        print("Warning: bones are likely colinear!")
    if np.any(x_ref_mag < 1):
        print("Warning: reference bone is very small!")
    return np.concatenate((x_norm, y_norm, z_norm), axis = 2)
    
def compute_rotation_matrix(k1, kv, k2):
    '''
    OLD- do not use
    Compute rotation matrix for the point at kv relative to the vectors from kv
    through k1 and k2. The local z-axis is the vector from kv to k2. The local x-
    axis is the vector orthogonal to z and coplanar with z and the vector between
    kv and k1. The local y-axis is the cross product of x and z.
    
    k1: T x 3 position vector. Assumed to be the position of the joint proximal to kv
    kv: T x 3 position vector.
    k2: T x 3 position vector. Assumed to be the position of the joint distal to kv

    returns T x 3 x 3 tensor giving rotation matrix
    '''
    z = k2 - kv
    x_ref = k1 - kv
    
    return _compute_rotation_matrix(z, x_ref)

def compute_rotation_matrix2(ref1, ref2, kv, k2):
    '''
        OLD- do not use
        For two edges that don't share a keypoint. Useful for shoulder.
       kv: vertex joint
       k2: distal joint
       ref1: origin of reference vector
       ref2: endpoint of reference vector
    '''
    z = k2 - kv
    x_ref = ref2 - ref1
    x_ref_at_kv = kv + x_ref
    
    return _compute_rotation_matrix(z, x_ref_at_kv)


def compute_euler_angles(rot, degrees = True, order = 'zyx'):
    '''
    OLD- do not use
    Compute Euler angles (xyz convention) given rotation matrices. This is the pitch,
    roll, and yaw (i.e. the abduction, rotation, and flexion)

    orientations: T x M x 3 x 3
    degrees: returns answer in degrees if True, radians if False

    Returns: angles, T x M x 3 giving rotation, flexion, and abduction
    '''
    
    # Reshape (T, M, 3, 3) → (T*M, 3, 3)
    T = rot.shape[0]
    M = rot.shape[1]
    R_matrices_flat = rot.reshape(-1, 3, 3)
    
    # Create batched Rotation object
    R_obj = R.from_matrix(R_matrices_flat)
    
    # Decompose using desired order (e.g., 'zyx')
    angles = R_obj.as_euler(order, degrees = degrees)  # shape: (T*M, 3)
    
    # Reshape back to (T, M, 3)
    angles = angles.reshape(T, M, 3)
    
    return angles


# def compute_orientation(pred, k1, kv, k2, index_base = 1):
#     '''
#     Compute orientation matrices across time for the specified joint using Gram-Schmidt. The reference edge aligned with the x-axis
#     is the vector from kv to k1. The y-axis is coplanar with the reference edge and the vector from kv to k2. The z-axis is orthogonal
#     to the local xy plane.

#     pred: DANNCE prediction data. Expected to be shape Tx1x3xK or Tx3xK
#     k1: keypoint to use as end point of vector aligned with x-axis
#     kv: keypoint to use as vertex point
#     k2: kepoint to use as end point of vector defining the xy plane
#     index_base: specifies whether joint indices are specified as indexed from 0 or 1. Only needed if angles_to_compute is given
#             as list of keypoint indices. Default: 1
#     returns orientation: a T x 9 array giving the flattened orientation matrix at each time point (C order)
#     '''
#     if len(pred.shape) > 3:
#         pred = np.squeeze(pred)

#     if index_base == 1:
#         k1 -= 1
#         kb -= 1
#         k2 -= 1
#     elif index_base != 0:
#         raise ValueError

#     ## Refactor so these calculations are only performed once if computing joint angles and orientations in one pass
#     v1 = pred[:,:,k1] - pred[:,:,kb]
#     v2 = pred[:,:,k2] - pred[:,:,kb]

#     v1mag = np.sqrt(v1[:,0]**2 + v1[:,1]**2 + v1[:,2]**2)
#     v2mag = np.sqrt(v2[:,0]**2 + v2[:,1]**2 + v2[:,2]**2)
    
#     x = v1 / np.column_stack((v1mag, v1mag, v1mag))
#     v2norm = v2 / np.column_stack((v2mag, v2mag, v2mag)) # is this operation necessary?

#     v3 = np.cross(x, v2norm)
#     v3mag = np.sqrt(v3[:,0]**2 + v3[:,1]**2 + v3[:,2]**2)
#     z = v3 / np.column_stack((v3mag, v3mag, v3mag)) # z-axis

#     y = np.cross(x, z) # y-axis
#     orientation = np.column_stack((x, y, z))
#     return orientation

# def compute_joint_angles(pred, edges, parsed_joint_names = None, angles_to_compute = 'all', mode = 'full', degrees = True, index_base = 1):
#     '''
#     TODO: Refactor this function so it's easier to understand. Probably want to dispatch 3 similar functions depending on mode.
#     Also, might want to add a mode that returns Euler angles instead of orientation matrix
#     Compute specified joint angles between keypoints across time and optionally compute orientations matrices. 
#     Iterates over each keypoint specified in "angles_to_compute" as the vertex and computes all possible angles and/or orientations
#     with pairs of connected nodes. Joint angles measure how open the joint is, while the orientation matrix captures its spatial alignment.

#     pred: DANNCE prediction data. Expected to be shape Tx1x3xK or Tx3xK
#     edges: an E x 2 numpy array containing the indices of each edge endpoint, where E is the total number of edges in the skeleton
#     parsed_joint_names: a python list with each index matching the corresponding joint name in the skeleton, or None.
#                         Only necessary to specify if passing joint names to angles_to_compute
#     angles_to_compute: a list of joint names whose angles will be computed, or 'all' for computing all joint angles
#     mode: a string specifying which calculations to perform. Default is 'full'. Options:
#                     - 'full': compute joint angles and orientation matrices
#                     - 'angles_only': compute only joint angles
#                     - 'orientations_only': compute only orientation matrices
#     degrees: boolean flag that returns angle in degrees if true and in radians if false 
#     index_base: specifies whether joint indices are specified as indexed from 0 or 1. Only needed if angles_to_compute is given
#                 as list of keypoint indices. Default: 1
    
#     If mode = 'full', returns angles: a T x M numpy array containing the angle at each specified joint for each time point, where 
#                                       M is the total number of computed joint angles
#                               orientations: a T x M x 9 numpy array containing the flattened orientation matrix (C style) 
#                                             at each joint for each time
#                               angle_points: a M x 3 numpy array containing the computed joint indices
#     If mode = 'angles_only', returns angles and angle_points
#     If mode = 'orientations_only', returns orientations and angle_points
#     '''
#     if len(pred.shape) > 3:
#         pred = np.squeeze(pred)
        
#     if angles_to_compute == 'all':
#         joint_idx = range(pred.shape[2]) # 0-indexed
                          
#     else:
#         # find all joints that match supplied names/idx, then compute those angles only
#         if isinstance(angles_to_compute[0], str):
#             try:
#                 joint_idx = np.where(np.isin(joint_names, angles_to_compute))[0] # 0-indexed
#             except ValueError:
#                 raise ValueError("Need to specify parsed_joint_names if passing joint names to angles_to_compute")
                
#         elif isinstance(angles_to_compute[0], int):
#             if index_base == 1:
#                 joint_idx = angles_to_compute - 1
#             elif index_base == 0:
#                 joint_idx = angles_to_compute
#             else:
#                 raise ValueError
    
#     ## Set up output arrays
#     angle_points = np.zeros((0, 3))
#     if mode == 'full':
#         angles = np.zeros((pred.shape[0], 0)) # Difficult to compute how many total angles will be computed, so will grow numpy array from this
#         orientations = np.zeros((pred.shape[0], 0, 9))
#     elif mode == 'angles_only':
#         angles = np.zeros((pred.shape[0], 0)) # Difficult to compute how many total angles will be computed, so will grow numpy array from this
#     elif mode == 'orientations_only':
#         orientations = np.zeros((pred.shape[0], 0, 9))
#     else:
#         raise ValueError

            
#     for i, joint in zip(joint_idx, parsed_joint_names[joint_idx]): # TODO: add a verbose mode with tqdm and a silent mode
#         # Find all nodes connected to this joint
#         connected_nodes = np.where(edges == i + 1)[0]
#         n_angles_this_joint = len(connected_nodes) * (len(connected_nodes) - 1) / 2 # given a vertex, we want to compute the angle between
#                                                                                     # all pairs of connected nodes. This becomes a pairwise
#                                                                                     # comparison, which means n(n-1)/2 angles for n connected
#                                                                                     # nodes
#         if mode == 'full':
#             angles_this_joint = np.zeros((pred.shape[0], n_angles_this_joint))
#             orientations_this_joint = np.zeros((pred.shape[0], n_angles_this_joint, 3, 3))
#         elif mode == 'angles_only':
#             angles_this_joint = np.zeros((pred.shape[0], n_angles_this_joint))
#         elif mode == 'orientations_only':
#             orientations = np.zeros((pred.shape[0], 0, 9))
        
#         points_this_joint = np.zeros((n_angles_this_joint, 3))    
#         # Iterate over each pair of connected nodes and compute joint angle. Will skip over joints with fewer than 2 connections
#         l = 0
#         for j in range(len(connected_nodes) - 1): # no need to consider the last connected node
#             for k in connected_nodes[j+1:]: # compute angle between j and all other nodes connected to i
#                 if mode == 'full':
#                     angles_this_joint[:, l] = compute_joint_angle(pred, connected_nodes[j], i + 1, k, degrees = degrees) # could probably refactor to just pass the relevant keypoints. Might be faster
#                     orientations_this_joint[:, l] = compute_orientation(pred, connected_nodes[j], i + 1, k)
#                 elif mode == 'angles_only':
#                     angles_this_joint[:, l] = compute_joint_angle(pred, connected_nodes[j], i + 1, k, degrees = degrees)
#                 elif mode == 'orientations_only':
#                     orientations_this_joint[:, l] = compute_orientation(pred, connected_nodes[j], i + 1, k)
                    
#                 points_this_joint[l, :] = [connected_nodes[j], i + 1, k]
#                 l += 1
#         angle_points = np.concatenate((angle_points, points_this_joint), axis = 0)
#         if mode == 'full':
#             angles = np.concatenate((angles, angles_this_joint), axis = 1)
#             orientations = np.concatenate((orientations, orientations_this_joint), axis = 1)
#             return angles, orientations, angle_points
#         elif mode == 'angles_only':
#             angles = np.concatenate((angles, angles_this_joint), axis = 1)
#             return angles, angle_points
#         elif mode == 'orientations_only':
#             orientations = np.concatenate((orientations, orientations_this_joint), axis = 1)
#             return orientations, angle_points
        

## Chris' joint angle functions from 20250605_3d
def cart2sphere(x, y, z):
    """Convert cartesian to spherical (length, azim, elev)
    
    Azimuth of zero is parallel to x
    Elevation of zero is parallel to z
    
    See figure 5.7.10 here:
    https://math.libretexts.org/Courses/Mount_Royal_University/Calculus_for_Scientists_II/7%3A_Vector_Spaces/5.7%3A_Cylindrical_and_Spherical_Coordinates
    """
    length = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azim = np.arctan2(y, x)
    elev = np.arccos(z / length)
    assert (azim > -np.pi).all()
    assert (azim < np.pi).all()
    return length, azim, elev

def sphere2cart(length, azim, elev):
    """Convert spherical (length, azim, elev) to cartesian"""
    z = length * np.cos(elev)
    
    # Distance in the XY plane
    r = length * np.sin(elev)
    x = r * np.cos(azim)
    y = r * np.sin(azim)

    return x, y, z

def joint_angle(x1, x2, x3=None):
    """Compute the joint angle
    
    This is defined as the difference in the azimuth and the difference
    in the elevation between the two limbs (x1, x2) and (x2, x3).
    If (x1, x2) and (x2, x3) are collinear, then the joint_angle is (0, 0)
    
    Note that this is not at all the same as the azimuth and elevation 
    of the difference of the two limbs! That would be (x1, x3). 
    For intuition, think about the joint angle of the elbow. When the arm
    is straight, this joint angle will be zero, no matter which way the arm
    is pointing.
    
    The "compound angle" is also computed, which is the scalar angle
    between the two vectors (arccos of the dot product of the normed vectors)
    
    The input can be provided as points or as limb vectors.
    If points:
        x1, x2, x3 == p1, p2, p3: the three points (eg, shoulder, elbow, wrist)
        
    If vectors:
        x1, x2 == v1, v2: the vector from p1 to p2, and from p2 to p3
        In this case, x3 should be set to None
    
    In either case, each argument must be a DataFrame with columns 
    ['x', 'y', 'z'].
    
    Returns: DataFrame
        index: same as index on input
        columns: azim, el, compound
    """
    ## Turn input into limb vectors
    if x3 is None:
        # the limb vectors were provided
        v1 = x1
        v2 = x2
        
    else:
        # the joint positions were provided
        # calculate limb vectors
        v1 = x2 - x1
        v2 = x3 - x2
    
    
    ## Compute angle
    # Convert each limb to spherical coordinates
    sphere1 = pandas.concat(
        cart2sphere(v1['x'], v1['y'], v1['z']), 
        axis=1, keys=['length', 'azim', 'elev'])
    sphere2 = pandas.concat(
        cart2sphere(v2['x'], v2['y'], v2['z']), 
        axis=1, keys=['length', 'azim', 'elev'])
    
    # Diff the two limbs (dropping the irrelevant 'length')
    res = sphere2[['azim', 'elev']] - sphere1[['azim', 'elev']]
    
    # Pin to the correct range
    res = np.mod(res + np.pi, 2 * np.pi) - np.pi
    
    
    ## Also compute the compound angle
    # There may be some identity to compute this from azim and elev
    # https://www.reddit.com/r/askmath/comments/xgqw53/whats_the_formula_to_get_compound_angle_from/
    # But easier to do it in cartesian space
    
    # Normalize
    v1norm = v1.divide(sphere1['length'], axis=0)
    v2norm = v2.divide(sphere2['length'], axis=0)
    
    # Arccos of the dot product
    res['compound'] = np.arccos((v1norm * v2norm).sum(axis=1))

    ## Return
    return res

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


def nanmean_infmean(loss, axis=None):
    """
    Adapted from sdannce.dannce.engine.models.metrics.py
    Safe mean that handles nans and infinities.
    """
    mask = (~np.isnan(loss)) & (~np.isposinf(loss)) & (~np.isneginf(loss))
    valid_count = np.sum(mask, axis=axis)
    
    if np.any(valid_count == 0):
        return 0  # or np.nan, depending on your desired behavior

    valid_sum = np.sum(np.where(mask, loss, 0), axis=axis)
    return valid_sum / valid_count

def euclidean_distance_3D(predicted, target, axis = 0):
    """
    Adapted from sdannce.dannce.engine.models.metrics.py
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    
    axis = 0 averages within keypoints across frames
    axis = 1 averages within frames across keypoints
    axis = None: no average
    """
    assert predicted.shape == target.shape
    assert predicted.shape[1] in [2, 3]

    mpjpe = np.linalg.norm((target - predicted), ord=2, axis=1)
    
    if axis is not None:
        return nanmean_infmean(mpjpe, axis)
    return mpjpe

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
            idx = edges - 1
        elif index_base == 0:
            idx = edges
        else:
            raise ValueError
        distances = np.zeros((pred.shape[0], len(idx)))
        
        for i, edge in zip(range(edges.shape[0]), idx):
            distances[:, i] = compute_distance(pred[:, :, edge[0]], pred[:, :, edge[1]])
            
    else:
        ## Compute true all2all distance matrix
        distances = np.zeros((pred.shape[0], int(pred.shape[2]*(pred.shape[2] - 1) / 2))) # n(n-1) / 2 pairwise distances between n keypoints
        k = 0
        for i in range(pred.shape[2] - 1):
            for j in range(i + 1, pred.shape[2]):
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
    ego = pred - pred[:, :, [bindcenter for i in range(pred.shape[2])]]

    if not keep_bindcenter:
        # Get indices of all keypoints that aren't the bindcenter
        idx_to_keep = np.setdiff1d(np.arange(pred.shape[2]), bindcenter)
        # Only keep the non-bindcenter points (since bindcenter is always at 0,0,0)
        ego = ego[:, :, idx_to_keep]
    
    if align is None:
        return ego
    
    ## Now do the alignment
    if align == '2d':
        # Grab the alignment vector
        if not keep_bindcenter:
            alignment_vector = ego[:, :2, b1] - ego[:, :2, b2 - 1] # subtract 1 to account for bindcenter removal
        else:
            alignment_vector = ego[:, :2, b1] - ego[:, :2, b2]
            
        # Normalize alignment vector
        alignment_vector /= np.linalg.norm(alignment_vector, axis=1)[:, np.newaxis]
    
        ## Compute and apply rotation matrix to each timepoint
        # Target is east
        target = np.array([1.0, 0.0])
        
        for t in range(ego.shape[0]):
            v = alignment_vector[t]
            
            R = np.array([
                [v[0], v[1]],
                [-v[1], v[0]] 
            ])
            
            # Apply rotation
            ego[t, :2, :] = R @ ego[t, :2, :]
        
    elif align == '3d':
        # Grab the alignment vector
        if not keep_bindcenter:
            alignment_vector = ego[:, :, b1] - ego[:, :, b2 - 1] # subtract 1 to account for bindcenter removal
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

# Smoothing and derivatives
def gaussianfilterdata(pred, sigma = 1):
    '''
    Convolve data with a Gaussian. This computes smoothed position.
    pred: DANNCE prediction data. Expected to be shape Tx1x3xK or Tx3xK
    sigma: the standard deviation of the Gaussian kernel
    '''
    
    L = pred.shape[0]
    xx = (np.arange(1, L) - np.round(L/2)).T
    g = np.exp(-.5 * xx**2 / sigma**2) / np.sqrt(2*np.pi*sigma**2) # Think this is correct. Confirm w/ a PI
    ddt = np.fft.fftshift(np.fft.ifft(np.fft.fft(pred[:-1]) * np.fft.fft(g)))
    return ddt
    
def gaussianfilterdata_derivative(pred, sigma = 1):
    '''
    Compute smoothed velocities by convolving position with the derivative of a Gaussian.
    Adapted from old matlab code written by Gordon.

    pred: angle data
    sigma: the standard deviation of the Gaussian derivative kernel. According to Gordon, this 
           approximately maps on to window size / 2 for a sliding window

    '''
    L = pred.shape[0]
    xx = (np.arange(1, L) - np.round(L/2)).T
    
    g = -xx * np.exp(-.5 * xx**2 / sigma**2) / np.sqrt(np.pi*sigma**6)
    ddt = np.zeros_like(pred)
    for i in range(pred.shape[1]):       # over x/y/z
            ddt[:, i] = np.fft.fftshift(
                np.fft.ifft(
                    np.fft.fft(pred[:, i]) * np.fft.fft(g, n=L)
                ).real
            )
    return ddt    