import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

SCALE_FACTOR=0.05
PI = np.pi
EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


R_EE_TO_GRIPPER = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0]
])
TARGET_IMG_SIZE = (120, 120)
CROP_PARAMETERS = {
    'pick_place': [20, 25, 80, 75]
}
CAMERA_FRONT_POS = np.array([0.45, -0.002826249197217832, 1.27])
CAMERA_FRONT_QUAT = np.array([0.26169506249574287, 0.25790267731943883, 0.6532651777140575, 0.6620018964346217])
fov_y = 60  # degrees


Y_SPAWN_REGION = [[0.255, 0.195], [0.105, 0.045], [-0.045, -0.105], [-0.195, -0.255]]
TARGET_BOX_ID_NAME_DICT = {
    'pick_place': {
        0: 'greenbox',
        1: 'yellowbox',
        2: 'bluebox',
        3: 'redbox'
    }
}

def get_spawn_region(y_position):
    for indx, y_region in enumerate(Y_SPAWN_REGION):
        if y_position <= y_region[0] and y_position >= y_region[1]:
            spawn_indx = indx
            break
    return spawn_indx

def vec(values):
    """
    Converts value tuple into a numpy vector.

    Args:
        values (n-array): a tuple of numbers

    Returns:
        np.array: vector of given values
    """
    return np.array(values, dtype=np.float32)

def crop_and_resize(image, crop_parameters, target_size=(120, 120)):
    crop_params = crop_parameters
    top, left = crop_params[0], crop_params[2]
    img_height, img_width = image.shape[0], image.shape[1]
    box_h, box_w = img_height - top - \
    crop_params[1], img_width - left - crop_params[3]

    image = image[top:top + box_h, left:left + box_w]
    image = cv2.resize(image, target_size)
    return image

def save_camera_projection(
    rgb_image,
    cam_pos_film,
    fov_y,
    img_width,
    img_height,
    output_path,
    title="Camera-space projection debug"
):
    """
    Save an RGB frame with an overlaid camera-space projection (OpenCV).

    Args:
        rgb_image (np.array): (H, W, 3) RGB image
        cam_pos_film (np.array): (3,) camera film coords (x right, y down, z forward)
        fov_y (float): vertical field of view in degrees
        img_width (int)
        img_height (int)
        output_path (str): path to save the image
    """

    # OpenCV uses BGR internally
    img = rgb_image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    arrow_len = 20

    x, y, z = cam_pos_film

    if z <= 0:
        cv2.putText(
            img,
            "Point behind camera",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return

    # --- Intrinsics from FOV ---
    f = 0.5 * img_height / np.tan(fov_y * np.pi / 360)  # focal length in pixels

    cx = img_width / 2
    cy = img_height / 2

    # --- Projection ---
    u = int(round(f * (x / z) + cx))
    v = int(round(f * (y / z) + cy))

    # --- Draw ---
    if 0 <= u < img_width and 0 <= v < img_height:
        cv2.drawMarker(
            img,
            (u, v),
            color=(255, 0, 0),  # Red in RGB
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=20 if img_height > 120 else 10,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

        # Draw text
        cv2.putText(
            img,
            f"Projected point: ({u}, {v})",
            (0, 10), # upper-left corner of image
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 if img_height > 120 else 0.2,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"3D coords (film frame): ({x:.2f}, {y:.2f}, {z:.2f})",
            (0, 30), # slightly below the previous text
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 if img_height > 120 else 0.2,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        
        
        img = cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx) + arrow_len, int(cy)), (255, 0, 0), 2, tipLength=0.3)
        img = cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx), int(cy) + arrow_len), (0, 255, 0), 2, tipLength=0.3)

        
    else:
        cv2.putText(
            img,
            "Projection out of image bounds",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Save (convert RGB → BGR for OpenCV)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def convert_from_world_to_camera_space(pos, quat, camera_pos, camera_quat, fov_y, img_width, img_height, img, t, crop_params, debug=False):
    """Convert the robot position 'pos' and orientation 'quat' from world space to camera space

    Args:
        pos (np.array): 3D position in world
        quat (np.array): Quaternion in world
        camera_pos (np.array): Camera position in world
        camera_quat (np.array): Camera orientation in world
        fov_y (float): Field of view in y direction (degrees)
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
    """
    # World → camera rotation
    R_wc = R.from_quat(camera_quat)
    R_cw = R_wc.inv()

    # Position: translate then rotate
    rel_pos = pos - camera_pos
    cam_pos = R_cw.apply(rel_pos)

    # Orientation: relative rotation
    obj_rot = R.from_quat(quat)
    cam_rot = R_cw * obj_rot

    # -------------------------------------------------
    # Convert to FILM FRAME (x right, y down, z forward)
    # -------------------------------------------------
    # Typical camera frame after rotation:
    #   x right, y up, z forward
    # We flip Y to make it point down
    film_flip = R.from_matrix(
        np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0,  1]
        ])
    )

    cam_pos_film = film_flip.apply(cam_pos)
    cam_rot_film = film_flip * cam_rot

    if debug:
        os.makedirs("tmp_img", exist_ok=True)
        save_camera_projection(
            rgb_image=img,
            cam_pos_film=cam_pos_film,
            fov_y=fov_y,
            img_width=img_width,
            img_height=img_height,
            output_path=f"tmp_img/debug_camera_projection_{t}.png",
        )
        
    # Bring cam_pos_film_px to film frame coordinates of cropped and resized image
    # --- Convert 3D to pixel coords in original image ---
    f = 0.5 * img_height / np.tan(fov_y * np.pi / 360)  # focal length in pixels
    fy = f
    fx = f
    cx = img_width / 2
    cy = img_height / 2

    X, Y, Z = cam_pos_film
    u_orig = fx * (X / Z) + cx
    v_orig = fy * (Y / Z) + cy

    # -------------------------------------------------
    # Adjust for crop + resize
    # -------------------------------------------------
    if TARGET_IMG_SIZE is not None:
        H_target, W_target = TARGET_IMG_SIZE
    else:
        H_target, W_target = img_height, img_width

    if crop_params is not None:
        top, bottom, left, right = crop_params
        crop_w = img_width - left - right
        crop_h = img_height - top - bottom
        u_crop = u_orig - left
        v_crop = v_orig - top
    else:
        crop_w = img_width
        crop_h = img_height
        u_crop = u_orig
        v_crop = v_orig

    # Resize scale
    sx = W_target / crop_w
    sy = H_target / crop_h
    u_new = u_crop * sx
    v_new = v_crop * sy

    # Back-project to 3D in cropped/resized film frame
    fx_new = fx * sx
    fy_new = fy * sy
    cx_new = W_target / 2
    cy_new = H_target / 2

    X_new = (u_new - cx_new) * Z / fx_new
    Y_new = (v_new - cy_new) * Z / fy_new
    cam_pos_film_resized = np.array([X_new, Y_new, Z])

    fy_new_degree = (np.arctan(0.5 * H_target / fy_new)) * (360 / np.pi)

    # -------------------------------------------------
    # Debug: plot on cropped/resized image
    # -------------------------------------------------
    if debug:
        cropped_img = img[top:top + crop_h, left:left + crop_w]
        resized_img = cv2.resize(cropped_img, (W_target, H_target))
        img = resized_img.copy() 
        
        # plot projection
        cv2.drawMarker(
            img,
            (int(u_new), int(v_new)),
            color=(255, 0, 0),  # Red in RGB
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=5,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        # Draw text
        cv2.putText(
            img,
            f"({int(u_new)}, {int(v_new)})",
            (0, 10), # upper-left corner of image
            cv2.FONT_HERSHEY_SIMPLEX,
            0.2,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"({X_new:.2f}, {Y_new:.2f}, {Z:.2f})",
            (0, 30), # slightly below the previous text
            cv2.FONT_HERSHEY_SIMPLEX,
            0.2,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        
        
        img = cv2.arrowedLine(img, (int(cx_new), int(cy_new)), (int(cx_new) + 10, int(cy_new)), (255, 0, 0), 2, tipLength=0.3)
        img = cv2.arrowedLine(img, (int(cx_new), int(cy_new)), (int(cx_new), int(cy_new) + 10), (0, 255, 0), 2, tipLength=0.3)
        cv2.imwrite(f"tmp_img/debug_cropped_resized_camera_projection_{t}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        
    return cam_pos_film_resized, cam_rot_film.as_quat()

def normalize_angle(angle, tol=1e-1):
    """
    Normalize angle to (-π, π], where -π wraps to π
    """
    norm = (angle + np.pi) % (2 * np.pi) - np.pi
    if np.isclose(norm, -np.pi, atol=tol):
        norm = np.pi
    return norm

def euler_to_axis_angle(aa):
    roll, pitch, yaw = aa[0], aa[1], aa[2]
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Rotation matrix R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    # Compute angle
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    if np.isclose(angle, 0):
        # No rotation
        return np.zeros(3)
    else:
        # Axis computation from skew-symmetric part
        rx = R[2,1] - R[1,2]
        ry = R[0,2] - R[2,0]
        rz = R[1,0] - R[0,1]
        axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        axis_angle = axis * angle
        return axis_angle

def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )

def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q

def mat2euler(rmat, axes="sxyz"):
    """
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (np.array): 3x3 rotation matrix
        axes (str): One of 24 axis sequences as string or encoded tuple (see top of this module)

    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.asarray(rmat, dtype=np.float32)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return vec((ax, ay, az))

def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def euler2mat(euler):
    """
    Converts euler angles into rotation matrix form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """

    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat

