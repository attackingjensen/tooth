"""2026-10-07：无GT局部定框和双向坐标变换；邻牙辅助估计，支持用户校正。"""
import numpy as np


def vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(name+'必须为三个有限数值')
    return a


def unit(value):
    a = np.asarray(value, dtype=float)
    if np.linalg.norm(a) < 1e-8:
        raise ValueError('方向退化，请重新选点或手动校正')
    return a/np.linalg.norm(a)


def parameters(frame):
    center = vector(frame['center_world'], '中心')
    side = float(frame['cube_side_world'])
    rotation = np.asarray(frame.get('rotation_model_to_world', np.eye(3)), dtype=float)
    if not np.isfinite(side) or side <= 0:
        raise ValueError('定框尺度必须为正数')
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
        raise ValueError('定框旋转必须为右手正交矩阵')
    return center, side, rotation


def to_model(vertices, frame):
    center, side, rotation = parameters(frame)
    return (np.asarray(vertices)-center)@rotation/side


def to_world(vertices, frame):
    center, side, rotation = parameters(frame)
    return np.asarray(vertices)@rotation.T*side+center


def estimate(mesh, anchor, anchor_normal=None, neighbors=None, diameter=None, basis='occlusal'):
    anchor = vector(anchor, '基牙选点')
    centers = mesh.triangles_center
    distances = np.linalg.norm(centers-anchor, axis=1)
    nearest = int(np.argmin(distances))
    reference = unit(mesh.face_normals[nearest] if anchor_normal is None else vector(anchor_normal, '表面法向'))
    # 正向法向筛选可排除近距离对颌的反向表面；仍须在预览中确认。
    patch = (distances < 2.) & (mesh.face_normals@reference > .5)
    if patch.sum() < 3:
        raise ValueError('选点附近没有足够表面，请重新点选基牙顶面')
    normal = unit((mesh.face_normals[patch]*mesh.area_faces[patch, None]).sum(0))
    near = centers[distances < 8.]
    if len(near) < 20:
        raise ValueError('局部牙列不足以估计坐标系')
    eigenvalues, axes = np.linalg.eigh(np.cov(near.T))
    warnings = ['局部估计需要人工确认基牙、咬合方向、中心和尺度；没有逐牙分割或自动牙位识别']
    if abs(axes[:,1]@normal)<.7:
        warnings.append('咬合方向与局部PCA中间轴差异较大，历史训练轴约定可能不匹配，请检查方向')
    neighbor_points = [] if neighbors is None else [vector(p, '邻牙选点') for p in neighbors]
    if len(neighbor_points) not in (0, 2):
        raise ValueError('请选两颗相邻牙的咬合面，或清除邻牙选点')
    tangent = axes[:, 2]
    if neighbor_points:
        tangent = unit(neighbor_points[1]-neighbor_points[0])
        between = anchor-(neighbor_points[0]+neighbor_points[1])/2
        projected = between-tangent*np.dot(between, tangent)
        if np.linalg.norm(projected) > 2.:
            warnings.append('基牙点偏离两邻牙连线超过2mm，请检查邻牙选点')
    else:
        warnings.append('尚未提供邻牙选点，方向和尺寸使用局部表面启发式')
    y = normal.copy()
    if basis == 'pca_middle':
        y = axes[:, 1].copy()
        if y@normal < 0:y = -y
    elif basis != 'occlusal':
        raise ValueError('定框轴约定无效')
    x = tangent-y*np.dot(tangent, y)
    if np.linalg.norm(x) < 1e-5:
        x = axes[:, 0]-y*np.dot(axes[:, 0], y)
    x = unit(x)
    # 固定轴符号，只决定预览初值；用户可旋转/翻转，不自动搜索输出。
    if x[np.argmax(np.abs(x))] < 0:x = -x
    z = unit(np.cross(x, y));x = unit(np.cross(y, z))
    rotation = np.column_stack([x, y, z])
    if diameter is None:
        if neighbor_points:
            # 邻牙牙冠中心间距包含目标牙和两邻牙半宽，二分只提供尺度初值。
            diameter = np.linalg.norm(neighbor_points[1]-neighbor_points[0])/2
        else:
            local = (distances < 5.) & (mesh.face_normals@normal > .3)
            xy = (centers[local]-anchor)@np.column_stack([x, z])
            diameter = float(np.max(np.quantile(xy, .95, axis=0)-np.quantile(xy, .05, axis=0)))
            diameter = float(np.clip(diameter, 7., 11.))
    diameter = float(diameter)
    if not np.isfinite(diameter) or not 3. <= diameter <= 25.:
        raise ValueError('牙冠最长边初值应在3–25mm，请校正尺度')
    # 点击点在基牙表面，中心向内移；邻牙只用于估计，不读取任何GT。
    offset = min(1.5, diameter*.18)
    center = anchor-normal*offset
    return {'version': 2, 'center_world': center.tolist(), 'cube_side_world': diameter/.7,
            'rotation_model_to_world': rotation.tolist(), 'anchor_world': anchor.tolist(),
            'anchor_normal_world': normal.tolist(), 'neighbor_points_world': [p.tolist() for p in neighbor_points],
            'estimated_crown_longest_edge_mm': diameter, 'fill': .7, 'basis': basis,
            'local_pca_variance': eigenvalues.tolist(), 'local_pca_middle_abs_dot_occlusal': float(abs(axes[:, 1]@normal)),
            'center_offset_from_anchor_mm': -offset, 'warnings': warnings, 'gt_used': False,
            'normalization': 'x_model = R.T @ (x_world-c) / L; L=d/0.7',
            'localization': 'Explicit user abutment anchor; local surface/optional neighbor estimates, manual confirmation required'}
