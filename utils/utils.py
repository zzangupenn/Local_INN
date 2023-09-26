import json
import numpy as np
import yaml
import cv2

BN_MOMENTUM = 0.1

def find_range(data):
    range_min = []
    range_max = []
    for k in range(data.shape[1]):
        range_min.append(np.min(data[:, k]))
        range_max.append(np.max(data[:, k]))
    return np.array([range_min, range_max])

def find_larger_range(range1, range2):
    range_ret = range1.copy()
    for k in range(range_ret.shape[1]):
        range_ret[0, k] = np.min([range1[0, k], range2[0, k]])
        range_ret[1, k] = np.max([range1[1, k], range2[1, k]])
    return range_ret

class open3dUtils:
    def __init__(self):
        from scipy.spatial.transform import Rotation
        import open3d as o3d
        self.Rotation = Rotation
        self.o3d = o3d
        self.object_list = []
        self.show_axis = True

    def create_camera_poses(self, extrinsic, size=1, color=[1, 0, 0], aspect_ratio=0.3, alpha=0.15, linewidths=0.1):
        focal_len_scaled = size
        vertex_std = np.array([[0, 0, 0, 1],
                                [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [0, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [0, 0, focal_len_scaled, 1],])
        T = np.eye(4)
        T[:3, :3] = self.Rotation.from_euler('Y', [180], degrees=True).as_matrix()
        for ind in range(vertex_std.shape[0]):
            vertex_std[ind] = T @ vertex_std[ind]
        vertex_transformed = vertex_std @ extrinsic.T
        
        points = vertex_transformed[:, :3]
        lines = [[0, 1], [0, 2], [1, 2], [2, 3], [0, 3], [0, 4], [3, 4], [1, 4], [5, 6], [0, 6]]
        colors = [color for i in range(len(lines))]
        line_set = self.o3d.geometry.LineSet()
        line_set.points = self.o3d.utility.Vector3dVector(points)
        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        line_set.colors = self.o3d.utility.Vector3dVector(colors)
        return line_set
    
    def add_object(self, object):
        self.object_list.append(object)
    
    def clear_object(self):
        self.object_list = []
        
    def show(self, background_color=np.asarray([0.0, 0.0, 0.0])):
        viewer = self.o3d.visualization.Visualizer()
        viewer.create_window()
        for geometry in self.object_list:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = self.show_axis
        opt.background_color = background_color
        viewer.run()
        viewer.destroy_window()
        del viewer
        del opt




def readTXT(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines

class ConfigJSON():
    def __init__(self) -> None:
        self.d = {}
    
    def load_file(self, filename):
        with open(filename, 'r') as f:
            self.d = json.load(f)
    
    def save_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.d, f, ensure_ascii=False, indent=4)
            
            
class DataProcessor():
    def __init__(self) -> None:
        pass
    
    def two_pi_warp(self, angles):
        twp_pi = 2 * np.pi
        return (angles + twp_pi) % (twp_pi)
        # twp_pi = 2 * np.pi
        # if angle > twp_pi:
        #     return angle - twp_pi
        # elif angle < 0:
        #     return angle + twp_pi
        # else:
        #     return angle
    
    def data_normalize(self, data):
        data_min = np.min(data)
        data = data - data_min
        data_max = np.max(data)
        data = data / data_max
        return data, [data_max, data_min]
    
    def runtime_normalize(self, data, params):
        return (data - params[1]) / params[0]
    
    def de_normalize(self, data, params):
        return data * params[0] + params[1]
    

class DrivableCritic():
    def __init__(self, yaml_dir, yaml_filename):
        with open(yaml_dir + yaml_filename) as f:
            map_yaml = yaml.load(f, Loader=yaml.FullLoader)
        # self.img = np.load(map_yaml['image'] + '.npy')
        self.img = cv2.imread(yaml_dir + map_yaml['image'], cv2.IMREAD_GRAYSCALE)
        self.img_ori = np.array(map_yaml['origin'])
        self.img_res = np.array(map_yaml['resolution'])
        self.x_in_m = self.img.shape[1] * self.img_res
        self.y_in_m = self.img.shape[0] * self.img_res
        
    def get_normalize_params(self):
        params = np.zeros((4, 2))
        params[0, 1] = self.img_ori[0]
        params[0, 0] = self.x_in_m
        params[1, 1] = self.img_ori[1]
        params[1, 0] = self.y_in_m
        params[2, 0] = np.pi * 2
        params[3, 0] = 30
        return params
        
    def pose_2_colrow(self, pose):
        colrow = (pose[:, :2] - self.img_ori[:2]) / self.img_res
        return np.int32(colrow)

    def normalized_pose_2_rowcol(self, normalized_pose):
        return np.int32([(1 - normalized_pose[:, 1]) * self.img.shape[0],
                  normalized_pose[:, 0] * self.img.shape[1]]).transpose(1, 0)

    def normalized_pose_find_drivable(self, normalized_pose):
        rowcol = self.normalized_pose_2_rowcol(normalized_pose)
        return self.img[rowcol[:, 0], rowcol[:, 1]]
    
    def normalized_pose_2_xy(self, normalized_pose):
        img_size_xy = np.array([self.img.shape[1], self.img.shape[0]])
        return (normalized_pose * img_size_xy) * self.img_res + self.img_ori[:2]


class plotlyUtils:
    def __init__(self, renderer = None) -> None:
        import plotly.graph_objects as go
        import plotly.io as pio
        self.pio = pio
        self.go = go
        if renderer is not None:
            self.pio.renderers.default = renderer
        
    def _extrinsic2pyramid_mesh(self, extrinsic, color='r', focal_len_scaled=1, aspect_ratio=1.3, alpha=0.15, linewidths=0.1):
        vertex_std = np.array([[0, 0, 0, 1],
                                [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                                [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        return vertex_transformed

    def _create_camera_pose(self, vertex, color=[1, 0, 0]):
        return self.go.Mesh3d(
            # 8 vertices of a cube
            x=vertex[:, 0],
            y=vertex[:, 1],
            z=vertex[:, 2],

            # i, j and k give the vertices of triangles
            i = [0, 0, 0, 0],
            j = [1, 2, 3, 4],
            k = [2, 3, 4, 1],
            showscale=True,
            opacity=0.5,
            facecolor=[list(color)] * 5
        )
        
    def create_point(self, xyz, color=[1, 0, 0]):
        return self.go.Scatter3d(
            x = xyz[0],
            y = xyz[1],
            z = xyz[2],
            opacity = 1,
            surfacecolor = color)

    def add_plot_data(self, data_list, data_in, color):
        for ind in range(len(data_in)):
            vertex = self._extrinsic2pyramid_mesh(data_in[ind], aspect_ratio=0.5)
            data_list.append(self._create_camera_pose(vertex, color=color))
        return data_list
    
    def plot3d_show(self, data_list, axis_limits):
        fig = self.go.Figure(data=data_list)
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            scene = dict(
                xaxis = dict(nticks=4, range=[axis_limits[0], axis_limits[1]],),
                yaxis = dict(nticks=4, range=[axis_limits[2], axis_limits[3]],),
                zaxis = dict(nticks=4, range=[axis_limits[4], axis_limits[5]],),
                aspectmode='data'),
            width = 1350
            )
        fig.show()