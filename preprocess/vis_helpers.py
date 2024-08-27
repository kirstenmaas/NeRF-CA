import numpy as np
import matplotlib.pyplot as plt
import torch

from tigre_helpers import get_ray_values_tigre

def visualize_geometry_tigre(train_view_points, grid, geo, depth_values, interpolator, device):
    fig = plt.figure(figsize=(8,8))
    
    sub_fig = fig.add_subplot(projection='3d')
    sub_fig.set_xlabel('X Label')
    sub_fig.set_ylabel('Y Label')
    sub_fig.set_zlabel('Z Labefl')

    ax_boundary = 15
    sub_fig.set_xlim3d(-ax_boundary, ax_boundary)
    sub_fig.set_ylim3d(-ax_boundary, ax_boundary)
    sub_fig.set_zlim3d(-ax_boundary, ax_boundary)

    colors = ['red', 'green', 'blue']
    x_pt = np.array([1, 0, 0, 1])
    y_pt = np.array([0, 1, 0, 1])
    z_pt = np.array([0, 0, 1, 1])
    pts = [x_pt, y_pt, z_pt]

    # plot world coordinate system
    for i, pt in enumerate(pts):
        points = np.array([[0,0,0], pt[:3]]).T
        sub_fig.plot(points[0, :], points[1, :], points[2, :], c=colors[i])

    img_width, img_height = geo.nDetector

    sub_fig = visualize_volume_bounds(sub_fig, grid.bounds)

    for j, viewpoint in enumerate(train_view_points):
        theta, phi = viewpoint

        ray_origins, ray_directions, src_matrix, ii, jj = get_ray_values_tigre(theta, phi, 0., geo, device)
    
        source_o = src_matrix.dot(np.array([0, 0, 0, 1]))

        source_x = src_matrix.dot(x_pt)
        source_y = src_matrix.dot(y_pt)
        source_z = src_matrix.dot(z_pt)
        source_pts = [source_x, source_y, source_z]

        # plot source coordinate system
        sub_fig.scatter(source_o[0], source_o[1], source_o[2], c='black')
        sub_fig.text(source_o[0], source_o[1], source_o[2], f'{theta}-{phi}', size=20, zorder=1, color='k')
        
        for i, pt in enumerate(source_pts):
            points = np.array([source_o[:3], pt[:3]]).T
            sub_fig.plot(points[0, :], points[1, :], points[2, :], c=colors[i])
        
        detector_o = ray_origins[0,0] + ray_directions[0,0] * geo.DSD
        detector_y = ray_origins[0,0] + ray_directions[0,img_height-1] * geo.DSD
        detector_x = ray_origins[0,0] + ray_directions[img_width-1,0] * geo.DSD
        detector_pts = [detector_x, detector_y]
        for i, pt in enumerate(detector_pts):
            points = np.array([detector_o[:3].cpu().numpy(), pt[:3].cpu().numpy()]).T
            sub_fig.plot(points[0, :], points[1, :], points[2, :], c=colors[i])
        
        visualize_query_points(sub_fig, img_width, img_height, ray_origins, ray_directions, depth_values, grid_scaling_factor=1)

        if j == 0:
            
            query_points = (ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]).cpu().numpy()
            query_points = query_points.reshape((-1, 3))
            query_scalars = interpolator(query_points)

            # visualize the occl volume from this perspective
            volume_scalar_ids = np.argwhere(query_scalars > 0)
            for k in range(0, volume_scalar_ids.shape[0], int(volume_scalar_ids.shape[0]*0.01)):
                vol_id = volume_scalar_ids[k]
                vol_query_point = query_points[vol_id].reshape(3)
                sub_fig.scatter(vol_query_point[0], vol_query_point[1], vol_query_point[2], color='grey')

            # visualize the artery in 3D
            artery_scalar_ids = np.argwhere(query_scalars >= 0.05)
            for k in range(0, artery_scalar_ids.shape[0], 25):
                artery_id = artery_scalar_ids[k]
                artery_query_point = query_points[artery_id].reshape(3)
                sub_fig.scatter(artery_query_point[0], artery_query_point[1], artery_query_point[2], color='purple')

    plt.show()


def visualize_volume_bounds(sub_fig, grid_bounds):
    # visualize volume
    x_bounds = grid_bounds[0:2]
    y_bounds = grid_bounds[2:4]
    z_bounds = grid_bounds[4:6]
    for x in x_bounds:
        for y in y_bounds:
            for z in z_bounds:
                sub_fig.scatter(x, y, z, color='black')
    return sub_fig

def visualize_query_points(sub_fig, img_width, img_height, ray_origins, ray_directions, depth_values, grid_scaling_factor=1):
    # defines a threshold based on the total distance, e.g. 10% of the ray start
    reg_perc = 0
    cum_dists = torch.cumsum(depth_values, dim=0)
    
    # reg perc defines the percentage of the ray that we use to calculate the occlusion from
    dists_range_front = reg_perc * cum_dists[-1]

    mask_front = torch.where(cum_dists < dists_range_front, 1., 0.).int()
    
    # last point in the ray that belongs to the occlusion ray
    last_ray_index = torch.argmin(mask_front) - 1

    x_pixs = [0, img_width//2-1, img_width-1]
    y_pixs = [0, img_height//2-1, img_height-1]
    for x in x_pixs:
        for y in y_pixs:
            query_points = (ray_origins[..., None, :][x, y] + ray_directions[..., None, :][x, y] * depth_values[..., :, None]).cpu().numpy()

            point1 = query_points[0]
            point2 = query_points[-1]
            points = np.array([point1, point2]).T

            # full ray
            sub_fig.plot(points[0, :], points[1, :], points[2, :], c='grey', alpha=0.5)

            # occl ray
            point3 = query_points[last_ray_index]
            points = np.array([point1, point3]).T
            sub_fig.plot(points[0, :], points[1, :], points[2, :], c='red')
    return sub_fig