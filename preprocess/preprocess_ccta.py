import numpy as np
import SimpleITK as sitk
import copy
import os
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion, zoom, gaussian_filter

def houndsfield_to_attenuation(vol_vals, mu_water=0.1494*2.5e-2, mu_air=0.0430*2.5e-2):
    vol_vals = vol_vals / 1000
    vol_vals *= mu_water - mu_air
    vol_vals += mu_water

    return vol_vals

# goes from segmentation mask and raw volume to two separate volumes
def read_nrrd_to_itk(filename='0-lca.seg', path='D:/data/4DCT/MAGIX/MAGIX/segmentations'):
    im_itk = sitk.ReadImage(f'{path}/{filename}.nrrd')
    return im_itk

def preprocess_ccta():
    rawdata_path = 'D:/data/4DCT/MAGIX/MAGIX/rawdata'
    segmentation_path = 'D:/data/4DCT/MAGIX/MAGIX/segmentations'
    store_path = 'D:/data/4DCT/MAGIX/MAGIX/processed'

    raw_data_name = '10 Cir  CardiacCirc  3.0  B20f  0-90% RETARD_DECLECHEMENT'

    total_labels = {
        'heart': 51,
        'aorta': 52,
        'ribs': np.arange(79, 118),
    }

    volume_ids = np.arange(0, 10)
    for volume_id in volume_ids:
        print(f'preparing {volume_id}')

        id_store_path = f'{store_path}/{volume_id}'
        if not os.path.isdir(id_store_path):
            os.mkdir(id_store_path)

        vol_raw_data_name = f'{raw_data_name} {volume_id*10} %'
        vol_segmentation_name = f'{raw_data_name} {volume_id*10} % lca.seg'
        vol_total_segmentation_name = f'{raw_data_name} {volume_id*10} % total.seg'

        mask = read_nrrd_to_itk(vol_segmentation_name, segmentation_path)
        mask_np = sitk.GetArrayFromImage(mask)
        raw = read_nrrd_to_itk(vol_raw_data_name, rawdata_path)
        raw_np = sitk.GetArrayFromImage(raw)
        total = read_nrrd_to_itk(vol_total_segmentation_name, segmentation_path)
        total_np = sitk.GetArrayFromImage(total)

        sitk.WriteImage(total, f'{store_path}/{volume_id}/total-{volume_id}.vtk')

        # update houndsfield to attenuation values similarly to XCAT;
        raw_np = houndsfield_to_attenuation(raw_np)
        
        # adjust for the spacing
        spacing = np.array(mask.GetSpacing())[::-1] #* 2
        mask_np = zoom(mask_np, (spacing))
        raw_np = zoom(raw_np, (spacing))
        total_np = zoom(total_np, (spacing))
        spacing = (1,1,1)

        # remove the aorta intensity
        raw_np[total_np == total_labels['aorta']] = np.mean(raw_np[total_np == total_labels['heart']])
        
        # increase value of ribs slightly
        rib_factor = 1
        for rib_val in total_labels['ribs']:
            raw_np[total_np == rib_val] = raw_np[total_np == rib_val]*rib_factor

        raw_no_vessel_np = copy.deepcopy(raw_np)
        raw_no_vessel_np[mask_np > 0] = 0
        raw_no_vessel = sitk.GetImageFromArray(raw_no_vessel_np)
        raw.SetOrigin((0, 0, 0))
        raw.SetSpacing(spacing)
        sitk.WriteImage(raw_no_vessel, f'{store_path}/{volume_id}/raw-no-vessel-{volume_id}.vtk')

        # take raw values of the areas we annotated
        raw_vessel_np = np.zeros(raw_np.shape)
        raw_vessel_np[mask_np > 0] = raw_np[mask_np > 0]

        # absolute to make sure whole value is visible (minus values are air)
        raw_vessel_np = np.abs(raw_vessel_np)

        # transfer function to smoothen the values of the raw vessel
        max_vessel_val = np.max(raw_vessel_np[mask_np > 0])

        erosion = 3
        dilation = 1
        mask_np = binary_erosion(binary_dilation(mask_np, iterations=erosion).astype('int'), iterations=dilation).astype('int')
        dist_mask_np = distance_transform_edt(mask_np.astype('int'), sampling=np.array(spacing))

        # smooth dist mask
        gauss_sigma = 1
        gauss_rad = 2
        dist_mask_np = gaussian_filter(dist_mask_np, sigma=gauss_sigma, radius=gauss_rad)

        dist_mask_np_n = (dist_mask_np - np.min(dist_mask_np)) / (np.max(dist_mask_np) - np.min(dist_mask_np))

        # factor to multiply the function with
        # we want to mimick the XCAT phantom values
        contrast_f = 0.05#/max_vessel_val

        dist_mask_np_t = dist_mask_np_n * contrast_f
        dist_mask = sitk.GetImageFromArray(dist_mask_np_t)
        dist_mask.SetOrigin((0, 0, 0))
        dist_mask.SetSpacing(spacing)
        sitk.WriteImage(dist_mask, f'{store_path}/{volume_id}/dist-mask.vtk')
        
        xp = np.array([0, 1, 2, 4, 5])
        fp = np.array([0, 0.2, 0.5, 0.75, 1]) * contrast_f

        dist_mask_np = np.interp(dist_mask_np, xp, fp)

        dist_mask = sitk.GetImageFromArray(dist_mask_np)
        dist_mask.SetOrigin((0, 0, 0))
        dist_mask.SetSpacing(spacing)
        sitk.WriteImage(dist_mask, f'{store_path}/{volume_id}/dist-mask-tf.vtk')

        raw_vessel_np[mask_np > 0] = dist_mask_np[mask_np > 0]

        # maintain the origin and spacing to store it in the same manner
        raw_vessel = sitk.GetImageFromArray(raw_vessel_np)
        raw_vessel.SetOrigin((0, 0, 0))
        raw_vessel.SetSpacing(spacing)
        sitk.WriteImage(raw_vessel, f'{store_path}/{volume_id}/raw-vessel-{volume_id}.vtk')

        full_volume_np = np.zeros(raw_np.shape)
        full_volume_np[mask_np == 0] = raw_np[mask_np == 0]
        full_volume_np[mask_np > 0] = raw_vessel_np[mask_np > 0]

        full_volume = sitk.GetImageFromArray(full_volume_np)
        full_volume.SetOrigin((0, 0, 0))
        full_volume.SetSpacing(spacing)
        sitk.WriteImage(full_volume, f'{store_path}/{volume_id}/raw-processed-{volume_id}.vtk')

        spacing = np.array(spacing)
        with open(f"{store_path}/{volume_id}/spacing.npy", 'wb') as f:
            np.save(f, spacing)

        volume_shape = np.array(full_volume_np.shape)[::-1]
        print(volume_shape)
        with open(f"{store_path}/{volume_id}/volume-shape.npy", 'wb') as f:
            np.save(f, volume_shape)

        with open(f"{store_path}/{volume_id}/full_volume.npy", 'wb') as f:
            np.save(f, full_volume_np.flatten())

if __name__ == "__main__":
    preprocess_ccta()