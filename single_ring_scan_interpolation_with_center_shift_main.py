from os import listdir
import numpy as np
import ring_scan_from_volume
import OCT_read
from plots_ring_scan import ring_scan_plots, thickness_plot
from tkinter.filedialog import askopenfilename


def thickness_ilm_rpe_layer(header, ilm_data, rpe_data):

    """
    This function computes the differences between ilm and rpe segmentation, the mean and std of that differences

    :param header: header of the input file
    :param ilm_data: ilm data vector
    :type ilm_data: 1d float
    :param rpe_data: rpe data vector
    :type rpe_data: 1d float
    :return: see description
    :rtype: 1d float, 2 x float scalars
    """

    # compute difference of ilm and rpe, mean and std of that difference
    diff_data = ((rpe_data - ilm_data) * header['ScaleZ']).reshape((1, -1))
    mean_diff = np.nanmean(diff_data, axis=1)
    std_diff = np.nanstd(diff_data, axis=1)

    return diff_data, mean_diff, std_diff


# show an "Open" dialog box and return the path to the selected file
vol_filename = askopenfilename()
file_id = vol_filename.split('/')[-1].split('.')[0]

# parameters for ring scan interpolation & plotting_boolean
radius = 1.75
filter_parameter = int(1)
plotting_boolean = False
save_boolean = False

# bmo_files
bmo_path = 'data/bmo/'
bmo_file = [f for f in listdir(bmo_path) if f.find(file_id) != -1]

# possible ring_scan_files
ring_scan_path = 'data/ring_scan/'
ring_scan_file = [f for f in listdir(ring_scan_path) if f.find(file_id[0:-5]) != -1]

compute = True

if len(bmo_file) == 0:
    print("No matching bmo text file for selected volume scan!")
    compute = False

elif len(ring_scan_file) == 0:
    print("No matching ring scan file for selected volume scan!")
    compute = False

if compute:
    # Get all the information about the input vol file
    oct_info_vol = OCT_read.OctRead(vol_filename)

    # Get the header of the input vol file
    header_vol = oct_info_vol.get_oct_hdr()

    # Scan Position
    scan_pos = str(header_vol['ScanPosition'])
    scan_pos_input = scan_pos[2:4]

    # check for correct scan position OS/OD of ring scans
    for i in range(len(ring_scan_file)):
        # Get the all information about the input ring scan file
        oct_info_ring_scan = OCT_read.OctRead(ring_scan_path + ring_scan_file[i])

        # Get the header of the input ring scan file
        header_ring_scan = oct_info_ring_scan.get_oct_hdr()

        scan_pos = str(header_ring_scan['ScanPosition'])
        scan_pos_input_ring_scan = scan_pos[2:4]

        if scan_pos_input == scan_pos_input_ring_scan:

            # Get the b scan stack of the input ring scan file
            b_scan_stack = oct_info_ring_scan.get_b_scans(header_ring_scan)

            # Get the segmentation data of the input ring scan file
            seg_data_full = oct_info_ring_scan.get_segmentation(header_ring_scan)

            # Get needed data
            ring_scan = b_scan_stack.reshape(header_ring_scan['SizeZ'], header_ring_scan['SizeX'])
            ilm_ring_scan = seg_data_full['SegLayers'][:, 0, :]
            rpe_ring_scan = seg_data_full['SegLayers'][:, 1, :]
            # RNFL_ring_scan = seg_data_full['SegLayers'][:, 2, :]
            number_circle_points = header_ring_scan['SizeX']

            # get layer thickness for actual ring scan
            diff_ilm_rpe, diff_mean, diff_std = thickness_ilm_rpe_layer(header_ring_scan, ilm_ring_scan,
                                                                        rpe_ring_scan)

            # Get the b scan stack of the input vol file
            b_scan_stack = oct_info_vol.get_b_scans(header_vol)

            # Get the segmentation data of the input vol file
            seg_data_full = oct_info_vol.get_segmentation(header_vol)

            # compute interpolated grey values, ilm and rpe segmentation
            ring_scan_interpolated = \
                ring_scan_from_volume.RingScanFromVolume(header_vol, b_scan_stack, seg_data_full,
                                                         bmo_path + bmo_file[0],
                                                         radius, number_circle_points, filter_parameter)

            # compute correct circle points to corresponding scan pattern (OS vs OD)
            circle_points = ring_scan_interpolated.circle_points_coordinates()

            step_res = min(header_vol['Distance'], header_vol['ScaleX'])
            m = 9
            n = 1
            step_space = np.arange(-n*m, n*m + 1) / n * step_res
            x, y = np.meshgrid(step_space, step_space)

            center_shift = np.array([x.flatten(), y.flatten()]).T

            # best shift found so far:
            # center_shift = np.array([[-6.5 * step_res, 11.66 * step_res], [0, 0]])

            epsilon_min = 1
            true_center_shift = np.array([[0], [0]])
            k = 0

            for j in center_shift:
                print(k)
                k = k + 1

                circle_points_shift = circle_points + j.reshape(-1, 1)

                # compute interpolated grey values, ilm and rpe segmentation
                ring_scan_int, ilm_ring_scan_int, rpe_ring_scan_int, remove_boolean = \
                    ring_scan_interpolated.ring_scan_interpolation(circle_points_shift)

                # get layer thickness for extracted ring scan
                diff_ilm_rpe_int, diff_mean_int, diff_std_int = thickness_ilm_rpe_layer(header_vol,
                                                                                        ilm_ring_scan_int,
                                                                                        rpe_ring_scan_int)

                # search for smallest mean error in difference
                epsilon = np.nansum(np.abs(diff_ilm_rpe - diff_ilm_rpe_int)) / number_circle_points

                if epsilon < epsilon_min:
                    epsilon_min = epsilon
                    true_center_shift = j.reshape(-1, 1)

            print(true_center_shift)
            print(epsilon_min / header_vol['ScaleZ'])
            circle_points_true_shift = circle_points + true_center_shift.reshape(-1, 1)

            # compute interpolated grey values, ilm and rpe segmentation
            ring_scan_int, ilm_ring_scan_int, rpe_ring_scan_int, remove_boolean = \
                ring_scan_interpolated.ring_scan_interpolation(circle_points_true_shift)

            # get layer thickness for extracted ring scan
            diff_ilm_rpe_int, diff_mean_int, diff_std_int = thickness_ilm_rpe_layer(header_vol,
                                                                                    ilm_ring_scan_int,
                                                                                    rpe_ring_scan_int)
            # plot thickness difference of actual and extracted ring scan
            thickness_plot(diff_ilm_rpe_int, diff_mean_int, diff_ilm_rpe, diff_mean)

            # plot ring scan of interpolated grey values and corresponding ilm and rpe segmentation
            ring_scan_plots(ring_scan_int, ilm_ring_scan_int, rpe_ring_scan_int, ring_scan, ilm_ring_scan,
                            rpe_ring_scan)

        else:
            print('No matching ring scan file for selected volume scan!')
