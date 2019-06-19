import numpy as np
from os import listdir
from os.path import isfile, join
import datetime
from datetime import date
import ring_scan_from_volume
import OCT_read
from plots_ring_scan import ring_scan_plots, thickness_plot, bland_altman_plot

# parameters for ring scan interpolation & plotting_boolean
radius = 1.75
filter_parameter = int(1)
plotting_boolean = False


def full_id(file_name):
    """
    This function returns a row vector ['EYE_ID' 'ScanPos' 'year' 'month' 'day' ] for an inputted file

    :param file_name: file name of scan for which ID is wanted
    :return: see description
    :rtype: (1, 5) array
    """

    # Get the all information about the input vol file
    oct_info_id = OCT_read.OctRead(file_name)

    # Get the header of the input vol file
    header_id = oct_info_id.get_oct_hdr()

    # Scan position
    scan_pos = str(header_id['ScanPosition'])
    scan_pos_input = scan_pos[2:4]

    # EYE ID
    eye_id = int(str(header_id['PatientID'])[5:10])

    eye_id_scan_pos = np.vstack((eye_id, scan_pos_input)).T

    # Scan time
    loc_val = header_id['ExamTime'] / (1e7 * 60 * 60 * 24) + date.toordinal(date(1601, 1, 1))
    date_exam = datetime.datetime.fromordinal(int(loc_val))
    time_vector = np.array([date_exam.year, date_exam.month, date_exam.day]).reshape(1, -1)

    # creating id vector
    scan_full_id = np.concatenate((eye_id_scan_pos, time_vector), axis=1)

    return scan_full_id


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


# vol_files and file id for assignment between ring and volume scans
vol_path = 'data/vol/'
vol_files = [f for f in listdir(vol_path) if isfile(join(vol_path, f))]
full_id_vol_scan = np.array([full_id(vol_path + e) for e in vol_files]).reshape(-1, 5)

# bmo_files
bmo_path = 'data/bmo/'
bmo_files = [f for f in listdir(bmo_path) if isfile(join(bmo_path, f))]

# ring_scan_files file id for assignment between ring and volume scans
ring_scan_path = 'data/ring_scan/'
ring_scan_files = [f for f in listdir(ring_scan_path) if isfile(join(ring_scan_path, f))]
full_id_ring_scan = np.array([full_id(ring_scan_path + e) for e in ring_scan_files]).reshape(-1, 5)

# index array for assignment ring scan to volume scan
index_assign = np.zeros((len(ring_scan_files), 1)).astype('int')

# index array for assignment ring scan to volume scan with different recording years
warning_diff_years = np.zeros((len(ring_scan_files), 1)).astype('int')

# Check for equal data and BMO files
if len(vol_files) != len(bmo_files):
    print("Number of vol files and bmo files do not match!")
    num_files = 0
else:
    # Number of ring scan files
    num_files = len(ring_scan_files)

    # check for a corresponding vol file for each ring scan
    for i in range(num_files):
        check_identical = full_id_vol_scan == full_id_ring_scan[i, :]
        check_identical = np.prod(check_identical, axis=1).astype('bool')
        index_i = np.argwhere(check_identical)

        # if empty, because date is not equal, check for volume scan with closest recording year
        if index_i.size == 0:
            check_same_id = full_id_vol_scan[:, 0:2] == full_id_ring_scan[i, 0:2]
            check_same_id = np.prod(check_same_id, axis=1).astype('bool')
            diff_year_index_i = np.argwhere(check_same_id)

            # if empty, no volume scan with same eye id and scan position as ring scan exists
            if diff_year_index_i.size == 0:
                index_assign[i] = - int(1)
            else:
                warning_diff_years[i] = int(1)
                ring_scan_year = full_id_ring_scan[i, 2].astype('int')
                vol_scan_years = full_id_vol_scan[diff_year_index_i, 2].astype('int')
                diff_years = np.abs(vol_scan_years - ring_scan_year)
                index_i = diff_year_index_i[np.argmin(diff_years)]
                index_assign[i] = index_i[0]
        else:
            index_assign[i] = int(index_i[0])

# number of matching ring and volume scan data & initialization of scan number used for statistics
num_corresponding_files = len(np.delete(index_assign, np.argwhere(index_assign.flatten() == - int(1))))
number_statistics = 0

if len(vol_files) == len(bmo_files):
    print('Number of corresponding ring and volume scans:', num_corresponding_files)

# data nan arrays for Bland-Altman plots & enumerator for correct array filling
mean_layer_thickness_ring_scan = np.full(num_corresponding_files, np.nan).reshape((1, -1))
mean_layer_thickness_ring_scan_int = np.full(num_corresponding_files, np.nan).reshape((1, -1))
enumerator_layer_thickness = int(0)
layer_thickness_per_a_scan = np.full(1, np.nan).reshape((1, -1))
layer_thickness_per_a_scan_int = np.full(1, np.nan).reshape((1, -1))

# computation for all ring scan files
for i in range(num_files):

    # cases if assignment between ring scan and vol scan was possible
    if index_assign[i] == - int(1):
        print('\x1b[0;30;41m', "Ring scan", ring_scan_files[i], ": no matching volume scan data!", '\x1b[0m')
    else:
        if warning_diff_years[i] == int(0):
            print("Ring scan", ring_scan_files[i], ": found matching volume scan",
                  vol_files[index_assign[i][0]])
        else:

            print('\x1b[0;30;43m', "Ring scan", ring_scan_files[i], "(", full_id_ring_scan[i, 2], ")",
                  ": found matching volume scan", vol_files[index_assign[i][0]], "(", full_id_vol_scan[i, 2], ")",
                  " with different recording date !",
                  '\x1b[0m')

        file_ring_scan = ring_scan_files[i]
        file_vol = vol_files[int(index_assign[i])]
        file_bmo = bmo_files[int(index_assign[i])]

        # Get the all information about the input ring scan file
        oct_info_ring_scan = OCT_read.OctRead(ring_scan_path + file_ring_scan)

        # Get the header of the input ring scan file
        header_ring_scan = oct_info_ring_scan.get_oct_hdr()

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

        # Get all the information about the input vol file
        oct_info_vol = OCT_read.OctRead(vol_path + file_vol)

        # Get the header of the input vol file
        header_vol = oct_info_vol.get_oct_hdr()

        # Get the b scan stack of the input vol file
        b_scan_stack = oct_info_vol.get_b_scans(header_vol)

        # Get the segmentation data of the input vol file
        seg_data_full = oct_info_vol.get_segmentation(header_vol)

        # compute interpolated grey values, ilm and rpe segmentation
        ring_scan_interpolated = \
            ring_scan_from_volume.RingScanFromVolume(header_vol, b_scan_stack, seg_data_full, bmo_path + file_bmo,
                                                     radius, number_circle_points, filter_parameter)

        # compute correct circle points to corresponding scan pattern (OS vs OD)
        circle_points = ring_scan_interpolated.circle_points_coordinates()

        # compute interpolated grey values, ilm and rpe segmentation
        ring_scan_int, ilm_ring_scan_int, rpe_ring_scan_int, remove_boolean = \
            ring_scan_interpolated.ring_scan_interpolation(circle_points)

        # get layer thickness for actual ring scan
        diff_ilm_rpe, diff_mean, diff_std = thickness_ilm_rpe_layer(header_ring_scan, ilm_ring_scan,
                                                                    rpe_ring_scan)

        # get layer thickness for extracted ring scan
        diff_ilm_rpe_int, diff_mean_int, diff_std_int = thickness_ilm_rpe_layer(header_vol,
                                                                                ilm_ring_scan_int, rpe_ring_scan_int)

        # checks if comparison is added to statistics
        if remove_boolean is False:

            number_statistics = number_statistics + 1

            # add up A scan differences of actual ring scan for bland-altman-plot
            layer_thickness_per_a_scan = np.concatenate((layer_thickness_per_a_scan, diff_ilm_rpe), axis=1)

            # add up A scan differences of interpolation for bland-altman-plot
            layer_thickness_per_a_scan_int = np.concatenate((layer_thickness_per_a_scan_int, diff_ilm_rpe_int), axis=1)

            # store diff_mean of actual ring scan data for bland-altman plot
            mean_layer_thickness_ring_scan[0, enumerator_layer_thickness] = diff_mean

            # store diff_mean of interpolated data for bland-altman plot
            mean_layer_thickness_ring_scan_int[0, enumerator_layer_thickness] = diff_mean_int

        enumerator_layer_thickness = enumerator_layer_thickness + 1

        if plotting_boolean is True:
            # plot ring scan of interpolated grey values and corresponding ilm and rpe segmentation
            ring_scan_plots(ring_scan_int, ilm_ring_scan_int, rpe_ring_scan_int, ring_scan, ilm_ring_scan,
                            rpe_ring_scan)

            # plot thickness difference of actual and extracted ring scan
            thickness_plot(diff_ilm_rpe_int, diff_mean_int, diff_ilm_rpe, diff_mean)

if num_files != 0:

    print(number_statistics, "ring scans have been compared!")

    # plot bland-altman
    bland_altman_plot(mean_layer_thickness_ring_scan, mean_layer_thickness_ring_scan_int)

    # plot bland-altman
    plot_rnd = np.random.randint(0, np.size(layer_thickness_per_a_scan, 1), size=(1, 2000))
    plot_ind = np.unique(plot_rnd, axis=0).astype('int')
    bland_altman_plot(layer_thickness_per_a_scan[0, plot_ind], layer_thickness_per_a_scan_int[0, plot_ind])
