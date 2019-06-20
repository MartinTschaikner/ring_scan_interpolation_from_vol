import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


class RingScanFromVolume:
    """
    This class computes interpolated ring scan data (image, ilm and rpe segmentation) from a given volume scan.
    """

    def __init__(self, file_header, b_scan_stack, seg_data_full, file_name_bmo, radius, number_circle_points,
                 filter_parameter):

        """
        declare the class variables

        :param file_header: header of vol file
        :param b_scan_stack: b scan data of vol file
        :param seg_data_full: segmentation data of vol file
        :param file_name_bmo: text file of bmo points of corresponding vol file
        :param radius: radius of ring scan
        :type radius: float (default: 1.75 mm)
        :param number_circle_points: number of equidistant circle points on ring scan
        :type number_circle_points: integer
        :param filter_parameter: weighting parameter for ring scan interpolation
        :type filter_parameter: integer
        """

        self.file_header = file_header
        self.b_scan_stack = b_scan_stack
        self.seg_data_full = seg_data_full
        self.file_name_bmo = file_name_bmo
        self.radius = radius
        self.number_circle_points = number_circle_points
        self.filter_parameter = filter_parameter

    def circle_points_coordinates(self):

        # import bmo points, scaling, computing of bmo center as mean of all points and projection on x,y plane
        bmo_data = pd.read_csv(self.file_name_bmo, sep=",", header=None)
        bmo_points = np.zeros([bmo_data.shape[1], bmo_data.shape[0]], dtype=float)
        bmo_points[:, 0] = bmo_data.iloc[0, :] * self.file_header['Distance']
        bmo_points[:, 1] = bmo_data.iloc[1, :] * self.file_header['ScaleX']
        bmo_points[:, 2] = bmo_data.iloc[2, :] * self.file_header['ScaleZ']
        bmo_center_3d = np.mean(bmo_points, axis=0)
        bmo_center_2d = np.array([bmo_center_3d[0], bmo_center_3d[1]])

        # compute noe equidistant circle points around bmo center with given radius
        noe = self.number_circle_points

        # Scan Position
        scan_pos = str(self.file_header['ScanPosition'])
        scan_pos_input = scan_pos[2:4]

        # OD clock wise, OS ccw ring scan interpolation
        if scan_pos_input == 'OS':
            phi = np.linspace(0, - 2 * np.pi, num=noe, endpoint=False)
        else:
            phi = np.linspace(0, 2 * np.pi, num=noe, endpoint=False) - np.pi

        center = np.linspace(bmo_center_2d, bmo_center_2d, num=noe).T
        circle_points_coordinates = center + self.radius * np.array((np.cos(phi), np.sin(phi)))

        return circle_points_coordinates

    def ring_scan_interpolation(self, circle_points_coordinates):

        noe = self.number_circle_points

        # create data arrays for interpolation
        ring_scan_data = np.zeros([self.file_header['SizeZ'], noe])
        ilm_ring_scan = np.zeros(noe)
        rpe_ring_scan = np.zeros(noe)

        # filter size
        f_x = int(2 * self.filter_parameter + 1)
        f_y = int(6 * self.filter_parameter + 1)

        # sigma^2
        if int(self.filter_parameter) != 0:
            sigma2 = np.square(2 / 3 * self.filter_parameter * self.file_header['ScaleX'])
        else:
            sigma2 = np.square(0.1 * self.file_header['ScaleX'])

        # loop over all circle points
        for i in range(noe):
            # reshape and calculating indices of nearest data grid point to ith circle point
            loc_var = np.reshape(circle_points_coordinates[:, i], [2, 1])
            index_x_0 = np.around(loc_var[0] / self.file_header['Distance']).astype(int)
            index_y_0 = np.around(loc_var[1] / self.file_header['ScaleX']).astype(int)

            # compute range of indices for filter mask
            index_x_min = int(index_x_0 - (f_x - 1) / 2)
            index_x_max = int(index_x_0 + (f_x - 1) / 2)
            index_y_min = int(index_y_0 - (f_y - 1) / 2)
            index_y_max = int(index_y_0 + (f_y - 1) / 2)

            # fill filter mask with corresponding indices
            index_xx, index_yy = np.meshgrid(np.linspace(index_x_min, index_x_max, f_x),
                                             np.linspace(index_y_min, index_y_max, f_y))

            # compute matrix of indices differences in x, y direction
            diff_x = index_xx - loc_var[0] / self.file_header['Distance']
            diff_y = index_yy - loc_var[1] / self.file_header['ScaleX']

            # compute weights and interpolated grey values
            w = np.exp(-(np.square(diff_x * self.file_header['Distance']) +
                         np.square(diff_y * self.file_header['ScaleX'])) / (2 * sigma2))
            gv = self.b_scan_stack[:, index_y_min:index_y_max + 1, index_x_min:index_x_max + 1]
            gv_w = np.sum(np.sum(w * gv, axis=1), axis=1) / np.sum(w)
            gv_w[gv_w >= 0.260e3] = 0

            # fill ring scan data array
            ring_scan_data[:, i] = gv_w

            # repeat for ilm data array
            z_ilm = self.seg_data_full['SegLayers'][index_y_min:index_y_max + 1, 0, index_x_min:index_x_max + 1]

            # handle nan in ilm data with nan sum
            check = np.isnan(z_ilm).astype('int')
            if np.sum(check) != 0:
                # print("nan in ILM data for circle point #", i, "@ index center", index_x_0, index_y_0)
                ind = np.where(check == int(1))
                w[ind] = 0

            ilm_ring_scan[i] = np.nansum(w * z_ilm) / np.sum(w)

            # repeat for rpe data array
            z_rpe = self.seg_data_full['SegLayers'][index_y_min:index_y_max + 1, 1, index_x_min:index_x_max + 1]

            # handle nan in rpe data with nan sum
            check = np.isnan(z_rpe).astype('int')
            if np.sum(check) != 0:
                if np.sum(check) != int(f_x * f_y):
                    # print("nan in RPE data for circle point #", i, "@ index center", index_x_0, index_y_0)
                    ind = np.where(check == int(1))
                    w[ind] = 0
                    rpe_ring_scan[i] = np.nansum(w * z_rpe) / np.sum(w)
                else:
                    # linear extrapolation from nearest neighbor points if all matrix elements are nan
                    rpe_ring_scan[i] = 2 * rpe_ring_scan[i - 1] - rpe_ring_scan[i - 2]
            else:
                rpe_ring_scan[i] = np.sum(w * z_rpe) / np.sum(w)

        rpe_ring_scan, remove_boolean_rpe = smooth_segmentation(rpe_ring_scan, 300)
        if remove_boolean_rpe is True:
            print('\x1b[0;30;41m', 'Smoothing failed (probably poor rpe segmentation of volume scan) : data '
                  'not used for Bland-Altman plot!', '\x1b[0m')

        ilm_ring_scan, remove_boolean_ilm = smooth_segmentation(ilm_ring_scan, 100)
        if remove_boolean_ilm is True:
            print('\x1b[0;30;41m', 'Smoothing failed (probably poor ilm segmentation of volume scan) : data '
                  'not used for Bland-Altman plot!', '\x1b[0m')

        remove_boolean = remove_boolean_rpe or remove_boolean_ilm

        return ring_scan_data, ilm_ring_scan, rpe_ring_scan, remove_boolean


def smooth_segmentation(segmentation_data, smoothing_factor):
    """
    This method checks the interpolated segmentation for critical points (gradient) and if there are any
    deletes them and fits a 3 deg order spline through the remaining segmentation points. The smoothed segmentation
    is then checked once more and if there still exist critical points, the segmentation smoothing has failed and
    the data will not be compared to an actual ring scan

    :param segmentation_data: interpolated ring scan segmentation data (ilm or rpe)
    :type segmentation_data: 1-dim float array
    :param smoothing_factor: smoothing factor of spline fitting
    :type: 1-dim float array
    :return: smoothed segmentation spline, boolean for smoothing ok/failed
    :rtype: 1-dim float array, boolean
    """

    noe = np.size(segmentation_data, 0)

    # line space for spline fitting
    x_spline = np.linspace(0, noe - 1, noe)

    # number of neighbors counted as critical points as well
    num_neighbor = 4

    # compute critical indices plus neighbours
    critical_indices = critical_points(segmentation_data, num_neighbor, 3.5)

    # minimal length of connected non critical part
    critical_length = 20

    # splits up the connected non critical parts of the segmentation and adds a part to critical if it´s length is
    # smaller than the critical length
    final_indices = segmentation_critical_split(segmentation_data, critical_indices, critical_length)

    # resulting line space for spline fit
    x_fit = np.delete(x_spline, final_indices).astype('int')

    # compute 3 deg order spline fit
    spl = UnivariateSpline(x_fit, segmentation_data[x_fit], s=int(noe))
    spl.set_smoothing_factor(smoothing_factor)
    y_spline = spl(x_spline)

    # plot spline fit and compare to original segmentation
    plot = False
    if plot:
        plt.figure()
        plt.plot(x_fit, segmentation_data[x_fit], 'ro', ms=4)
        if critical_indices.size != 0:
            plt.plot(critical_indices, segmentation_data[critical_indices], 'bx', ms=6)
            plt.plot(final_indices, segmentation_data[final_indices], 'b+', ms=6)
        plt.plot(x_spline, y_spline, 'g', lw=3)
        plt.plot(np.linspace(0, noe - 1, noe), segmentation_data, 'b', lw=1)
        plt.ylim([0, 300])
        plt.gca().invert_yaxis()
        plt.show()

    # checks for spikes in spline fit and if there are any, exclude from statistics.
    std_factor = 8
    spline_critical = critical_points(y_spline, 0, std_factor).astype('int')

    if np.size(spline_critical):
        remove_boolean = True
    else:
        remove_boolean = False

    return y_spline, remove_boolean


def critical_points(segmentation_data, num_neighbor, factor_std):
    """
    This static method computes critical points, defined as points which gradients are off the mean gradient (of all
    segmentation points) for more than the standard deviation times a chosen factor

    :param segmentation_data: interpolated ring scan segmentation data (ilm or rpe)
    :type segmentation_data: 1-dim float array
    :param num_neighbor: number of neighbors counted as critical points as well
    :type num_neighbor: integer
    :param factor_std: regulates when a segmentation data point is flagged as critical
    :type factor_std: float
    :return: indices of critical points
    :rtype: 1-dim integer array
    """

    # compute number of ilm/rpe data points and corresponding gradient, mean and std of grad
    grad_data = np.gradient(segmentation_data)
    mean_grad = np.mean(grad_data)
    std_grad = np.std(grad_data)

    # compute critical points for which the gradient is far off (peaks)
    critical_indices = np.argwhere((grad_data > mean_grad + factor_std * std_grad) |
                                   (grad_data < mean_grad - factor_std * std_grad))

    # compute neighbourhood of critical points to close gaps, e.g. at max/min
    if critical_indices.size != 0:
        neighbor = np.linspace(-num_neighbor, num_neighbor, 2 * num_neighbor + 1).reshape((1, -1))
        critical_neighborhood = (critical_indices + neighbor).reshape((-1, 1))
        critical_indices = np.unique(critical_neighborhood, axis=0).astype('int')
    else:
        critical_indices = np.array([[]])

    return critical_indices


def segmentation_critical_split(segmentation_data, critical_indices, critical_length):
    """
    This static method flags connected non critical points as critical, if the length is smaller than a chosen critical
    length.

    :param segmentation_data: interpolated ring scan segmentation data (ilm or rpe)
    :type segmentation_data: 1-dim float array
    :param critical_indices: indices of critical points from corresponding method
    :type critical_indices: 1-dim integer array
    :param critical_length: minimal number of connected non critical points to not be flagged as critical
    :type critical_length: integer
    :return: final indices of critical points
    :rtype: 1-dim integer array
    """

    # checks if there are any critical points at all
    if critical_indices.size != 0:
        noe = np.size(segmentation_data, 0)
        final_indices = critical_indices.reshape(1, -1)

        # group_indices for finding start and end indices of non critical point groups
        group_indices = np.zeros(noe)
        group_indices[critical_indices] = int(1)
        group_indices[0] = int(1)
        group_indices = np.diff(group_indices).astype('int')

        # compute start/end indices of non critical point groups
        start_ind = (np.argwhere(group_indices == - int(1)) + 1).reshape(1, -1)
        start_ind[0, 0] = 0
        end_ind = np.argwhere(group_indices == int(1)).reshape(1, -1)
        if np.size(start_ind, 1) != np.size(end_ind, 1):
            end_ind = np.concatenate((end_ind, np.array([[noe - 1]])), axis=1)

        # compute length of non critical point groups
        group_len = (end_ind - start_ind).flatten()

        # compute indices for which group lengths are bigger than critical length
        length_pass = np.argwhere(group_len > critical_length)

        # compute indices for non critical point groups smaller than the critical length
        start_ind_close = (np.delete(start_ind, length_pass))
        end_ind_close = np.delete(end_ind, length_pass)

        # flags points within those groups as critical points
        if start_ind_close.size != 0:
            for i in range(len(start_ind_close)):
                new_critical = np.arange(start_ind_close[i], end_ind_close[i] + 1).reshape(1, -1)
                final_indices = np.concatenate((final_indices.reshape(1, -1), new_critical), axis=1)
            final_indices = np.unique(final_indices, axis=0).astype('int')
    else:
        final_indices = critical_indices

    return final_indices
