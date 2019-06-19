"""

This file contains the following functions:

- bland_altman_plot:

    Bland Altman plot for comparison between the mean thicknesses of the ilm-rpe layer computed
    once with an actual ring scan data and once with an extracted data set.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def ring_scan_plots(ring_scan_data_int, ilm_ring_scan_int, rpe_ring_scan_int, ring_scan_data, ilm_ring_scan,
                    rpe_ring_scan):

    noe = np.size(ring_scan_data_int, 1)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.plot(np.arange(0, noe), ilm_ring_scan_int, color='green', linewidth=5.0)
    ax1.plot(np.arange(0, noe), rpe_ring_scan_int, color='blue', linewidth=5.0)
    ax1.imshow(ring_scan_data_int, cmap='gray', vmin=0, vmax=255)
    ax1.title.set_text('Extracted ring scan and corresponding ilm and rpe segmentation')
    ax1.set_xlabel('A scan #')
    ax1.set_ylabel('interpolated grey value and segmentation')
    ax2.plot(np.arange(0, noe), ilm_ring_scan, color='green', linewidth=5.0)
    ax2.plot(np.arange(0, noe), rpe_ring_scan, color='blue', linewidth=5.0)
    ax2.imshow(ring_scan_data, cmap='gray', vmin=0, vmax=255)
    ax2.title.set_text('Spectralis ring scan and corresponding ilm and rpe segmentation')
    ax2.set_xlabel('A scan #')
    ax2.set_ylabel('grey value and segmentation')
    plt.show()

    return


def thickness_plot(diff_ilm_rpe_int, diff_mean_int, diff_ilm_rpe, diff_mean):

    noe = np.size(diff_ilm_rpe_int, 1)
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, noe), diff_ilm_rpe_int.flatten(), color='green', linewidth=2.0, label='interpolated')
    ax.plot(np.arange(0, noe), np.ones(noe) * diff_mean_int, 'g:', linewidth=2.0)
    ax.plot(np.arange(0, noe), diff_ilm_rpe.flatten(), color='blue', linewidth=2.0, label='ring scan')
    ax.plot(np.arange(0, noe), np.ones(noe) * diff_mean, 'b:', linewidth=2.0)
    ax.plot(np.array([noe - 5, noe - 5]), np.array([diff_mean_int, diff_mean_int + 0.0039]), color='red',
            linewidth=2.0, label='pixel size')
    plt.xlabel('A scan #')
    plt.ylabel('ilm/rpe layer thicknesses [mm]')
    ax.legend(loc='upper right')

    plt.show()

    return


def bland_altman_plot(measurement_1, measurement_2):

    """

    :param layer_thickness_ring_scan:
    :param layer_thickness_ring_scan_extracted:
    :return:
    """

    # check for nan in data
    nan_check = measurement_1 * measurement_2
    measurement_1 = measurement_1[np.isnan(nan_check).astype('int') == 0].reshape((1, -1))
    measurement_2 = measurement_2[np.isnan(nan_check).astype('int') == 0]\
        .reshape((1, -1))

    num = np.size(measurement_1, 1)

    diff = measurement_1 - measurement_2
    mean = (measurement_1 + measurement_2) / 2
    min_mean = np.min(mean)
    max_mean = np.max(mean)

    x = np.linspace(min_mean, max_mean, num)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    plt.figure()
    plt.plot(mean, diff, color='black', marker='o', markersize=3)
    plt.plot(x, np.ones(num) * mean_diff, 'black', linewidth=1)
    plt.plot(x, np.ones(num) * (mean_diff + 1.96 * std_diff), 'r--', linewidth=2)
    plt.plot(x, np.ones(num) * (mean_diff - 1.96 * std_diff), 'r--', linewidth=2)

    model = LinearRegression().fit(mean.reshape((-1, 1)), diff.reshape((-1, 1)))
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    y_predict = model.predict(x.reshape((-1, 1)))
    plt.plot(x, y_predict, 'b:', linewidth=2)

    plt.title('Bland-Altman-Plot')
    plt.xlabel('mean of thicknesses [mm]')
    plt.ylabel('difference of thicknesses [mm]')
    plt.show()

    return
