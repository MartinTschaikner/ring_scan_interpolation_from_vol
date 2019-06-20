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

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    noe = np.size(ring_scan_data_int, 1)

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.plot(np.arange(0, noe), ilm_ring_scan_int, color='green', linewidth=5.0)
    ax1.plot(np.arange(0, noe), rpe_ring_scan_int, color='blue', linewidth=5.0)
    ax1.imshow(ring_scan_data_int, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Interpolated ring scan', pad=22)
    ax1.title.set_size(25)
    ax1.set_xlabel('number of A scans [ ]', labelpad=18)
    ax1.set_ylabel('Z axis [ ]', labelpad=18)
    ax1.xaxis.label.set_size(20)
    ax1.yaxis.label.set_size(20)

    ax2.plot(np.arange(0, noe), ilm_ring_scan, color='green', linewidth=5.0)
    ax2.plot(np.arange(0, noe), rpe_ring_scan, color='blue', linewidth=5.0)
    ax2.imshow(ring_scan_data, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Spectralis ring scan', pad=22)
    ax2.title.set_size(25)
    ax2.set_xlabel('number of A scans [ ]', labelpad=18)
    ax2.xaxis.label.set_size(20)
    ax2.yaxis.set_major_locator(plt.NullLocator())
    plt.show()

    return


def thickness_plot(diff_ilm_rpe_int, diff_mean_int, diff_ilm_rpe, diff_mean):

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    noe = np.size(diff_ilm_rpe_int, 1)
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, noe), diff_ilm_rpe_int.flatten(), color='green', linewidth=2.0, label='Interpolated ring scan')
    ax.plot(np.arange(0, noe), np.ones(noe) * diff_mean_int, 'g:', linewidth=2.0, label='Mean of interpolated scan')
    ax.plot(np.arange(0, noe), diff_ilm_rpe.flatten(), color='blue', linewidth=2.0, label='Spectralis ring scan')
    ax.plot(np.arange(0, noe), np.ones(noe) * diff_mean, 'b:', linewidth=2.0, label='Mean of Spectralis scan')
    # ax.plot(np.array([noe - 5, noe - 5]), np.array([diff_mean_int, diff_mean_int + 0.0039]), color='red',
    #        linewidth=2.0)
    plt.xlabel('number of A scans [ ]', labelpad=18)
    plt.ylabel('layer thickness [mm]', labelpad=10)
    ax.legend(loc='upper right', fontsize=20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    plt.show()

    return


def bland_altman_plot(measurement_1, measurement_2, title, save, plot):

    """

    :param measurement_1:
    :param measurement_2:
    :param title:
    :param save:
    :param plot:
    :return:
    """

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # check for nan in data
    nan_check = measurement_1 * measurement_2
    measurement_1 = measurement_1[np.isnan(nan_check).astype('int') == 0].reshape((1, -1))
    measurement_2 = measurement_2[np.isnan(nan_check).astype('int') == 0]\
        .reshape((1, -1))

    num = np.size(measurement_1, 1)

    # compute statistics on differences
    diff = measurement_1 - measurement_2
    mean = (measurement_1 + measurement_2) / 2
    min_mean = np.min(mean)
    max_mean = np.max(mean)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    # create parameters for plot
    x = np.linspace(min_mean, max_mean, 2)

    my_dpi = 100
    height = 1080
    width = 1920
    plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
    plt.plot(mean, diff, color='black', marker='o', markersize=3)
    plt.plot(x, np.ones(2) * mean_diff, 'black', linewidth=1)
    plt.plot(x, np.ones(2) * (mean_diff + 1.96 * std_diff), 'r--', linewidth=2)
    plt.plot(x, np.ones(2) * (mean_diff - 1.96 * std_diff), 'r--', linewidth=2)

    # compute linear regression for slope
    model = LinearRegression().fit(mean.reshape((-1, 1)), diff.reshape((-1, 1)))
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    # create lin reg parameters & data for plot
    y_plot = model.coef_ * max_mean + model.intercept_
    lin_reg_str = 'SLOPE: ' + "{:.4f}".format(model.coef_[0, 0]) + '\n' +\
                  'INTERCEPT: ' + "{:.4f}".format(model.intercept_[0])
    if model.coef_[0, 0] > 0:
        str_pos = 'bottom'
    else:
        str_pos = 'top'

    y_predict = model.predict(x.reshape((-1, 1)))
    plt.plot(x, y_predict, 'b:', linewidth=2)

    # format plot
    plt.title('Bland-Altman plot for ' + str(num) + ' ' + title, fontsize=25, pad=18)
    plt.xlabel('mean of thicknesses [mm]', fontsize=20, labelpad=18)
    plt.ylabel('difference of thicknesses [mm]', fontsize=20, labelpad=10)
    plt.text(max_mean, mean_diff, 'MEAN: ' + "{:.4f}".format(mean_diff), fontsize=15, horizontalalignment='right',
             verticalalignment='bottom')
    plt.text(max_mean, mean_diff + 1.96 * std_diff, '+1.96 SD: ' + "{:.4f}".format(mean_diff + 1.96 * std_diff),
             fontsize=15, horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.text(max_mean, mean_diff - 1.96 * std_diff, '-1.96 SD: ' + "{:.4f}".format(mean_diff - 1.96 * std_diff),
             fontsize=15, horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.text(max_mean, y_plot, lin_reg_str, fontsize=15, horizontalalignment='right', verticalalignment=str_pos,
             color='blue')

    if save:
        plt.savefig('Bland_Altman/' + title + '.png', dpi=my_dpi)

    if plot:
        plt.show()

    return
