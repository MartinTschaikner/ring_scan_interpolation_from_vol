from os import listdir
import ring_scan_from_volume
import OCT_read
from plots_ring_scan import ring_scan_plots
from tkinter.filedialog import askopenfilename

# show an "Open" dialog box and return the path to the selected file
vol_filename = askopenfilename()
file_id = vol_filename.split('/')[-1].split('.')[0]

# parameters for ring scan interpolation & plotting_boolean
radius = 1.75
number_circle_points = 756
filter_parameter = int(2)
plotting_boolean = False
save_boolean = False

# bmo_files
bmo_path = 'data/bmo/'
bmo_file = [f for f in listdir(bmo_path) if f.find(file_id) != -1]

compute = True

if len(bmo_file) == 0:
    print("No matching bmo text file for selected volume scan!")
    compute = False

if compute:

    # Get all the information about the input vol file
    oct_info_vol = OCT_read.OctRead(vol_filename)

    # Get the header of the input vol file
    header_vol = oct_info_vol.get_oct_hdr()

    # Get the b scan stack of the input vol file
    b_scan_stack = oct_info_vol.get_b_scans(header_vol)

    # Get the segmentation data of the input vol file
    seg_data_full = oct_info_vol.get_segmentation(header_vol)

    # compute interpolated grey values, ilm and rpe segmentation
    ring_scan_interpolated = \
        ring_scan_from_volume.RingScanFromVolume(header_vol, b_scan_stack, seg_data_full, bmo_path + bmo_file[0],
                                                 radius, number_circle_points, filter_parameter)

    # compute correct circle points to corresponding scan pattern (OS vs OD)
    circle_points = ring_scan_interpolated.circle_points_coordinates()

    # compute interpolated grey values, ilm and rpe segmentation
    ring_scan_int, ilm_ring_scan_int, rpe_ring_scan_int, remove_boolean = \
        ring_scan_interpolated.ring_scan_interpolation(circle_points)

    # plot ring scan of interpolated grey values and corresponding ilm and rpe segmentation
    ring_scan_plots(ring_scan_int, ilm_ring_scan_int, rpe_ring_scan_int, ring_scan_int, ilm_ring_scan_int,
                    rpe_ring_scan_int)
