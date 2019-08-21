from unittest import TestCase
from ring_scan_from_volume import RingScanFromVolume
import ring_scan_from_volume
import OCT_read
import numpy as np


class TestRingScanFromVolume(TestCase):
    def setUp(self):
        oct_info = OCT_read.OctRead("test_EYE00113_239.vol")
        file_header = oct_info.get_oct_hdr()
        b_scan_stack = oct_info.get_b_scans(file_header)
        seg_data_full = oct_info.get_segmentation(file_header)
        file_name_bmo = "test_BMO.txt"
        radius = 1.5
        number_circle_points = 4
        filter_parameter = 1
        self.ring_scan_from_volume_analysis = RingScanFromVolume(file_header, b_scan_stack, seg_data_full,
                                                                 file_name_bmo, radius, number_circle_points,
                                                                 filter_parameter)


class TestInit(TestRingScanFromVolume):

    def test_initial_b_scans(self):
        self.assertEqual(self.ring_scan_from_volume_analysis.b_scan_stack.shape,
                         (self.ring_scan_from_volume_analysis.file_header['SizeZ'],
                          self.ring_scan_from_volume_analysis.file_header['SizeX'],
                          self.ring_scan_from_volume_analysis.file_header['NumBScans']),
                         "size matches")

    def test_initial_segmentation(self):
        self.assertEqual(self.ring_scan_from_volume_analysis.seg_data_full['SegLayers'][:, 0, :].shape,
                         (self.ring_scan_from_volume_analysis.file_header['SizeX'],
                          self.ring_scan_from_volume_analysis.file_header['NumBScans']), "size matches")
        self.assertEqual(self.ring_scan_from_volume_analysis.seg_data_full['SegLayers'][:, 1, :].shape,
                         (self.ring_scan_from_volume_analysis.file_header['SizeX'],
                          self.ring_scan_from_volume_analysis.file_header['NumBScans']), "size matches")

    def test_initial_header(self):
        self.assertNotEqual(self.ring_scan_from_volume_analysis.file_header['ScaleX'],
                            self.ring_scan_from_volume_analysis.file_header['ScaleZ'],
                            "Scaling unequal in different directions")
        self.assertNotEqual(self.ring_scan_from_volume_analysis.file_header['ScaleX'],
                            self.ring_scan_from_volume_analysis.file_header['Distance'],
                            "Scaling unequal in different directions")

    def test_initial_radius(self):
        self.assertEqual(self.ring_scan_from_volume_analysis.radius, 1.5)

    def test_initial_number_circle_points(self):
        self.assertEqual(self.ring_scan_from_volume_analysis.number_circle_points, 4)

    def test_initial_filter_parameter(self):
        self.assertEqual(self.ring_scan_from_volume_analysis.filter_parameter, 1)


class TestCirclePoints(TestRingScanFromVolume):
    def test_circle_points_coordinates(self):
        np.testing.assert_allclose(self.ring_scan_from_volume_analysis.circle_points_coordinates(),
                                   np.array([[3.5, 2, 0.5, 2], [2, 0.5, 2, 3.5]]), rtol=1e-1, atol=0.1)


class TestCriticalPoints(TestRingScanFromVolume):
    def test_critical_points(self):
        seg_data_test = np.array([1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(ring_scan_from_volume.critical_points(
            seg_data_test, 1, 1), np.array([[6], [7], [8], [9], [10]]), rtol=1e-5, atol=0)


class TestSmoothSegmentation(TestRingScanFromVolume):
    def test_smooth_segmentation(self):
        seg_data_test = np.array([1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(
            ring_scan_from_volume.smooth_segmentation(seg_data_test, 100)[0], seg_data_test, rtol=1e-1, atol=7)


class TestSegmentationCriticalSplit(TestRingScanFromVolume):
    def test_segmentation_critical_split(self):
        seg_data_test = np.array([1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1])
        critical_ind_test = ring_scan_from_volume.critical_points(seg_data_test, 0, 1)
        final_ind_test = ring_scan_from_volume.segmentation_critical_split(seg_data_test, critical_ind_test, 2)
        # print(critical_ind_test, '\n', final_ind_test)
        np.testing.assert_allclose(final_ind_test, np.array([[4, 5, 6, 7, 8, 9, 10, 11]]), rtol=1e-3, atol=0)


class TestGeometricMedian(TestRingScanFromVolume):
    def test_geometric_median(self):
        points_test_1 = np.array([[0, 0], [0, 2], [3, 0], [1, 1]])
        geometric_median_test_1 = ring_scan_from_volume.geometric_median(points_test_1, eps=1e-5)
        np.testing.assert_allclose(geometric_median_test_1, np.array([1, 1]), rtol=1e-3, atol=0)
        points_test_2 = np.array([[0, 1], [2, 7], [3, 0], [5, 6]])
        geometric_median_test_2 = ring_scan_from_volume.geometric_median(points_test_2, eps=1e-5)
        np.testing.assert_allclose(geometric_median_test_2, np.array([2.5, 3.5]), rtol=1e-3, atol=0)


class TestRingScanInterpolation(TestRingScanFromVolume):
    def test_ring_scan_interpolation(self):
        ring_scan_int, ilm_ring_scan_int, rpe_ring_scan_int, remove_boolean =\
            self.ring_scan_from_volume_analysis.ring_scan_interpolation(
                self.ring_scan_from_volume_analysis.circle_points_coordinates())
        self.assertEqual(ring_scan_int.shape,
                         (self.ring_scan_from_volume_analysis.file_header['SizeZ'],
                          self.ring_scan_from_volume_analysis.number_circle_points), "size matches")
        self.assertEqual(ilm_ring_scan_int.shape,
                         (self.ring_scan_from_volume_analysis.number_circle_points, ), "size matches")
        self.assertEqual(rpe_ring_scan_int.shape,
                         (self.ring_scan_from_volume_analysis.number_circle_points,), "size matches")
