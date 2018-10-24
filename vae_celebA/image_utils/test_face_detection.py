from quva_code.utils.face_detection import bb_overlap
import numpy as np


def test_bb_overlap():
    assert np.allclose(bb_overlap([0, 0, 4, 4], [[0, 0, 4, 4], [2, 2, 4, 4], [1, 1, 3, 3], [10, 10, 3, 3]]), [1., 4./(16+16-4), 9./16, 0])


if __name__ == '__main__':
    test_bb_overlap()