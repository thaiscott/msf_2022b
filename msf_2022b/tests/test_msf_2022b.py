"""
Unit and regression test for the msf_2022b package.
"""

# Import package, test suite, and other packages as needed
from cmath import pi
import sys
from types import coroutine
from msf_2022b.functions import calculate_angle

import numpy as np
import pytest

import msf_2022b


@pytest.fixture
def methane_molecule():
    symbols = np.array(["C","H","H","H","H"])
    coordinates = np.array ([
       [1,1,1],
       [2.4,1,1],
       [-0.4,1,1],
       [1,1,2.4],
       [1,1,-0.4] 
    ])
    return symbols,coordinates

@pytest.mark.skip
def test_calculate_angles():
    """Test that the calculate_angle function returns what we expect"""
    r1 = np.array([0, 0, -1])
    r2 = np.array([0, 0, 0])
    r3 = np.array([1, 0, 0])

    expected_output = pi / 2

    observed_output = msf_2022b.calculate_angle(r1, r2, r3)

    assert expected_output == observed_output

    expected_output_degrees = 90.0

    observed_output_degrees = msf_2022b.calculate_angle(r1, r2, r3, degrees=True)

    assert expected_output_degrees == observed_output_degrees

@pytest.mark.parametrize("p1,p2,p3, expected_angle",[
    (np.array([0,0,-1]),np.array([0,0,0]),np.array([1,0,0]),90),
    (np.array([np.sqrt(2)/2,np.sqrt(2)/2,0]),np.array([0,0,0]),np.array([1,0,0]),45.0)
]
)
def test_calculate_angle_many(p1,p2,p3,expected_angle):

    calculated_angle = msf_2022b.calculate_angle(p1,p2,p3,degrees=True)
    assert expected_angle == calculated_angle


def test_build_bond_list_error(methane_molecule):

   # coordinates = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]])
    coordinates = methane_molecule[1]
    # msf_2022b.build_bond_list(coordinates,min_bond=-1)

    with pytest.raises(ValueError):
        msf_2022b.build_bond_list(coordinates, min_bond=-1)


def test_calculate_distances():
    """Test that the calculate_distance function calculated what we expect"""

    r1 = np.array([0, 0, 0])
    r2 = np.array([0, 1, 0])

    expected_output = 1.0

    observed_output = msf_2022b.calculate_distance(r1, r2)

    assert expected_output == observed_output


def test_msf_2022b_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "msf_2022b" in sys.modules
