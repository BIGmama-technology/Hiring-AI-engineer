import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pytest
import numpy as np
from src.models.kernels import (
    SumKernel,
    ProductKernel,
    GaussianKernel,
    RBFKernel,
    RationalQuadraticKernel,
    ExponentiatedKernelSineKernel
)


# defining x1 and x2 that we are going to use in the tests using 'fixtures'
@pytest.fixture
def input_arrays():
    return np.array([1.0]), np.array([2.0])

@pytest.mark.parametrize("kernel_class, length_scale, expected_result",
                         [
                            (GaussianKernel, 1.0, np.exp(-0.5)),
                            (GaussianKernel, 2.0, np.exp(-0.125)),
                            (RBFKernel, 1.0, np.exp(-0.25)),
                            (RBFKernel, 2.0, np.exp(-0.0625)),
                            (RationalQuadraticKernel, 1.0, 2/3),
                            (RationalQuadraticKernel, 2.0, 8/9),
                            (ExponentiatedKernelSineKernel, 1.0, np.exp(-0.5)),
                            (ExponentiatedKernelSineKernel, 2.0, np.exp(-0.125)),
                            # we can add more test here
                         ]
                        )
# we will test the compute method for each kernel with the examples provided in this parametrize mark
# Note: only "length_scale" can be changed the others parameters like alpha took their default value
def test_multiple_kernel_compute(input_arrays, kernel_class, length_scale, expected_result):
    x1, x2 = input_arrays
    kernel_instance = kernel_class(length_scale=length_scale)
    result = kernel_instance.compute(x1, x2)
    assert np.isclose(result, expected_result)


@pytest.mark.parametrize("kernel_class_1, length_scale_1, kernel_class_2, length_scale_2, expected_result",
                         [
                             (GaussianKernel, 1.0, GaussianKernel, 2.0, np.exp(-0.5) + np.exp(-0.125)),
                             (RBFKernel, 2.0, RBFKernel, 1.0, np.exp(-0.0625) + np.exp(-0.25)),
                             (RationalQuadraticKernel, 1.0, RationalQuadraticKernel, 2.0, 14/9),
                             (ExponentiatedKernelSineKernel, 2.0, ExponentiatedKernelSineKernel, 1.0, np.exp(-0.125) + np.exp(-0.5)),
                             # we can add more test here
                         ]
                        )
# we will test the sumation of kernels with the examples provided in this parametrize mark
# Note: only "length_scale" can be changed the others parameters like alpha took their default value
def test_multiple_sum_kernel(input_arrays, kernel_class_1, length_scale_1, kernel_class_2, length_scale_2, expected_result):
    x1, x2 = input_arrays
    kernel_1 = kernel_class_1(length_scale = length_scale_1)
    kernel_2 = kernel_class_2(length_scale = length_scale_2)
    result = SumKernel(kernel_1, kernel_2).compute(x1, x2)
    assert np.isclose(result, expected_result)


@pytest.mark.parametrize("kernel_class_1, length_scale_1, kernel_class_2, length_scale_2, expected_result",
                         [
                             (GaussianKernel, 1.0, GaussianKernel, 2.0, np.exp(-5/8)),
                             (RBFKernel, 2.0, RBFKernel, 1.0, np.exp(-5/16)),
                             (RationalQuadraticKernel, 1.0, RationalQuadraticKernel, 2.0, 16/27),
                             (ExponentiatedKernelSineKernel, 2.0, ExponentiatedKernelSineKernel, 1.0, np.exp(-5/8)),
                             # we can add more test here
                         ]
                        )
# we will test the product of kernels with the examples provided in this parametrize mark
# Note: only "length_scale" can be changed the others parameters like alpha took their default value
def test_multiple_product_kernel(input_arrays, kernel_class_1, length_scale_1, kernel_class_2, length_scale_2, expected_result):
    x1, x2 = input_arrays
    kernel_1 = kernel_class_1(length_scale = length_scale_1)
    kernel_2 = kernel_class_2(length_scale = length_scale_2)
    result = ProductKernel(kernel_1, kernel_2).compute(x1, x2)
    assert np.isclose(result, expected_result)