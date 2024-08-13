# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import List, Tuple

import torch

from .preprocess_custom_ops import preprocess_op_lib  # noqa


class PreprocessTest(unittest.TestCase):

    def setUp(self):
        # pad
        self.pad_input = torch.ones(3, 200, 300)

        # tile_crop
        self.tile_size = 224

    def _compare_pad(self, image: torch.Tensor, padding: List[int]) -> None:
        output = torch.ops.preprocess.pad.default(image, padding)
        output_ref = torch.nn.functional.pad(image, padding)
        self.assertTrue(torch.allclose(output_ref, output, 1e-6))

    def _test_tile_crop(self, image: torch.Tensor, expected_shape: Tuple[int]) -> None:
        output = torch.ops.preprocess.tile_crop.default(image, self.tile_size)
        self.assertTrue(output.shape == expected_shape)

    def test_op_pad_without_padding(self):
        self._compare_pad(self.pad_input, [0, 0, 0, 0])

    def test_op_pad_with_right_bottom_padding(self):
        self._compare_pad(self.pad_input, [0, 124, 0, 148])

    def test_op_pad_with_right_padding(self):
        self._compare_pad(self.pad_input, [0, 124, 0, 0])

    def test_op_pad_with_bottom_padding(self):
        self._compare_pad(self.pad_input, [0, 0, 0, 148])

    def test_op_tile_crop_2x2(self):
        self._test_tile_crop(torch.ones(3, 448, 448), (4, 3, 224, 224))

    def test_op_tile_crop_1x3(self):
        self._test_tile_crop(torch.ones(3, 224, 672), (3, 3, 224, 224))

    def test_op_tile_crop_4x2(self):
        self._test_tile_crop(torch.ones(3, 896, 448), (8, 3, 224, 224))
