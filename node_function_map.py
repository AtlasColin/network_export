# -*- coding: utf-8 -*-

import re
from placeholder_node import get_placeholder_node
from convolution_node import get_convolution_node
from relu_node import get_relu_node
from sigmoid_node import get_sigmoid_node
from scale_node import get_scale_node
from global_avg_pooling_node import get_global_avg_pooling_node
from batch_normalization_node import get_batch_normalization_node
from max_pooling_node import get_max_pooling_node
from flat_node import get_flat_node
from fully_connected_node import get_fully_connected_node
from res_add_node import get_res_add_node
from separable_convolution_node import get_depthwise_node


class CNodeFunctionMap(object):
    def __init__(self):
        self.function_map = {"Placeholder": get_placeholder_node,
                             ".*/depthwise": get_depthwise_node,
                             ".*/convolution": get_convolution_node,
                             ".*/BatchNorm.*": get_batch_normalization_node,
                             ".*/Relu": get_relu_node,
                             ".*/MaxPool": get_max_pooling_node,
                             ".*/Reshape": get_flat_node,
                             ".*/Tensordot": get_fully_connected_node,
                             ".*/Mul": get_scale_node,
                             ".*/AvgPool": get_global_avg_pooling_node,
                             ".*/Sigmoid": get_sigmoid_node,
                             ".*/Res_Add_\\d+": get_res_add_node
                             }

    def __getitem__(self, item):
        for key in self.function_map.keys():
            if re.compile(key).match(item):
                return self.function_map[key]
        raise ValueError("unsupported node")


def main():
    pass


if __name__ == "__main__":
    main()
