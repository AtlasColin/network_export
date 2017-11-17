# -*- coding: utf-8 -*-


class CSeparableConvolutionParam(object):
    def __init__(self, kernel_size, stride_size, input_channel, output_channel, padding_type):
        assert len(kernel_size) == 2
        assert len(stride_size) == 2
        assert isinstance(padding_type, str) and padding_type in ["VALID", "SAME"]

        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.input_channel = input_channel
        self.output_channel = output_channel


# see TensorFlow source code:
# tensorflow/tensorflow/core/framework/common_shape_fnc.cc
def get_separable_convolution_output_size_and_pad(input_size, separable_convolution_params):
    assert isinstance(separable_convolution_params, CSeparableConvolutionParam)
    assert isinstance(input_size, tuple) and len(input_size) == 2

    kernel = separable_convolution_params.kernel_size
    stride = separable_convolution_params.stride_size
    padding_type = separable_convolution_params.padding_type
    if padding_type == "VALID":
        output_size = [(input_size[i] - kernel[i] + stride[i]) // stride[i]
                       for i in range(len(input_size))]
        padding = [0, 0, 0, 0]
    else:
        assert padding_type == "SAME"
        output_size = [(input_size[i] + stride[i] - 1) // stride[i]
                       for i in range(len(input_size))]
        padding_needed = [max(0, (output_size[i] - 1) * stride[i] + kernel[i] - input_size[i])
                          for i in range(len(input_size))]
        padding_top = padding_needed[0] // 2
        padding_bottom = padding_needed[0] - padding_top
        padding_left = padding_needed[1] // 2
        padding_right = padding_needed[1] - padding_left
        padding = [padding_top, padding_bottom, padding_left, padding_right]
    return output_size, padding


def get_input_to_padded_table(input_size, separable_convolution_params):
    assert isinstance(input_size, tuple) and len(input_size) == 2
    assert isinstance(separable_convolution_params, CSeparableConvolutionParam)

    output_size, padding = get_separable_convolution_output_size_and_pad(input_size, separable_convolution_params)
    assert len(padding) == 4 and all([e >= 0 for e in padding])

    input_height, input_width = input_size
    padded_height = input_height + padding[0] + padding[1]
    padded_width = input_width + + padding[2] + padding[3]
    padding_top = padding[0]
    padding_left = padding[2]
    input_channel = separable_convolution_params.input_channel
    padded_table = [-1 for _ in range(input_channel * padded_height * padded_width)]
    # input 包含于 padded
    for c in range(input_channel):  # padded 和 input 包含相同的 channel
        for y in range(input_height):
            for x in range(input_width):
                # (x + padding_left, y + padding_top) 对应于 input 在 padded 中的左上角起始位置
                padded_y = y + padding_top
                padded_x = x + padding_left
                padded_table[c * padded_height * padded_width + padded_y * padded_width + padded_x] = \
                    c * input_height * input_width + y * input_width + x
    return padded_table, (padded_height, padded_width), output_size


def get_padded_to_col_table(padded_size, separable_convolution_output_size, separable_convolution_params):
    assert isinstance(padded_size, tuple) and len(padded_size) == 2
    assert isinstance(separable_convolution_output_size, tuple) and len(separable_convolution_output_size) == 2
    assert isinstance(separable_convolution_params, CSeparableConvolutionParam)

    input_height, input_width = padded_size[0], padded_size[1]
    output_height, output_width = separable_convolution_output_size[0], separable_convolution_output_size[1]
    kernel_height, kernel_width = separable_convolution_params.kernel_size
    stride_height, stride_width = separable_convolution_params.stride_size
    # col_table的高宽分别为kernel_height * kernel_width 和 output_height * output_width
    col_table = [[-1 for _ in range(output_height * output_width)]
                 for _ in range(kernel_height * kernel_width)]
    for output_x in range(output_height * output_width):
        for output_y in range(kernel_width * kernel_height):
            input_y = output_x // output_width * stride_height + \
                output_y % (kernel_width * kernel_height) // kernel_width
            input_x = output_x % output_width * stride_width + \
                output_y % (kernel_width * kernel_height) % kernel_width
            input_index = input_y * input_width + input_x
            col_table[output_y][output_x] = input_index
    col_table_1d = [-1 for _ in range(kernel_height * kernel_width * output_height * output_width)]
    for y in range(kernel_height * kernel_width):
        for x in range(output_height * output_width):
            col_table_1d[y * output_height * output_width + x] = col_table[y][x]
    return col_table_1d


def get_input_to_col_table(input_size, separable_convolution_params):
    padded_table, padded_size, output_size = \
        get_input_to_padded_table(input_size, separable_convolution_params)
    padded_to_col_table = \
        get_padded_to_col_table(padded_size, tuple(output_size), separable_convolution_params)
    input_to_col_table = [-1 for _ in range(len(padded_to_col_table))]
    for i in range(len(input_to_col_table)):
        input_to_col_table[i] = padded_table[padded_to_col_table[i]]
    return input_to_col_table, output_size


def main():
    pass


if __name__ == "__main__":
    main()
