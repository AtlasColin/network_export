from network_factory.layer_name import Conv, DepthSepConv, MaxPool, FC, AvgPool, SE_Module, Res_Add
from network_factory.network_arch import CONV_DEFS, FC_DEFS
import network_factory.layer_factory as layer_factory
import tensorflow as tf
slim = tf.contrib.slim


def network(inputs,
             final_endpoint='fc_3',
             min_depth=8,
             depth_multiplier=1.0,
             conv_defs=None,
             fc_defs=None,
             output_stride=None,
             scope=None):
    end_points = {}
    if conv_defs is None:
        conv_defs = CONV_DEFS
    if fc_defs is None:
        fc_defs = FC_DEFS
    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')
    with tf.variable_scope(scope, 'Network', [inputs]):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
            current_stride = 1
            # The atrous convolution rate parameter.
            rate = 1
            net = inputs
            conv_layer_num = 1
            fc_layer_num = 1
            maxpool_layer_num = 1
            depthsepconv_layer_num = 1
            avgpool_layer_num = 1
            SE_Module_layer_num = 1
            Res_Add_layer_num=1
            for i, conv_def in enumerate(conv_defs):
                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride
                if isinstance(conv_def, Conv):
                    end_point = 'Conv2d_%d' % conv_layer_num
                    depth = layer_factory.depth(conv_def.depth, depth_multiplier, min_depth)
                    net = slim.conv2d(net, depth, conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    conv_layer_num += 1
                elif isinstance(conv_def, SE_Module):
                    end_point = 'SE_Module_%d' % SE_Module_layer_num
                    net = layer_factory.se_module(net, scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    SE_Module_layer_num += 1
                elif isinstance(conv_def, Res_Add):
                    end_point = 'Res_Add_%d' % Res_Add_layer_num
                    net = layer_factory.res_add(end_points[conv_def.inputs1], end_points[conv_def.inputs2], scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    Res_Add_layer_num += 1
                elif isinstance(conv_def, DepthSepConv):
                    end_point = "DepthSepConv_%d" % depthsepconv_layer_num
                    net = layer_factory.depthsepconv(net, conv_def.depth, conv_def.kernel,
                                                     depth_multiplier=depth_multiplier,
                                                     min_depth=min_depth,
                                                     stride=layer_stride,
                                                     rate=layer_rate,
                                                     scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    depthsepconv_layer_num += 1
                elif isinstance(conv_def, MaxPool):
                    end_point = 'MaxPool_%d' % maxpool_layer_num
                    net = slim.max_pool2d(net,
                                          kernel_size=conv_def.kernel,
                                          stride=layer_stride,
                                          scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    maxpool_layer_num += 1
                elif isinstance(conv_def, AvgPool):
                    end_point = 'AvgPool_%d' % avgpool_layer_num
                    net = layer_factory.golbal_average_pooling(net, scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    avgpool_layer_num += 1
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
            for i, fc_def in enumerate(fc_defs):
                if isinstance(fc_def, FC):
                    end_point = 'fc_%d' % fc_layer_num
                    net = slim.fully_connected(net, fc_def.depth, scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    fc_layer_num += 1
                else:
                    raise ValueError('Unknown fc type %s for layer %d'
                                     % (fc_def.ltype, i))
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def network_arg_scope(weight_decay=0.0001):

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def main():
    with tf.Graph().as_default():
        input_x = tf.placeholder(dtype=tf.float32, shape=[8, 108, 120, 3])
        conv_defs = CONV_DEFS
        fc_defs = FC_DEFS
        min_depth = 16
        depth_multiplier = 0.50
        with slim.arg_scope(network_arg_scope()):
            net, end_points = network(inputs=input_x, conv_defs=conv_defs, fc_defs=fc_defs, min_depth=min_depth,
                                      final_endpoint="DepthSepConv_9", depth_multiplier=depth_multiplier)

            conv_layer_num = 1
            fc_layer_num = 1
            maxpool_layer_num = 1
            avgpool_layer_num = 1
            depthsepconv_layer_num = 1
            SE_Module_layer_num = 1
            Res_Add_layer_num = 1
            pre_channel = 3
            params = 0
            ops = 0
            for i, conv_def in enumerate(conv_defs):
                if isinstance(conv_def, Conv):
                    end_point = 'Conv2d_%d' % conv_layer_num
                    shape = end_points[end_point].get_shape().as_list()
                    depth = layer_factory.depth(conv_def.depth, depth_multiplier, min_depth=min_depth)
                    param = conv_def.kernel[0] * conv_def.kernel[1] * pre_channel * depth
                    op = conv_def.kernel[0] * conv_def.kernel[1] * shape[1]//conv_def.stride * shape[2]//conv_def.stride * pre_channel * depth
                    params += param
                    ops += op
                    print(end_points[end_point].op.name, " ", shape, "param: ", param/1000000, "op: ", op/1000000)
                    pre_channel = depth
                    conv_layer_num += 1
                elif isinstance(conv_def, DepthSepConv):
                    end_point = "DepthSepConv_%d" % depthsepconv_layer_num
                    shape = end_points[end_point].get_shape().as_list()
                    depth = layer_factory.depth(conv_def.depth, depth_multiplier, min_depth=min_depth)
                    param = conv_def.kernel[0] * conv_def.kernel[1] * pre_channel + \
                            1 * 1 * pre_channel * depth
                    op = conv_def.kernel[0] * conv_def.kernel[1] * pre_channel * shape[1]//conv_def.stride * shape[2]//conv_def.stride \
                         + 1 * 1 * pre_channel * shape[1] * shape[2] * depth
                    params += param
                    ops += op
                    print(end_points[end_point].op.name, " ", shape, "param: ", param/1000000, "op: ", op/1000000)
                    pre_channel = depth
                    depthsepconv_layer_num += 1
                elif isinstance(conv_def, SE_Module):
                    end_point = 'SE_Module_%d' % SE_Module_layer_num
                    shape = end_points[end_point].get_shape().as_list()
                    param = pre_channel * pre_channel / 8
                    op = pre_channel * pre_channel / 8 + pre_channel*pre_channel
                    params += param
                    ops += op
                    print(end_points[end_point].op.name, " ", shape, "param: ", param/1000000, "op: ", op/1000000)
                    SE_Module_layer_num += 1
                elif isinstance(conv_def, Res_Add):
                    end_point = 'Res_Add_%d' % Res_Add_layer_num
                    shape = end_points[end_point].get_shape().as_list()
                    print(end_points[end_point].op.name, " ", shape)
                    print("values: ", end_points[end_point].op.values())
                    Res_Add_layer_num += 1
                elif isinstance(conv_def, MaxPool):
                    end_point = 'MaxPool_%d' % maxpool_layer_num
                    shape = end_points[end_point].get_shape().as_list()
                    print(end_points[end_point].op.name, " ", shape)
                    maxpool_layer_num += 1
                elif isinstance(conv_def, AvgPool):
                    end_point = 'AvgPool_%d' % avgpool_layer_num
                    shape = end_points[end_point].get_shape().as_list()
                    print(end_points[end_point].op.name, " ", shape)
                    avgpool_layer_num += 1
            for i, fc_def in enumerate(fc_defs):
                if isinstance(fc_def, FC):
                    end_point = 'fc_%d' % fc_layer_num
                    shape = end_points[end_point].get_shape().as_list()
                    if fc_layer_num == 1:
                        param = shape[1] * shape[2] * pre_channel * fc_def.depth
                        op = shape[1] * shape[2] * pre_channel * fc_def.depth
                        params += param
                        ops += op
                    else:
                        param = pre_channel * fc_def.depth
                        op = pre_channel * fc_def.depth
                        params += param
                        ops += op
                    print(end_points[end_point].op.name, " ", shape, "param: ", param/1000000, "op: ", op/1000000)
                    pre_channel = fc_def.depth
                    fc_layer_num += 1
            print("Parameter numbers is {} millions".format(params/1000000))
            print("Operations numbers is {} millions".format(ops/1000000))


if __name__ == '__main__':
    main()
    # model_folder_path = r"D:\workspace\Visual_Studio\lijun\ssd_passenger_head_detect\ssd_train\model\darknet_depth_seqconv_05000"
    # freeze_graph(model_folder_path)
    # convert_trained_model_to_pb(model_folder_path)
    # print_graph("./frozen_model.pb")
