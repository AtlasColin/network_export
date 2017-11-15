import tensorflow as tf
import tensorflow.contrib.slim as slim


def batch_normalization(inputs, name, relu=True, is_training=False):
        """contribution by miraclebiu"""
        if relu:
            temp_layer = slim.batch_norm(inputs=inputs, scale=True, center=True,
                                         is_training=is_training, scope=name)
            return tf.nn.relu(temp_layer)
        else:
            return slim.batch_norm(inputs=inputs, scale=True, center=True,
                                   is_training=is_training, scope=name)


def tensor_negation(inputs, name):
    """ simply multiplies -1 to the tensor"""
    return tf.multiply(inputs, -1.0, name=name)


def scale(inputs, num_input, name):
    with tf.variable_scope(name) as scope:
        alpha = tf.get_variable('alpha', shape=[num_input, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0), trainable=True,
                                regularizer=l2_regularizer(0.00001))
        beta = tf.get_variable('beta', shape=[num_input, ], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), trainable=True,
                               regularizer=l2_regularizer(0.00001))
        return tf.add(tf.multiply(inputs, alpha), beta)


def l2_regularizer(weight_decay=0.0005, scope=None):
    def regularizer(tensor):
        with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
            l2_weight = tf.convert_to_tensor(weight_decay,
                                             dtype=tensor.dtype.base_dtype,
                                             name='weight_decay')
            return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
    return regularizer


def bn_scale_combo(input, c_in, name, relu=True):
    """ PVA net BN -> Scale -> Relu"""
    with tf.variable_scope(name) as scope:
        bn = batch_normalization(input, name='bn', relu=False, is_training=False)
        alpha = tf.get_variable('bn_scale/alpha', shape=[c_in, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0), trainable=True,
                                regularizer=l2_regularizer(0.00001))
        beta = tf.get_variable('bn_scale/beta', shape=[c_in, ], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), trainable=True,
                               regularizer=l2_regularizer(0.00001))
        bn = tf.add(tf.multiply(bn, alpha), beta)
        if relu:
            bn = tf.nn.relu(bn, name='relu')
        return bn


def pva_negation_block(inputs, num_output, kernel_size, stride,
                              name, biased=True, trainable=True,
                              scale=True, negation=True):
    """ for PVA net, Conv -> BN -> Neg -> Concat -> Scale -> Relu"""
    with tf.variable_scope(name) as scope:
        conv = slim.conv2d(inputs=inputs,
                           num_outputs=num_output,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding='SAME',
                           trainable=trainable,
                           scope=name)
        conv = batch_normalization(conv, name='bn', relu=False, is_training=False)
        num_input = num_output
        if negation:
            conv_neg = tensor_negation(inputs=conv, name='neg')
            conv = tf.concat(axis=3, values=[conv, conv_neg], name='concat')
            num_input += num_input
        if scale:
            # y = \alpha * x + \beta
            alpha = tf.get_variable('scale/alpha', shape=[num_input, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=True,
                                    regularizer=l2_regularizer(0.00001))
            beta = tf.get_variable('scale/beta', shape=[num_input, ], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0), trainable=True,
                                   regularizer=l2_regularizer(0.00001))
            # conv = conv * alpha + beta
            conv = tf.add(tf.multiply(conv, alpha), beta)
    return tf.nn.relu(conv, name='relu')


def pva_negation_block_v2(inputs, num_output, kernel_size, stride,
                             num_input, name, biased=True,trainable=True,
                             scale = True, negation = True):
    """ for PVA net, BN -> [Neg -> Concat ->] Scale -> Relu -> Conv"""
    with tf.variable_scope(name) as scope:
        bn = batch_normalization(inputs=inputs, name='bn', relu=False, is_training=False)
        if negation:
            bn_neg = tensor_negation(bn, name='neg')
            bn = tf.concat(axis=3, values=[bn, bn_neg], name='concat')
            num_input += num_input
        if scale:
            # y = \alpha * x + \beta
            alpha = tf.get_variable('scale/alpha', shape=[num_input,], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=True,
                                    regularizer=l2_regularizer(0.00004))
            beta = tf.get_variable('scale/beta', shape=[num_input, ], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0), trainable=True,
                                   regularizer=l2_regularizer(0.00004))
            bn = tf.add(tf.multiply(bn, alpha), beta)
        bn = tf.nn.relu(bn, name='relu')
        conv = slim.conv2d(inputs=bn,
                           num_outputs=num_output,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding='SAME',
                           trainable=trainable,
                           scope=name)
        return conv


def mCReLu(inputs, num_input, kernel_sequence, scope, down_sample=False):
    if down_sample:
        stride = 2
    else:
        stride = 1
    net = pva_negation_block_v2(inputs=inputs, num_output=kernel_sequence[0],
                                kernel_size=[1, 1], stride=stride, num_input=num_input,
                                negation=False, name=scope+"/1")
    net = pva_negation_block_v2(inputs=net, num_output=kernel_sequence[1],
                                kernel_size=[3, 3], stride=1, num_input=kernel_sequence[0],
                                negation=False, name=scope+"/2")
    net = pva_negation_block_v2(inputs=net, num_output=kernel_sequence[2],
                                kernel_size=[1, 1], stride=1, num_input=kernel_sequence[1],
                                negation=True, name=scope+"/3")
    if num_input != kernel_sequence[2] or stride != 1:
        temp_conv = slim.conv2d(inputs=inputs, num_outputs=kernel_sequence[2],
                                kernel_size=[1, 1], stride=stride)
        temp_relu = tf.nn.relu(temp_conv)
        net = tf.add(temp_relu, net)
    else:
        net = tf.add(inputs, net)
    return net


def pva_inception_res_stack(inputs, c_in, name, block_start=False, type='a'):

    if type == 'a':
        (c_0, c_1, c_2, c_pool, c_out) = (64, 64, 24, 128, 256)
    elif type == 'b':
        (c_0, c_1, c_2, c_pool, c_out) = (64, 96, 32, 128, 384)
    else:
        raise ('Unexpected inception-res type')
    if block_start:
        stride = 2
    else:
        stride = 1
    with tf.variable_scope(name+'/incep') as scope:
        bn = batch_normalization(inputs=inputs, name='bn', relu=False, is_training=False)
        bn_scale = scale(bn, c_in, name='bn_scale')

        # 1 x 1
        conv = slim.conv2d(inputs=bn_scale, num_outputs=c_0, kernel_size=[1, 1],
                           stride=stride, scope="0/conv")
        conv_0 = bn_scale_combo(conv, c_in=c_0, name='0', relu=True)

        # 3 x 3
        bn_relu = tf.nn.relu(bn_scale, name='relu')
        if name == 'conv4_1':
            tmp_c = c_1
            c_1 = 48
        conv = slim.conv2d(inputs=bn_relu, num_outputs=c_1, kernel_size=[1, 1],
                           stride=stride, scope='1_reduce/conv')
        conv = bn_scale_combo(conv, c_in=c_1, name='1_reduce', relu=True)
        if name == 'conv4_1':
            c_1 = tmp_c
        conv = slim.conv2d(inputs=conv, num_outputs=c_1*2, kernel_size=[3, 3],
                           stride=1, scope='1_0/conv')
        conv_1 = bn_scale_combo(conv, c_in=c_1 * 2, name='1_0', relu=True)

        # 5 x 5
        conv = slim.conv2d(inputs=bn_scale, num_outputs=c_2, kernel_size=[1, 1],
                           stride=stride, scope='2_reduce/conv')
        conv = bn_scale_combo(conv, c_in=c_2, name='2_reduce', relu=True)
        conv = slim.conv2d(inputs=conv, num_outputs=c_2 * 2, kernel_size=[3, 3],
                           stride=1, scope='2_0/conv')
        conv = bn_scale_combo(conv, c_in=c_2 * 2, name='2_0', relu=True)
        conv = slim.conv2d(inputs=conv, num_outputs=c_2 * 2, kernel_size=[3, 3],
                           stride=1, scope='2_1/conv')
        conv_2 = bn_scale_combo(conv, c_in=c_2 * 2, name='2_1', relu=True)

        # pool
        if block_start:
            pool = slim.max_pool2d(inputs=bn_scale, kernel_size=[3, 3],
                                   stride=2, padding='SAME', scope='pool')
            pool = slim.conv2d(inputs=pool, num_outputs=c_pool, kernel_size=[1, 1],
                               stride=1, scope='poolproj/conv')
            pool = bn_scale_combo(pool, c_in=c_pool, name='poolproj', relu=True)

    with tf.variable_scope(name) as scope:
        if block_start:
            concat = tf.concat(axis=3, values=[conv_0, conv_1, conv_2, pool], name='concat')
            proj = slim.conv2d(inputs=inputs, num_outputs=c_out, kernel_size=[1, 1],
                               stride=2, scope='proj')
        else:
            concat = tf.concat(axis=3, values=[conv_0, conv_1, conv_2], name='concat')
            proj = inputs

        conv = slim.conv2d(inputs=concat, num_outputs=c_out, kernel_size=[1, 1],
                           stride=1, scope='out/conv')
        if name == 'conv5_4':
            conv = bn_scale_combo(conv, c_in=c_out, name='out', relu=False)
        conv = tf.add(conv, proj, name='sum')
        return conv


def pva_inception_res_block(inputs, name, name_prefix='conv4_', type='a'):
        """build inception block"""
        node = inputs
        if type == 'a':
            c_ins = (128, 256, 256, 256, 256, )
        else:
            c_ins = (256, 384, 384, 384, 384, )
        for i in range(1, 5):
            node = pva_inception_res_stack(inputs=node,
                                           c_in=c_ins[i-1],
                                           name=name_prefix + str(i),
                                           block_start=(i == 1),
                                           type=type)
        return node


def upconv(inputs, shape, stride=2, name='upconv', relu=True):
        """ up-conv"""
        with tf.variable_scope(name) as scope:
            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            weights = tf.get_variable(name="weights", shape=shape,
                                      initializer=tf.contrib.layers.xavier_initializer_conv2d())
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.00001, name="weight_decay_loss")
            tf.add_to_collection("weight_loss", weight_decay)
            input_shape = inputs.get_shape().as_list()
            assert len(input_shape) == 4
            output_shape = [input_shape[0], input_shape[1]*stride, input_shape[2]*stride, input_shape[3]]

            deconv = tf.nn.conv2d_transpose(value=inputs, filter=weights, output_shape=output_shape,
                                            strides=[1, stride, stride, 1], name=scope.name)
            if relu:
                return tf.nn.relu(deconv)
            return deconv


def block_inception_c(inputs, depth, scope=None, reuse=None):
    """Builds Inception-C block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    # 输出深度为4 * depth
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, depth, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, depth * 3 // 2, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_1, depth, [1, 3], scope='Conv2d_0b_1x3'),
                    slim.conv2d(branch_1, depth, [3, 1], scope='Conv2d_0c_3x1')])
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, depth * 3 // 2, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth * 7 // 4, [3, 1], scope='Conv2d_0b_3x1')
                branch_2 = slim.conv2d(branch_2, depth * 2, [1, 3], scope='Conv2d_0c_1x3')
                branch_2 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_2, depth, [1, 3], scope='Conv2d_0d_1x3'),
                    slim.conv2d(branch_2, depth, [3, 1], scope='Conv2d_0e_3x1')])
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth, [1, 1], scope='Conv2d_0b_1x1')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def nin_conv(inputs, kernel_size, stride, depth, scope, add_layer=False):
        with slim.arg_scope(list_ops_or_scope=[slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu):
            # Deeper Bottleneck Architectures
            net = slim.conv2d(inputs=inputs, num_outputs=depth, kernel_size=kernel_size,
                              stride=stride, scope=scope.join('_1'), padding='SAME')
            net = slim.conv2d(inputs=net, num_outputs=depth//2, kernel_size=[1, 1],
                              stride=stride, scope=scope.join('_2'), padding='SAME')
            net = slim.conv2d(inputs=net, num_outputs=depth, kernel_size=kernel_size,
                              stride=stride, scope=scope.join('_3'), padding='SAME')
            if add_layer:
                net = slim.conv2d(inputs=net, num_outputs=depth//2, kernel_size=[1, 1],
                                  stride=stride, scope=scope.join('_4'), padding='SAME')
                net = slim.conv2d(inputs=net, num_outputs=depth, kernel_size=kernel_size,
                                  stride=stride, scope=scope.join('_5'), padding='SAME')
        return net


# squeeze_net #
def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat(3, [e1x1, e3x3])


# net = fire_module(net, 16, 64, scope='fire1')
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
        return outputs
#


def depth(depth, depth_multiplier, min_depth=16):
        return max(int(depth * depth_multiplier), min_depth)


def depthsepconv(inputs, num_outputs, kernel_size, stride, depth_multiplier=1.0, min_depth=16, rate=1, scope=None):
    net = slim.separable_conv2d(inputs, None, kernel_size,
                                depth_multiplier=1,
                                stride=stride,
                                rate=rate,
                                normalizer_fn=slim.batch_norm,
                                padding='SAME',
                                scope=scope+'_depthwise')
    net = slim.conv2d(net, depth(num_outputs, depth_multiplier, min_depth), [1, 1],
                      stride=1,
                      normalizer_fn=slim.batch_norm,
                      padding='SAME',
                      scope=scope+'_pointwise')
    return net


def depthsepconv_v1(inputs, num_outputs, kernel_size, stride, depth_multiplier=1.0, rate=1, scope=None):
    # separable_conv2d计算量由kernel_size[0]*kernel_size[1]*width*height*num_input减少至
    # kernel_size[0]*width*height*num_input + kernel_size[1]*width*height*num_input，如3*3减少了1/3的计算量
    net = slim.separable_conv2d(inputs, None, [kernel_size[0], 1],
                                depth_multiplier=1,
                                stride=stride,
                                rate=rate,
                                normalizer_fn=slim.batch_norm,
                                padding='SAME',
                                scope=scope+'_depthwise_1')
    net = slim.separable_conv2d(net, None, [1, kernel_size[1]],
                                depth_multiplier=1,
                                stride=1,
                                rate=rate,
                                normalizer_fn=slim.batch_norm,
                                padding='SAME',
                                scope=scope+'_depthwise_2')
    # 计算量由width*height*num_input*num_output*depth_multiplier至
    # width*height*num_input*num_output*depth_multiplier + width*height*num_output*num_output*depth_multiplier
    # 而输出通道由 depth(num_outputs, depth_multiplier)至num_outputs
    net = slim.conv2d(net, depth(num_outputs, depth_multiplier), [1, 1],
                      stride=1,
                      normalizer_fn=slim.batch_norm,
                      padding='SAME',
                      scope=scope+'_pointwise')
    return net


def golbal_average_pooling(inputs,scope=None):
    """
    golbal average pooling
    :param x: [batch, height, width, channels]
    """
    shapes = inputs.get_shape().as_list()
    kernel_height = shapes[1]
    kernel_width = shapes[2]
    return slim.avg_pool2d(inputs, kernel_size=[kernel_height, kernel_width], stride=1, padding='VALID',
                           scope=scope)


def res_add(inputs_1, inputs_2, num_outputs=None, scope=None):
    shapes = inputs_1.get_shape().as_list()
    num_inputs_1 = shapes[3]
    shapes = inputs_2.get_shape().as_list()
    num_inputs_2 = shapes[3]
    if num_outputs is None:
        num_outputs = num_inputs_2
    if num_inputs_1 != num_outputs:
        inputs_1 = slim.conv2d(inputs=inputs_1, num_outputs=num_outputs, kernel_size=[1, 1],
                               stride=1, padding='SAME')
    if num_inputs_2 != num_outputs:
        inputs_2 = slim.conv2d(inputs=inputs_2, num_outputs=num_outputs, kernel_size=[1, 1],
                               stride=1, padding='SAME')
    return tf.add(inputs_1, inputs_2, scope)


def se_module(inputs, scope=None):
    shapes = inputs.get_shape().as_list()
    num_inputs = shapes[3]
    with tf.variable_scope(scope, 'se_module', [inputs]) as sc:
        global_avg = golbal_average_pooling(inputs)
        fc1 = slim.fully_connected(inputs=global_avg, num_outputs=num_inputs//16)
        fc2 = slim.fully_connected(inputs=fc1, num_outputs=num_inputs, activation_fn=tf.sigmoid)
        return tf.multiply(inputs, fc2)


def time_tensorflow_run(session, target, info_string, num_batches):
    import time
    import math
    # Args:
    # session:the TensorFlow session to run the computation under.
    # target:需要评测的运算算子。
    # info_string:测试名称。

    num_steps_burn_in = 10  # 先定义预热轮数（头几轮跌代有显存加载、cache命中等问题因此可以跳过，只考量10轮迭代之后的计算时间）
    total_duration = 0.0  # 记录总时间
    total_duration_squared = 0.0  # 总时间平方和  -----用来后面计算方差
    for i in range(num_batches + num_steps_burn_in):  # 迭代轮数
        start_time = time.time()  # 记录时间
        _ = session.run(target)  # 每次迭代通过session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('step %d, duration = %.3f' % (i - num_steps_burn_in, duration))
            total_duration += duration  # 累加便于后面计算每轮耗时的均值和标准差
            total_duration_squared += duration * duration
    mn = total_duration / num_batches  # 每轮迭代的平均耗时
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)  # 标准差
    print(' %s across %d steps, %.3f +/- %.3f sec / batch' % (info_string, num_batches, mn, sd))
