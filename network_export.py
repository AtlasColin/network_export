from network_factory.network_arch import CONV_DEFS, FC_DEFS
from network_factory.network import network, network_arg_scope
from tensorflow.python.framework import graph_util
from collections import OrderedDict
import tensorflow as tf
import os
slim = tf.contrib.slim


def get_operation_name_index(operation_names, node_name):
    try:
        return [operation_name in node_name.split("/") for operation_name in operation_names].index(True)
    except ValueError:
        return -1


def get_node_input(tf_nodes, node, exported_nodes_record_keys):
    result = list()
    for input_node_name in node.input:  # input is one of exported_nodes_record keys or is empty
        # print("input_node_name: ", input_node_name)
        if input_node_name in exported_nodes_record_keys:
            result += [input_node_name]
        else:
            index = list([node.name == input_node_name for node in tf_nodes]).index(True)
            result += get_node_input(tf_nodes, tf_nodes[index], exported_nodes_record_keys)
    result = sorted(set(result), key=result.index)
    return result


def get_exported_node_name_and_input_names(graph_def):
    """
    返回一个OrderedDict, key值是导出的节点的名称, value值是节点对应的输入节点的名称
    """
    # Reshape is for flatten operation
    # BiasAdd is for convolution and fully connected
    operation_names = ["Placeholder", "depthwise", "convolution", "BatchNorm",
                       "Sigmoid", "Mul", "Relu", "MaxPool", "AvgPool2D"]
    tf_nodes = graph_def.node
    exported_node_name_and_input_names = OrderedDict()
    for i in range(len(tf_nodes)):
        if "Res_Add" in tf_nodes[i].name:
            tf_nodes[i].name += "/Res_Add"
            exported_node_name_and_input_names[tf_nodes[i].name] = \
                get_node_input(tf_nodes, tf_nodes[i], exported_node_name_and_input_names.keys())
        index = get_operation_name_index(operation_names, tf_nodes[i].name)
        # print("{} tf_nodes[{}].name :{}".format(index, i, tf_nodes[i].name))
        next_index = get_operation_name_index(operation_names, tf_nodes[i+1].name) \
            if i < len(tf_nodes) - 1 else -1
        if index == -1 or index == next_index:
            continue
        exported_node_name_and_input_names[tf_nodes[i].name] = \
            get_node_input(tf_nodes, tf_nodes[i], exported_node_name_and_input_names.keys())
        if "fully_connected" and "BatchNorm" in tf_nodes[i].name:
            node_name = tf_nodes[i].name
            input_node_names = exported_node_name_and_input_names[tf_nodes[i].name]
            del exported_node_name_and_input_names[tf_nodes[i].name]
            exported_node_name_and_input_names[node_name.split("/")[0] + "/Res_Add"] = input_node_names
            exported_node_name_and_input_names[node_name] = [node_name.split("/")[0] + "/Res_Add"]
    return exported_node_name_and_input_names


def freeze_graph(model_folder, output_graph):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We retrieve the protobuf graph definition
    with tf.Graph().as_default() as graph:
        input_x = tf.placeholder(dtype=tf.float32, shape=[1, 320, 288, 3])
        conv_defs = CONV_DEFS
        fc_defs = FC_DEFS
        min_depth = 16
        depth_multiplier = 0.50
        with slim.arg_scope(network_arg_scope()):
            network(inputs=input_x, conv_defs=conv_defs, fc_defs=fc_defs, min_depth=min_depth,
                    final_endpoint="DepthSepConv_9", depth_multiplier=depth_multiplier)
        input_graph_def = graph.as_graph_def()
        # for node in input_graph_def.node:
        #     assert isinstance(node.name, str)
        #     print(node.name, node.input)
        last_node_name = input_graph_def.node[-1].name
        print("last node name: ", last_node_name)
        # We import the meta graph and retrive a Saver
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
        # saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, input_checkpoint)
            # We use a built-in TF helper to export variables to constant
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                [last_node_name],
                # output_node_names.split(",")  # We split on comma for convenience
            )
            exported_node_name_and_input_names = get_exported_node_name_and_input_names(output_graph_def)
            for item in exported_node_name_and_input_names.items():
                print(item)
            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))


def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def print_graph(frozen_model_filename, save_to_txt=False):
    graph = load_graph(frozen_model_filename)
    # 输入,输出结点也是operation,所以,我们可以得到operation的名字
    # for op in graph.get_operations():
    #     print(op.name, op.values())
    for node in graph.as_graph_def().node:
        print(node.name, node.input)
    # #  注意prefix/Placeholder仅仅是操作的名字,prefix/Placeholder:0才是tensor的名字
    # x = graph.get_tensor_by_name('prefix/Placeholder:0')
    # print("input_tensor", x)
    # y = graph.get_tensor_by_name('prefix/fully_connected_2/BiasAdd:0')
    if save_to_txt:
        pb2txt(frozen_model_filename)


def pb2txt(pb_file_path):
    if os.path.isfile(pb_file_path) is False:
        print("pb_file_path is not a file!")
        exit()
    pb_base_name = os.path.basename(pb_file_path)
    suffix = ".txt"
    log_file = open(pb_base_name + suffix, "w")
    graph = load_graph(pb_file_path)
    # for op in graph.get_operations():
    #     log_file.write("{}\n".format(op.name))
    for node in graph.as_graph_def().node:
        log_file.write("{} {}\n".format(node.name, node.input))


def main():
    model_folder_path = r"./models"
    output_graph = "./pb_folders/frozen_model.pb"
    freeze_graph(model_folder_path, output_graph)
    # convert_trained_model_to_pb(model_folder_path)
    # print_graph(output_graph, save_to_txt=True)
    pass


if __name__ == '__main__':
    main()
