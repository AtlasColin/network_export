from collections import namedtuple

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
FC = namedtuple('FC', ['depth'])
MaxPool = namedtuple('MaxPool', ['kernel', 'stride'])
AvgPool = namedtuple('AvgPool', ['stride'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
SE_Module = namedtuple('SE_Module', ['stride'])
Res_Add = namedtuple('Res_Add', ['inputs1', 'inputs2', 'stride'])
