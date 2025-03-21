"""
This is a zkml tutorial using ezkl

Based on https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_public_network_output.ipynb

Steps are defined and are called by API endpoints.
"""

# make sure you have the dependencies required here already installed
import ezkl
from torch import nn
import ezkl
import os
import json
import torch


# Defines the model
# we got convs, we got relu, we got linear layers
# What else could one want ????
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=5, stride=2)

        self.relu = nn.ReLU()

        self.d1 = nn.Linear(48, 48)
        self.d2 = nn.Linear(48, 10)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim=1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = self.relu(x)

        # logits => 32x10
        logits = self.d2(x)

        return logits

# output file names
model_path = "data/model1/network.onnx"
compiled_model_path = "data/model1/network.compiled"
pk_path = "data/model1/test.pk"
vk_path = "data/model1/test.vk"
settings_path = "data/model1/settings.json"
witness_path = "data/model1/witness.json"
data_path = "data/model1/input.json"
cal_path = "data/model1/calibration.json"
proof_path = "data/model1/test.pf"


async def step0():
    circuit = MyModel()
    circuit


    shape = [1, 28, 28]
    x = 0.1 * torch.rand(1, *shape, requires_grad=True)

    # Flips the neural net into inference mode
    circuit.eval()

    # export to onnx (network.onnx)
    torch.onnx.export(
        circuit,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    # create a data file (input.json)
    data_array = ((x).detach().numpy()).reshape([-1]).tolist()
    data = dict(input_data=[data_array])
    json.dump(data, open(data_path, "w"))

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "private"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"  # private by default

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)

    assert res is True


    data_array = (
        (torch.rand(20, *shape, requires_grad=True).detach().numpy()).reshape([-1]).tolist()
    )
    data = dict(input_data=[data_array])
    json.dump(data, open(cal_path, "w"))

    await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")




async def step1():
    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)

    assert res is True

    # srs path
    res = await ezkl.get_srs(settings_path)
    return res


async def step2():
    # now generate the witness file

    res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)


async def step3():
    # HERE WE SETUP THE CIRCUIT PARAMS
    # WE GOT KEYS
    # WE GOT CIRCUIT PARAMETERS
    # EVERYTHING ANYONE HAS EVER NEEDED FOR ZK


    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
    )

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)    


async def step4():
    # GENERATE A PROOF

    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
    )

    assert os.path.isfile(proof_path)
    return res


async def step5():
    # VERIFY IT

    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
    )

    assert res is True
    return "verified"



async def steps_all():
    await step0()
    await step1()
    await step2()
    await step3()
    await step4()
    return await step5()
