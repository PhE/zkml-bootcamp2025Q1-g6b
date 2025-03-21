# make sure you have the dependencies required here already installed
import io
from pathlib import Path
import tempfile
from fastapi import Response, UploadFile
from torch import nn
import ezkl
import os
import torch
import json
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import PIL
from PIL import Image
from .. import tools

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X, y

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X, y
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# Defines the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


base_path = Path("data/fashion")

srs_path = "kzg.srs"
pk_path = "test.pk"
model_path2 = "model.ezlk"
proof_path = "proof.pf"

model_path = "network.onnx"
compiled_model_path = "network.compiled"
vk_path = "test.vk"
settings_path = "settings.json"
witness_path = "witness.json"
data_path = "input.json"


async def setup():
    # Train the model as you like here (skipped for brevity)
    model = NeuralNetwork()
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), model_path2)

    print(f"train_step(): Saved PyTorch Model State to {model_path2}")

    # TODO: shall run the train_step() here ??
    model = NeuralNetwork()

    model.eval()
    model.load_state_dict(torch.load(model_path2, weights_only=True))
    dummy_input = test_data[0][0]

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
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

    data_array = ((dummy_input).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_data=[data_array])

    # Serialize data into file:
    json.dump(data, open(data_path, "w"))

    print(f"export_step: done !")

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "private"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"  # private by default

    res = ezkl.gen_settings(
        model_path,
        settings_path,
        py_run_args=py_run_args,
    )

    assert res == True

    res = ezkl.compile_circuit(
        model_path,
        compiled_model_path,
        settings_path,
    )
    assert res == True

    # srs path
    res = await ezkl.get_srs(
        settings_path,
        srs_path=srs_path,
    )

    return {
        "model": f"/api/fashion/file/{model_path2}",
        "srs": f"/api/fashion/file/{srs_path}",
        "pk": f"/api/fashion/file/{pk_path}",
    }


async def prover_step(input_img: UploadFile):
    # TOOD: where to inject input ?? data_path ?

    # now generate the witness file
    res_prove = await ezkl.gen_witness(
        data_path,
        compiled_model_path,
        witness_path,
    )
    assert os.path.isfile(witness_path)

    # HERE WE SETUP THE CIRCUIT PARAMS
    # WE GOT KEYS
    # WE GOT CIRCUIT PARAMETERS
    # EVERYTHING ANYONE HAS EVER NEEDED FOR ZK

    res_setup = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path=srs_path,
    )

    assert res_setup == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # GENERATE A PROOF
    res_prove = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
        srs_path=srs_path,
    )

    # print(res)
    assert os.path.isfile(proof_path)
    # return {"filename": input_img.filename, "ezkl_prove": res_prove}
    # return tools.zipfile_stream("proof", "", [proof_path])

    # TODO: where is the output ??
    output = "Pullover"

    return {
        "output": output,
        "proof": f"/api/fashion/file/{proof_path}",
    }


async def verifier_step(
    proof: UploadFile,
):
    # VERIFY IT
    res = ezkl.verify(
        # TODO: use proof arg as proof_path
        proof_path,
        settings_path,
        vk_path,
        srs_path=srs_path,
    )

    assert res == True
    return {"status": "verified"}


def get_image(image_idx: int):
    image_matrix = test_data[image_idx][0][0]
    # image_matrix.shape

    numpy_array = image_matrix.numpy()
    numpy_array = (numpy_array * 255).astype("uint8")
    # Convert to PIL Image
    image = Image.fromarray(numpy_array, mode="L")  # 'L' mode for grayscale
    # Save as PNG
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        # image.save("output_grayscale.png")
        image.save(f.name)
        # raw = open("output_grayscale.png", "rb").read()
        raw = open(f.name, "rb").read()
        return Response(content=raw, media_type="image/png")
