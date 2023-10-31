"""PyTesseract OCR plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import replicate
import requests

import fiftyone.operators as foo
from fiftyone.operators import types

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def get_min_dist(query, dataset):
    return dataset.sort_by_similarity(query, k=1, dist_field="dist").first().dist


def generate_filename(prompt):
    for pattern in [" ", ",", "'", ":", "?", "!"]:
        prompt = prompt.replace(pattern, "")
    return prompt.replace(" ", "_") + ".png"


def generate_sample_from_prompt(prompt, dataset, clip_model):
    input = {
        "width": 1024,
        "height": 1024,
        "prompt": f"A TOK emoji of {prompt}, white background",
        "refine": "expert_ensemble_refiner",
        "scheduler": "K_EULER",
        "lora_scale": 0.6,
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "apply_watermark": False,
        "high_noise_frac": 0.99,
        "negative_prompt": "",
        "prompt_strength": 0.8,
        "num_inference_steps": 50,
    }

    url = replicate.run(
        "fofr/sdxl-emoji:dee76b5afde21b0f01ed7925f0665b7e879c50ee718c5f78a9d38e04d523cc5e",
        input=input,
    )[0]

    content = requests.get(url).content

    filename = generate_filename(prompt)
    base_path = os.path.join(*dataset.first().filepath.split("/")[:-2])
    filepath = "/" + os.path.join(base_path, "generated_emojis", filename)
    with open(filepath, "wb") as f:
        f.write(content)

    sample = fo.Sample(
        filepath=filepath,
        name=prompt,
        text_embedding=clip_model.embed_prompt(prompt),
        original=False,
    )
    dataset.add_sample(sample)

    dataset.delete_brain_run("text_clip_umap")
    fob.compute_visualization(
        dataset, embeddings="text_embedding", method="umap", brain_key="text_clip_umap"
    )


class CreateEmoji(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="create_emoji",
            label="Emoji Creator: add emoji to dataset",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Create Emoji",
                icon="/assets/icon.svg",
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Emoji Creator",
            description=("Create an emoji from a prompt"),
        )
        inputs.str(
            "prompt",
            label="Prompt",
            description="The prompt to generate an emoji from",
        )
        return types.Property(inputs, view=form_view)

    def resolve_output(self, ctx):
        outputs = types.Object()

        dist = ctx.params.get("min_dist", 0)
        if dist < 0.1:
            outputs.str("prompt", label="Prompt")
            outputs.float("min_dist", label="Minimum Distance")
            view = types.View(label="Emoji Not Unique Enough")
            return types.Property(outputs, view=view)


    def execute(self, ctx):
        dataset = ctx.dataset
        model = foz.load_zoo_model("clip-vit-base32-torch")

        prompt = ctx.params.get("prompt", None)

        dist = get_min_dist(prompt, dataset)
        ctx.params["min_dist"] = dist

        if dist > 0.1:
            generate_sample_from_prompt(prompt, dataset, model)
            ctx.trigger("reload_dataset")
            return {}
        return {
            "prompt": prompt,
            "min_dist": dist,
        }


def register(plugin):
    plugin.register(CreateEmoji)
