"""PyTesseract OCR plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import requests

import numpy as np
import openai
from PIL import Image
import replicate
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors

import fiftyone as fo

import fiftyone.operators as foo
from fiftyone.operators import types

# pylint disable=no-name-in-module,import-error
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F

K = 3
DIST_WEIGHTS = np.array([0.6, 0.3, 0.1])
# DIST_THRESH = 0.15
# UNIQUENESS_THRESH = 0.7
DIST_THRESH = 0.0
UNIQUENESS_THRESH = 0.0


MODEL_NAME = "clip-vit-base32-torch"

TEXT_SIM_KEY = "text_sim"
TEXT_UNIQUENESS_FIELD = "text_uniqueness"
TEXT_EMBEDDING_FIELD = "clip_emoji_of_text_embedding"

IMAGE_SIM_KEY = "image_sim"
IMAGE_UNIQUENESS_FIELD = "image_uniqueness"
IMAGE_EMBEDDING_FIELD = "clip_image_embedding"


def _ensure_compliance(prompt):
    response = openai.Moderation.create(input=prompt)
    flagged = response["results"][0]["flagged"]
    return not flagged


def _compute_query_uniqueness(query_embedding, dataset):
    def _compute_unscaled_query_uniqueness(query, knn):
        distances, _ = knn.kneighbors(query.reshape(1, -1))
        res = distances @ DIST_WEIGHTS
        return res[0]

    embeddings = dataset.values(TEXT_EMBEDDING_FIELD)
    knn = NearestNeighbors(n_neighbors=K).fit(embeddings)

    most_unique_sample = (
        dataset.exists(TEXT_UNIQUENESS_FIELD).sort_by(TEXT_UNIQUENESS_FIELD).last()
    )
    mu_embedding = most_unique_sample[TEXT_EMBEDDING_FIELD]

    uqu = _compute_unscaled_query_uniqueness(query_embedding, knn)
    scale = _compute_unscaled_query_uniqueness(mu_embedding, knn)
    return uqu / scale


def _update_uniqueness(dataset):
    fob.compute_uniqueness(
        dataset,
        uniqueness_field=TEXT_UNIQUENESS_FIELD,
        embeddings=TEXT_EMBEDDING_FIELD,
    )
    # fob.compute_uniqueness(
    #     dataset,
    #     uniqueness_field=IMAGE_UNIQUENESS_FIELD,
    #     embeddings=IMAGE_EMBEDDING_FIELD,
    # )


def get_min_dist(query_embedding, dataset):
    closest_sample = dataset.sort_by_similarity(
        query_embedding, brain_key=TEXT_SIM_KEY, k=1
    ).first()
    return cosine(query_embedding, closest_sample[TEXT_EMBEDDING_FIELD])


def generate_filename(prompt):
    for pattern in [" ", ",", "'", ":", "?", "!"]:
        prompt = prompt.replace(pattern, "")
    return prompt.replace(" ", "_") + ".png"


def generate_sample_embeddings(sample, model):
    name = sample.name
    prompt = f"An emoji of {name}"
    sample[TEXT_EMBEDDING_FIELD] = model.embed_prompt(prompt)
    # im_arr = np.array(Image.open(sample.filepath))
    # sample[IMAGE_EMBEDDING_FIELD] = model.embed(im_arr)


def generate_sample_from_prompt(prompt, dataset, model):
    input = {
        "width": 1024,
        "height": 1024,
        "prompt": f"A TOK emoji of {prompt}, white background",
        "refine": "expert_ensemble_refiner",
        "scheduler": "K_EULER",
        "lora_scale": 0.99,
        "num_outputs": 1,
        "guidance_scale": 8,
        "apply_watermark": False,
        "high_noise_frac": 0.99,
        "negative_prompt": "",
        "prompt_strength": 0.95,
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

    url = replicate.run(
        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
        input={"image": open(filepath, "rb")},
    )
    content = requests.get(url).content
    with open(filepath, "wb") as f:
        f.write(content)

    sample = fo.Sample(
        filepath=filepath,
        name=prompt,
        original=False,
    )
    generate_sample_embeddings(sample, model)
    dataset.add_sample(sample)

    sample_ids = [sample.id]

    # image_index = dataset.load_brain_results(IMAGE_SIM_KEY)
    # image_embeddings = np.array([sample[IMAGE_EMBEDDING_FIELD]])
    # image_index.add_to_index(image_embeddings, sample_ids)

    text_index = dataset.load_brain_results(TEXT_SIM_KEY)
    text_embeddings = np.array([sample[TEXT_EMBEDDING_FIELD]])
    text_index.add_to_index(text_embeddings, sample_ids)

    _update_uniqueness(dataset)


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
        if "emoji" in ctx.dataset.name.lower():
            return types.Placement(
                types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
                types.Button(
                    label="Create Emoji",
                    icon="/assets/icon.svg",
                ),
            )
        else:
            return types.Placement()

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

        outputs.str("prompt", label="Prompt")
        outputs.float(
            "dist",
            label=f"Minimum Distance to Existing Emoji. Must be > {DIST_THRESH}",
        )
        outputs.float(
            "uniqueness",
            label=f"Uniqueness of Emoji. Must be > {UNIQUENESS_THRESH}",
        )
        view = types.View(label="Emoji Not Distinct Enough")
        return types.Property(outputs, view=view)

    def execute(self, ctx):
        dataset = ctx.dataset
        model = foz.load_zoo_model(MODEL_NAME)

        prompt = ctx.params.get("prompt", None)
        compliant = _ensure_compliance(prompt)
        if not compliant:
            return {}

        query_embedding = model.embed_prompt(f"An emoji of {prompt}")

        uniqueness = _compute_query_uniqueness(query_embedding, dataset)
        dist = get_min_dist(query_embedding, dataset)

        lcond = len(prompt) <= 20
        ucond = uniqueness > UNIQUENESS_THRESH
        dcond = dist > DIST_THRESH

        # if lcond and ucond and dcond:
        generate_sample_from_prompt(prompt, dataset, model)
        ctx.trigger("reload_samples")
        return {}
        # else:
        #     return {
        #         "prompt": prompt,
        #         "dist": dist,
        #         "uniqueness": uniqueness,
        # }


def register(plugin):
    plugin.register(CreateEmoji)
