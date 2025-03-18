import click
from InferenceInterface import InferenceInterface
import os
import json

dirname = os.path.dirname(__file__)


@click.command()
@click.option(
    "--config_file",
    "-f",
    default="config.json",
    help="Path to the config file",
    type=click.Path(exists=True),
)
def start_inference(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
        inference_config = config.pop("inference_config")
        ii = InferenceInterface(**config)
    
    inference_fun = getattr(ii, inference_config["type"])
    inference_fun(**inference_config["args"])

if __name__ == "__main__":
    start_inference()