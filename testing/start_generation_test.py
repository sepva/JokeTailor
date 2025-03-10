import click
import os
import json
from TestInterface import TestInterface

dirname = os.path.dirname(__file__)


@click.command()
@click.option(
    "--config_file",
    "-f",
    default="config.json",
    help="Path to the config file",
    type=click.Path(exists=True),
)
def start_generation_test(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
        test_class = TestInterface(**config)

    test_class.test_joke_generation_output()


if __name__ == "__main__":
    start_generation_test()
