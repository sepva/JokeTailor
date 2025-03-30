import click
import TrainingInstances
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
@click.option(
    "--local_rank",
    help="Local rank from deepspeed",
)
def start_training(config_file, local_rank):
    print(f"Rank {local_rank} started training script")
    with open(config_file, "r") as f:
        config = json.load(f)
        trainer_class = getattr(TrainingInstances, config["training_class"])
        trainer = trainer_class(**config["training_args"])

    trainer.start_training()


if __name__ == "__main__":
    start_training()
