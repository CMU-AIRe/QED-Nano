import hydra
from omegaconf import DictConfig

from pipelinerl.rc_actor import run_actor_loop
from pipelinerl.utils import better_crashing


@hydra.main(config_path="../../conf", config_name="test_rc", version_base="1.3.2")
def hydra_entrypoint(cfg: DictConfig):
    with better_crashing("rc_actor"):
        run_actor_loop(cfg)


if __name__ == "__main__":
    hydra_entrypoint()

