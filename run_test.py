import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from typing import Dict
from collision_test import CollisionTest
import numpy as np

def omegaconf_to_dict(d: DictConfig)->Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret

@hydra.main(config_name="config", config_path="./config", version_base="1.2")
def main(cfg: DictConfig):
    env = CollisionTest(
        config=omegaconf_to_dict(cfg),
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=False,
    )
    env.reset()
    while True:
        actions = env.zero_actions()
        _ = env.step(actions)
        env.render()

if __name__ == "__main__":
    main()