import hydra
from src.game_engine.game_engine import GameEngine
from src.models.flow import FlowLearner

seed = 685
config_path = f"../logs/models/flow_engine/{seed}/.hydra"


@hydra.main(config_path=config_path, config_name="config", version_base="1.3")
def main(cfg) -> None:
    model_cls = "FlowLearner"
    model = globals()[model_cls].load(cfg, dir = cfg.savepath, step = 130000)

    Game_Engine = GameEngine(model=model)

    Game_Engine.play()


if __name__ == "__main__":
    main()