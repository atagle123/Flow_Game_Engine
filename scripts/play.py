import hydra
from src.game_engine.game_engine import GameEngine
from src.models.flow import FlowLearner

SEED = 123
N_FLOW_STEPS = 20
config_path = f"../logs/models/flow_engine/{SEED}/.hydra"


@hydra.main(config_path=config_path, config_name="config", version_base="1.3")
def main(cfg) -> None:
    model_cls = "FlowLearner"
    model = globals()[model_cls].load(
        cfg, dir=cfg.savepath, step=cfg.flow_model.train.steps
    )  # Load latest model step

    Game_Engine = GameEngine(model=model)

    Game_Engine.play(n_flow_steps=N_FLOW_STEPS)


if __name__ == "__main__":
    main()
