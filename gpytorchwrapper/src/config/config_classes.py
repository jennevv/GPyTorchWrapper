from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConf:
    num_inputs: int
    num_outputs: int
    output_index: Optional[int | list[int]] = None


@dataclass
class TransformerConf:
    transform_data: bool = False
    transformer_class: str = "DefaultTransformer"
    transformer_options: Optional[dict] = field(default_factory=dict)
    columns: Optional[list[int]] = None


@dataclass
class TransformConf:
    transform_input: TransformerConf = field(default_factory=TransformerConf)
    transform_output: TransformerConf = field(default_factory=TransformerConf)

@dataclass
class OptimizerConf:
    optimizer_class: str = "Adam"
    optimizer_options: Optional[dict] = field(default_factory= lambda: {"lr": 0.1})

@dataclass
class LikelihoodConf:
    likelihood_class: str = "GaussianLikelihood"
    likelihood_options: Optional[dict] = field(default_factory= dict)

@dataclass
class ModelConf:
    model_class: str
    model_options: Optional[dict] = field(default_factory= dict)

@dataclass
class TrainingConf:
    model: ModelConf = field(default_factory=ModelConf)
    likelihood: LikelihoodConf = field(default_factory=LikelihoodConf)
    learning_iterations: int = 100
    botorch: Optional[bool] = False
    debug: Optional[bool] = True
    optimizer: OptimizerConf = field(default_factory=OptimizerConf)


@dataclass
class TestingConf:
    test: bool = False
    test_size: float = 0.2
    strat_shuffle_split: bool = False
    kfold: bool = False
    kfold_bins: Optional[int] = None


@dataclass
class Config:
    data_conf: DataConf
    transform_conf: TransformConf
    training_conf: TrainingConf
    testing_conf: TestingConf


# Function to create a Config object from a dictionary with defaults
def create_config(config_dict: dict) -> Config:
    return Config(
        data_conf=DataConf(
            num_inputs=config_dict["data_conf"]["num_inputs"],
            num_outputs=config_dict["data_conf"]["num_outputs"],
            output_index=config_dict["data_conf"]["output_index"],
        ),
        transform_conf=TransformConf(
            transform_input=TransformerConf(
                transform_data=config_dict["transform_conf"]["transform_input"].get(
                    "transform_data", False
                ),
                transformer_class=config_dict["transform_conf"]["transform_input"].get(
                    "transformer_class", "DefaultTransformer"
                ),
                transformer_options=config_dict["transform_conf"][
                    "transform_input"
                ].get("transformer_options", {}),
                columns=config_dict["transform_conf"]["transform_input"].get("columns"),
            ) if config_dict["transform_conf"]["transform_input"] else TransformerConf(),
            transform_output=TransformerConf(
                transform_data=config_dict["transform_conf"]["transform_output"].get(
                    "transform_data", False
                ),
                transformer_class=config_dict["transform_conf"]["transform_output"].get(
                    "transformer_class", "DefaultTransformer"
                ),
                transformer_options=config_dict["transform_conf"][
                    "transform_output"
                ].get("transformer_options", {}),
                columns=config_dict["transform_conf"]["transform_output"].get(
                    "columns", []
                ),
            ) if config_dict["transform_conf"]["transform_output"] else TransformerConf()
        ),
        training_conf=TrainingConf(
            model=ModelConf(
                model_class=config_dict["training_conf"]["model"].get(
                    "model_class", "SingleGPRBF"
                ),
                model_options=config_dict["training_conf"]["model"].get(
                    "model_options", {}
                )
            ),
            likelihood=LikelihoodConf(
                likelihood_class=config_dict["training_conf"]["likelihood"].get(
                    "likelihood_class", "GaussianLikelihood"
                ),
                likelihood_options=config_dict["training_conf"]["likelihood"].get(
                    "likelihood_options", {}
                ) if config_dict["training_conf"]["likelihood"] else LikelihoodConf()
            ),
            learning_iterations=config_dict["training_conf"].get(
                "learning_iterations", 100
            ),
            botorch=config_dict["training_conf"].get("botorch", False),
            debug=config_dict["training_conf"].get("debug", True),
            optimizer=OptimizerConf(
                optimizer_class=config_dict["training_conf"]["optimizer"].get(
                    "optimizer_class", "Adam"
                ),
                optimizer_options=config_dict["training_conf"]["optimizer"].get(
                    "optimizer_options", {"lr": 0.1}
                )
            ) if config_dict["training_conf"]["optimizer"] else OptimizerConf()
        ),
        testing_conf=TestingConf(
            test=config_dict["testing_conf"].get("test", False),
            test_size=config_dict["testing_conf"].get("test_size", 0.2),
            strat_shuffle_split=config_dict["testing_conf"].get(
                "strat_shuffle_split", False
            ),
            kfold=config_dict["testing_conf"].get("kfold", False),
            kfold_bins=config_dict["testing_conf"].get("kfold_bins", None),
        ),
    )
