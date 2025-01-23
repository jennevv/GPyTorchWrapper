from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConf:
    num_inputs: int
    num_outputs: int
    output_index: Optional[int | list[int]] = None


@dataclass
class TransformerConf:
    transform_data: bool
    transformer_class: str
    transformer_options: Optional[dict] = None
    columns: Optional[list[int]] = None


@dataclass
class TransformConf:
    transform_input: TransformerConf
    transform_output: TransformerConf


# TODO: Remove MeanConf and relegate it to a ConstantKernel
@dataclass
class TrainingConf:
    model_class: str
    likelihood_class: str
    learning_rate: float
    learning_iterations: int
    noiseless: Optional[bool] = False
    botorch: Optional[bool] = False
    debug: Optional[bool] = True


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


# Function to create a Config object from a dictionary
def create_config(config_dict: dict) -> Config:
    return Config(
        data_conf=DataConf(
            num_inputs=config_dict["data_conf"]["num_inputs"],
            num_outputs=config_dict["data_conf"]["num_outputs"],
            output_index=config_dict["data_conf"]["output_index"],
        ),
        transform_conf=TransformConf(
            transform_input=TransformerConf(
                transform_data=config_dict["transform_conf"]["transform_input"][
                    "transform_data"
                ],
                transformer_class=config_dict["transform_conf"]["transform_input"][
                    "transformer_class"
                ],
                transformer_options=config_dict["transform_conf"]["transform_input"][
                    "transformer_options"
                ],
                columns=config_dict["transform_conf"]["transform_input"]["columns"],
            ),
            transform_output=TransformerConf(
                transform_data=config_dict["transform_conf"]["transform_output"][
                    "transform_data"
                ],
                transformer_class=config_dict["transform_conf"]["transform_output"][
                    "transformer_class"
                ],
                transformer_options=config_dict["transform_conf"]["transform_output"][
                    "transformer_options"
                ],
                columns=config_dict["transform_conf"]["transform_output"]["columns"],
            ),
        ),
        training_conf=TrainingConf(
            model_class=config_dict["training_conf"]["model_class"],
            likelihood_class=config_dict["training_conf"]["likelihood_class"],
            learning_rate=config_dict["training_conf"]["learning_rate"],
            learning_iterations=config_dict["training_conf"]["learning_iterations"],
            noiseless=config_dict["training_conf"]["noiseless"],
            botorch=config_dict["training_conf"]["botorch"],
            debug=config_dict["training_conf"]["debug"],
        ),
        testing_conf=TestingConf(
            test=config_dict["testing_conf"]["test"],
            test_size=config_dict["testing_conf"]["test_size"],
            strat_shuffle_split=config_dict["testing_conf"]["strat_shuffle_split"],
            kfold=config_dict["testing_conf"]["kfold"],
            kfold_bins=config_dict["testing_conf"]["kfold_bins"],
        ),
    )
