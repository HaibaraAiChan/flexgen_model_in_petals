from bloombee.models.falcon.block import WrappedFalconBlock
from bloombee.models.falcon.config import DistributedFalconConfig
from bloombee.models.falcon.model import (
    DistributedFalconForCausalLM,
    DistributedFalconForSequenceClassification,
    DistributedFalconModel,
)
from bloombee.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedFalconConfig,
    model=DistributedFalconModel,
    model_for_causal_lm=DistributedFalconForCausalLM,
    model_for_sequence_classification=DistributedFalconForSequenceClassification,
)
