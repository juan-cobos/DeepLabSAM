from deeplabsam.pose.models.criterions.aggregators import WeightedLossAggregator
from deeplabsam.pose.models.criterions.base import CRITERIONS, LOSS_AGGREGATORS, BaseCriterion, BaseLossAggregator
from deeplabsam.pose.models.criterions.weighted import WeightedBCECriterion, WeightedHuberCriterion, WeightedMSECriterion
