import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from lir import Transformer
from lir.config.base import ContextAwareDict, pop_field
from lir.data.models import InstanceData

LOG = logging.getLogger(__name__)


class ContentFilter(Transformer):
    """
    Filter instances based on their content rather than just their indices.

    This filter allows selection based on column values, enabling complex filtering
    like selecting only instances from specific weapon types or matching weapon types.

    Parameters
    ----------
    filter_fn : Callable[[InstanceData], np.ndarray]
        A function that takes an InstanceData object and returns a boolean array
        indicating which rows to keep.

    Examples
    --------
    This filter can be used in a YAML configuration:

    .. code-block:: yaml

        data:
          provider: [...]
          strategy: [...]
          filter:
            method: lrmodule.content_filter.parse_content_filter
            condition:
              type: equals
              column: weapon1
              value: "1"

    Or to filter for matching weapon types:

    .. code-block:: yaml

        data:
          provider: [...]
          strategy: [...]
          filter:
            method: lrmodule.content_filter.parse_content_filter
            condition:
              type: columns_equal
              column1: weapon1
              column2: weapon2
    """

    def __init__(self, filter_fn: Callable[[InstanceData], np.ndarray]):
        self._filter_fn = filter_fn

    def apply[DataType: InstanceData](self, instances: DataType) -> DataType:
        """
        Apply the content-based filter to a dataset.

        Parameters
        ----------
        instances : InstanceData
            The dataset to filter.

        Returns
        -------
        InstanceData
            A dataset with only the instances that match the filter condition.
        """
        mask = self._filter_fn(instances)
        return instances[mask]


class _ConditionBuilder:
    """Builder for creating filter conditions based on instance data content."""

    @staticmethod
    def equals(column: str, value: str) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where column equals value."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col_data = getattr(instances, column)
            return col_data == value

        return filter_fn

    @staticmethod
    def not_equals(column: str, value: str) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where column does not equal value."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col_data = getattr(instances, column)
            return col_data != value

        return filter_fn

    @staticmethod
    def columns_equal(column1: str, column2: str) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where two columns are equal."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col1_data = getattr(instances, column1)
            col2_data = getattr(instances, column2)
            return col1_data == col2_data

        return filter_fn

    @staticmethod
    def columns_not_equal(column1: str, column2: str) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where two columns are not equal."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col1_data = getattr(instances, column1)
            col2_data = getattr(instances, column2)
            return col1_data != col2_data

        return filter_fn

    @staticmethod
    def in_list(column: str, values: list[str]) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where column value is in the list."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col_data = getattr(instances, column)
            return np.isin(col_data, values)

        return filter_fn

    @staticmethod
    def not_in_list(column: str, values: list[str]) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where column value is not in the list."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col_data = getattr(instances, column)
            return ~np.isin(col_data, values)

        return filter_fn

    @staticmethod
    def logical_and(conditions: list[dict]) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that combines multiple conditions with AND."""
        filter_fns = [_ConditionBuilder.build_condition(cond) for cond in conditions]

        def filter_fn(instances: InstanceData) -> np.ndarray:
            result = np.ones(len(instances), dtype=bool)
            for fn in filter_fns:
                result &= fn(instances)
            return result

        return filter_fn

    @staticmethod
    def logical_or(conditions: list[dict]) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that combines multiple conditions with OR."""
        filter_fns = [_ConditionBuilder.build_condition(cond) for cond in conditions]

        def filter_fn(instances: InstanceData) -> np.ndarray:
            result = np.zeros(len(instances), dtype=bool)
            for fn in filter_fns:
                result |= fn(instances)
            return result

        return filter_fn

    @staticmethod
    def logical_not(condition: dict) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that negates a condition."""
        filter_fn = _ConditionBuilder.build_condition(condition)

        def negated_fn(instances: InstanceData) -> np.ndarray:
            return ~filter_fn(instances)

        return negated_fn

    @staticmethod
    def build_condition(condition_spec: dict) -> Callable[[InstanceData], np.ndarray]:
        """Build a filter function from a condition specification."""
        condition_type = condition_spec.get("type")

        condition_builders = {
            "equals": lambda spec: _ConditionBuilder.equals(spec["column"], spec["value"]),
            "not_equals": lambda spec: _ConditionBuilder.not_equals(spec["column"], spec["value"]),
            "columns_equal": lambda spec: _ConditionBuilder.columns_equal(spec["column1"], spec["column2"]),
            "columns_not_equal": lambda spec: _ConditionBuilder.columns_not_equal(spec["column1"], spec["column2"]),
            "in": lambda spec: _ConditionBuilder.in_list(spec["column"], spec["values"]),
            "not_in": lambda spec: _ConditionBuilder.not_in_list(spec["column"], spec["values"]),
            "and": lambda spec: _ConditionBuilder.logical_and(spec["conditions"]),
            "or": lambda spec: _ConditionBuilder.logical_or(spec["conditions"]),
            "not": lambda spec: _ConditionBuilder.logical_not(spec["condition"]),
        }

        if condition_type not in condition_builders:
            raise ValueError(f"Unknown condition type: {condition_type}")

        return condition_builders[condition_type](condition_spec)


def parse_content_filter(config: ContextAwareDict, _: Path) -> ContentFilter:
    """Parse ContentFilter configuration."""
    condition_spec = pop_field(config, "condition", validate=lambda x: isinstance(x, dict))
    filter_fn = _ConditionBuilder.build_condition(condition_spec)
    return ContentFilter(filter_fn)
