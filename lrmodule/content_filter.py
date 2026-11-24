import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from lir import Transformer
from lir.config.base import ConfigValue, check_is_empty, config_parser, pop_field
from lir.data.models import InstanceData
from lir.util import check_type

LOG = logging.getLogger(__name__)


class SourcePairRepeatFilter(Transformer):
    """
    Filter instances based on the repeat ID of the source pair in the dataset.

    The first time a source pair is present in a dataset has a repeat ID of 1, the second one has a repeat ID of 2, etc.
    When filtered on repeat ID = 1, only the first repeats of all source pairs are selected.
    """

    def __init__(self, repeat: int | None = 1):
        if repeat is not None and repeat < 1:
            raise ValueError("`repeat` should be a positive integer.")
        else:
            self.repeat = repeat

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
        if instances.source_ids is None and self.repeat is not None:
            raise ValueError("Source IDs are required to filter based on source pairs.")

        if self.repeat is not None and instances.source_ids is not None:
            sorted_source_pairs = np.sort(instances.source_ids, axis=1)
            merged_source_pairs = sorted_source_pairs[:, 0] + sorted_source_pairs[:, 1]
            repeats = np.empty(shape=len(merged_source_pairs), dtype=np.int32)
            for i_row, source_pair in enumerate(merged_source_pairs):
                repeats[i_row] = np.count_nonzero(merged_source_pairs[:i_row,] == source_pair) + 1
            mask = repeats == self.repeat
            return instances[mask]
        else:
            return instances


class PairDirectionFilter(Transformer):
    """
    Filter that keeps a single direction for each unordered source pair.

    When the data contains a comparison in both directions, i.e. a row for pair (A, B) and a
    row for pair (B, A), this filter keeps exactly one of the two rows, based on the direction
    rule:

    - ``smallest_first``: keep the row of which the first source has the lowest value in ``column``;
    - ``largest_first``: keep the row of which the first source has the highest value in ``column``.

    The value in ``column`` is taken from the features matrix, by resolving the column name against
    ``feature_columns``. Rows of which the value is tied between the two directions keep the first
    occurring row, so that both rules select the same row for that pair.
    Pairs that occur only once are kept as-is, since there is no direction to choose.
    """

    def __init__(self, feature_columns: list[str], column: str, direction: str):
        if direction not in ("smallest_first", "largest_first"):
            raise ValueError(f"unknown direction: {direction}")
        if column not in feature_columns:
            raise ValueError(f"column {column} not found in feature_columns: {feature_columns}")
        self._column_index = feature_columns.index(column)
        self.direction = direction

    def apply[DataType: InstanceData](self, instances: DataType) -> DataType:
        """
        Apply the direction filter to a dataset.

        Parameters
        ----------
        instances : InstanceData
            The dataset to filter. Requires two-column ``source_ids``.

        Returns
        -------
        InstanceData
            A dataset with one row per unordered source pair.
        """
        source_ids = check_type(np.ndarray, instances.source_ids, "source_ids are required to filter on pair direction")
        if len(source_ids.shape) != 2:  # noqa: PLR2004
            raise ValueError("Two-column source_ids are required to filter on pair direction.")

        features = check_type(np.ndarray, getattr(instances, "features", None))
        values = features[:, self._column_index]

        sorted_pairs = np.sort(source_ids, axis=1)
        keys = sorted_pairs[:, 0] + "|" + sorted_pairs[:, 1]

        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        boundaries = np.append(np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1, len(keys))

        keep = np.zeros(len(instances), dtype=bool)
        n_single = 0
        start = 0
        for end in boundaries:
            positions = order[start:end]
            if len(positions) > 1:
                if self.direction == "smallest_first":
                    selected = positions[np.argmin(values[positions])]
                else:
                    selected = positions[np.argmax(values[positions])]
                keep[selected] = True
            else:
                keep[positions[0]] = True
                n_single += 1
            start = end

        if n_single:
            LOG.warning(
                "PairDirectionFilter: %s of %s source pairs occurred in a single direction and were kept as-is.",
                n_single,
                len(boundaries),
            )
        return instances[keep]


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
              type: or
              conditions:
                - type: equals
                  column: weapon1
                  value: "1"
                - type: equals
                  column: weapon2
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
    def _get_column_data(instances: InstanceData, column: str) -> np.ndarray:
        """Get column data from instances, raising a clear error if the column doesn't exist."""
        if not hasattr(instances, column):
            available_fields = instances.all_fields
            raise ValueError(f"Column '{column}' not found in data. Available fields: {available_fields}")
        return getattr(instances, column)

    @staticmethod
    def equals(column: str, value: str) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where column equals value."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col_data = _ConditionBuilder._get_column_data(instances, column)
            return col_data == value

        return filter_fn

    @staticmethod
    def not_equals(column: str, value: str) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where column does not equal value."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col_data = _ConditionBuilder._get_column_data(instances, column)
            return col_data != value

        return filter_fn

    @staticmethod
    def columns_equal(column1: str, column2: str) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where two columns are equal."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col1_data = _ConditionBuilder._get_column_data(instances, column1)
            col2_data = _ConditionBuilder._get_column_data(instances, column2)
            return col1_data == col2_data

        return filter_fn

    @staticmethod
    def columns_not_equal(column1: str, column2: str) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where two columns are not equal."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col1_data = _ConditionBuilder._get_column_data(instances, column1)
            col2_data = _ConditionBuilder._get_column_data(instances, column2)
            return col1_data != col2_data

        return filter_fn

    @staticmethod
    def in_list(column: str, values: list[str]) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where column value is in the list."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col_data = _ConditionBuilder._get_column_data(instances, column)
            return np.isin(col_data, values)

        return filter_fn

    @staticmethod
    def not_in_list(column: str, values: list[str]) -> Callable[[InstanceData], np.ndarray]:
        """Return a filter function that selects rows where column value is not in the list."""

        def filter_fn(instances: InstanceData) -> np.ndarray:
            col_data = _ConditionBuilder._get_column_data(instances, column)
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


@config_parser
def parse_content_filter(config: ConfigValue, _: Path) -> ContentFilter:
    """Parse ContentFilter configuration."""
    condition_spec = pop_field(config, "condition", validate_type=dict)
    check_is_empty(config)
    filter_fn = _ConditionBuilder.build_condition(condition_spec)
    return ContentFilter(filter_fn)
