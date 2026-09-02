from pathlib import Path

import numpy as np
from lir.data.models import FeatureData
from numpy import array

from lrmodule.content_filter import ContentFilter, _ConditionBuilder


def test_equals_filter():
    """Test filtering rows where a column equals a specific value."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    weapon1 = array(["1", "2", "1", "3"])
    instances = FeatureData(features=features, weapon1=weapon1)

    filter_fn = _ConditionBuilder.equals("weapon1", "1")
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 2
    assert np.all(result.features == array([[1.0, 2.0], [5.0, 6.0]]))
    assert np.all(getattr(result, "weapon1") == array(["1", "1"]))


def test_not_equals_filter():
    """Test filtering rows where a column does not equal a specific value."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    weapon1 = array(["1", "2", "1", "3"])
    instances = FeatureData(features=features, weapon1=weapon1)

    filter_fn = _ConditionBuilder.not_equals("weapon1", "1")
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 2
    assert np.all(result.features == array([[3.0, 4.0], [7.0, 8.0]]))
    assert np.all(getattr(result, "weapon1") == array(["2", "3"]))


def test_columns_equal_filter():
    """Test filtering rows where two columns are equal."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    weapon1 = array(["1", "2", "1", "3"])
    weapon2 = array(["1", "2", "3", "1"])
    instances = FeatureData(features=features, weapon1=weapon1, weapon2=weapon2)

    filter_fn = _ConditionBuilder.columns_equal("weapon1", "weapon2")
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 2
    assert np.all(result.features == array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.all(getattr(result, "weapon1") == array(["1", "2"]))
    assert np.all(getattr(result, "weapon2") == array(["1", "2"]))


def test_columns_not_equal_filter():
    """Test filtering rows where two columns are not equal."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    weapon1 = array(["1", "2", "1", "3"])
    weapon2 = array(["1", "2", "3", "1"])
    instances = FeatureData(features=features, weapon1=weapon1, weapon2=weapon2)

    filter_fn = _ConditionBuilder.columns_not_equal("weapon1", "weapon2")
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 2
    assert np.all(result.features == array([[5.0, 6.0], [7.0, 8.0]]))
    assert np.all(getattr(result, "weapon1") == array(["1", "3"]))
    assert np.all(getattr(result, "weapon2") == array(["3", "1"]))


def test_in_list_filter():
    """Test filtering rows where column value is in a list."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    weapon1 = array(["1", "2", "1", "3"])
    instances = FeatureData(features=features, weapon1=weapon1)

    filter_fn = _ConditionBuilder.in_list("weapon1", ["1", "3"])
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 3
    assert np.all(result.features == array([[1.0, 2.0], [5.0, 6.0], [7.0, 8.0]]))
    assert np.all(getattr(result, "weapon1") == array(["1", "1", "3"]))


def test_not_in_list_filter():
    """Test filtering rows where column value is not in a list."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    weapon1 = array(["1", "2", "1", "3"])
    instances = FeatureData(features=features, weapon1=weapon1)

    filter_fn = _ConditionBuilder.not_in_list("weapon1", ["1", "3"])
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 1
    assert np.all(result.features == array([[3.0, 4.0]]))
    assert np.all(getattr(result, "weapon1") == array(["2"]))


def test_logical_and_filter():
    """Test combining multiple conditions with AND."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    weapon1 = array(["1", "2", "1", "3"])
    weapon2 = array(["1", "2", "3", "1"])
    instances = FeatureData(features=features, weapon1=weapon1, weapon2=weapon2)

    condition = {
        "type": "and",
        "conditions": [
            {"type": "equals", "column": "weapon1", "value": "1"},
            {"type": "equals", "column": "weapon2", "value": "1"},
        ],
    }
    filter_fn = _ConditionBuilder.build_condition(condition)
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 1
    assert np.all(result.features == array([[1.0, 2.0]]))


def test_logical_or_filter():
    """Test combining multiple conditions with OR."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    weapon1 = array(["1", "2", "1", "3"])
    instances = FeatureData(features=features, weapon1=weapon1)

    condition = {
        "type": "or",
        "conditions": [
            {"type": "equals", "column": "weapon1", "value": "1"},
            {"type": "equals", "column": "weapon1", "value": "3"},
        ],
    }
    filter_fn = _ConditionBuilder.build_condition(condition)
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 3
    assert np.all(result.features == array([[1.0, 2.0], [5.0, 6.0], [7.0, 8.0]]))


def test_logical_not_filter():
    """Test negating a condition."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    weapon1 = array(["1", "2", "1", "3"])
    instances = FeatureData(features=features, weapon1=weapon1)

    condition = {
        "type": "not",
        "condition": {"type": "equals", "column": "weapon1", "value": "1"},
    }
    filter_fn = _ConditionBuilder.build_condition(condition)
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 2
    assert np.all(result.features == array([[3.0, 4.0], [7.0, 8.0]]))


def test_complex_nested_condition():
    """Test a complex nested condition with multiple logical operators."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    weapon1 = array(["1", "2", "1", "3", "2"])
    weapon2 = array(["1", "2", "3", "1", "2"])
    instances = FeatureData(features=features, weapon1=weapon1, weapon2=weapon2)

    # Select: (weapon1 == "1" AND weapon2 == "1") OR (weapon1 == "2" AND weapon2 == "2")
    condition = {
        "type": "or",
        "conditions": [
            {
                "type": "and",
                "conditions": [
                    {"type": "equals", "column": "weapon1", "value": "1"},
                    {"type": "equals", "column": "weapon2", "value": "1"},
                ],
            },
            {
                "type": "and",
                "conditions": [
                    {"type": "equals", "column": "weapon1", "value": "2"},
                    {"type": "equals", "column": "weapon2", "value": "2"},
                ],
            },
        ],
    }
    filter_fn = _ConditionBuilder.build_condition(condition)
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 3
    assert np.all(result.features == array([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]]))


def test_filter_preserves_all_fields():
    """Test that filtering preserves all fields in the InstanceData."""
    # Arrange
    features = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = array([1, 0, 1])
    weapon1 = array(["1", "2", "1"])
    split1 = array(["t", "v", "t"])
    instances = FeatureData(features=features, labels=labels, weapon1=weapon1, split1=split1)

    filter_fn = _ConditionBuilder.equals("weapon1", "1")
    content_filter = ContentFilter(filter_fn)

    # Act
    result = content_filter.apply(instances)

    # Assert
    assert len(result) == 2
    assert np.all(result.features == array([[1.0, 2.0], [5.0, 6.0]]))
    assert np.all(result.labels == array([1, 1]))  # type: ignore[attr-defined]
    assert np.all(getattr(result, "weapon1") == array(["1", "1"]))
    assert np.all(getattr(result, "split1") == array(["t", "t"]))
