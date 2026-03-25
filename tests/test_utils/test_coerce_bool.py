# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmdeploy.utils.config_utils import coerce_bool


class TestCoerceBool:
    """Tests for coerce_bool, which safely converts config values to bool.

    The primary motivation is YAML configs that store booleans as quoted
    strings (e.g. ``bgr_to_rgb: 'false'``).  Plain ``bool('false')`` returns
    ``True`` because every non-empty string is truthy in Python, which silently
    inverts the intended value.
    """

    def test_true_bool(self):
        assert coerce_bool(True) is True

    def test_false_bool(self):
        assert coerce_bool(False) is False

    @pytest.mark.parametrize(
        'value', ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES'])
    def test_truthy_strings(self, value):
        assert coerce_bool(value) is True, (
            f'Expected coerce_bool({value!r}) == True')

    @pytest.mark.parametrize(
        'value', ['false', 'False', 'FALSE', '0', 'no', 'No', 'NO', ''])
    def test_falsy_strings(self, value):
        assert coerce_bool(value) is False, (
            f'Expected coerce_bool({value!r}) == False')

    def test_quoted_false_string_is_false(self):
        """Core regression:
        bool('false') == True but coerce_bool('false') must be False."""
        result = coerce_bool('false')
        assert result is False, (
            "coerce_bool('false') returned True. "
            "This is the bug: Python's bool('false') is True because any "
            'non-empty string is truthy. coerce_bool must compare the string '
            'value explicitly.')

    def test_quoted_true_string_is_true(self):
        assert coerce_bool('true') is True

    def test_none_returns_default_false(self):
        assert coerce_bool(None) is False

    def test_none_returns_default_false_explicit(self):
        assert coerce_bool(None, default=False) is False

    def test_none_returns_default_true(self):
        assert coerce_bool(None, default=True) is True

    def test_int_zero_is_false(self):
        assert coerce_bool(0) is False

    def test_int_one_is_true(self):
        assert coerce_bool(1) is True

    def test_int_negative_is_true(self):
        assert coerce_bool(-1) is True

    def test_padded_false_string(self):
        assert coerce_bool('  false  ') is False

    def test_padded_true_string(self):
        assert coerce_bool('  true  ') is True

    @pytest.mark.parametrize('value',
                             ['maybe', 'nope', 'enabled', 'off', 'on'])
    def test_unrecognised_string_raises(self, value):
        with pytest.raises(ValueError, match='Cannot coerce'):
            coerce_bool(value)
