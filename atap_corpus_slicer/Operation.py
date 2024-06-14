import re
from enum import Enum
from typing import Callable, Type, Any

import pandas as pd
import param
import panel as pn
from pandas import Series

pn.extension()


class Operations(param.Parameterized):
    def __init__(self, data_series: Series, **params):
        super().__init__(**params)
        self.data_series: Series = data_series

    def call_operation(self, data_value: Any) -> bool:
        raise NotImplementedError()


class DefaultOperations(Operations):
    query_value = param.String()

    def __panel__(self):
        return pn.Row("is equal to", self.param.query_value)

    @param.depends('query_value')
    def call_operation(self, data_value: Any) -> bool:
        return str(data_value) == self.query_value


class TextOperations(Operations):
    @staticmethod
    def text_equal(data_value: str, query_value: str, ignore_case: bool) -> bool:
        if ignore_case:
            data_value = data_value.lower()
            query_value = query_value.lower()
        return data_value == query_value

    @staticmethod
    def text_contains(data_value: str, query_value: str, ignore_case: bool) -> bool:
        if ignore_case:
            data_value = data_value.lower()
            query_value = query_value.lower()
        return query_value in data_value

    @staticmethod
    def starts_with(data_value: str, query_value: str, ignore_case: bool) -> bool:
        if ignore_case:
            data_value = data_value.lower()
            query_value = query_value.lower()
        return data_value.startswith(query_value)

    @staticmethod
    def ends_with(data_value: str, query_value: str, ignore_case: bool) -> bool:
        if ignore_case:
            data_value = data_value.lower()
            query_value = query_value.lower()
        return data_value.endswith(query_value)

    @staticmethod
    def regex_match(data_value: str, query_value: str, ignore_case: bool) -> bool:
        flag = 0
        if ignore_case:
            flag = re.I
        return re.search(query_value, data_value, flag) is not None

    OPERATIONS_MAP: dict[str, Callable] = {"equals": text_equal, "contains": text_contains,
                                           "starts with": starts_with, "ends with": ends_with,
                                           "regex search": regex_match}

    operation = param.Selector(objects=OPERATIONS_MAP)
    query_value = param.String()
    ignore_case = param.Boolean()

    def __panel__(self):
        return pn.Row(self.param.operation, self.param.query_value, self.param.ignore_case)

    @param.depends('query_value', 'ignore_case')
    def call_operation(self, data_value: str) -> bool:
        return self.operation(data_value, self.query_value, self.ignore_case)


class IntegerOperations(Operations):
    data_range = pn.widgets.EditableRangeSlider(step=1)

    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.data_range.start = data_series.min()
        self.data_range.end = data_series.max()
        self.data_range.value = (self.data_range.start, self.data_range.end)

    def __panel__(self):
        return pn.Row("is within the range", self.data_range)

    @param.depends('data_range')
    def call_operation(self, data_value: int) -> bool:
        return self.data_range.value[0] <= data_value <= self.data_range.value[1]


class FloatOperations(Operations):
    data_range = pn.widgets.EditableRangeSlider()

    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.data_range.start = data_series.min()
        self.data_range.end = data_series.max()
        self.data_range.value = (self.data_range.start, self.data_range.end)

    def __panel__(self):
        return pn.Row("is within the range", self.data_range)

    @param.depends('data_range')
    def call_operation(self, data_value: float) -> bool:
        return self.data_range.value[0] <= data_value <= self.data_range.value[1]


class BooleanOperations(Operations):
    query_value = param.Selector(objects=[True, False])

    def __panel__(self):
        return pn.Row("is equal to", self.param.query_value)

    @param.depends('query_value')
    def call_operation(self, data_value: bool) -> bool:
        return data_value == self.query_value


class DateOperations(Operations):
    date_range = pn.widgets.DatetimeRangePicker()

    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.date_range.start = data_series.min()
        self.date_range.end = data_series.max()
        self.date_range.value = (self.date_range.start, self.date_range.end)

    def __panel__(self):
        return pn.Row("is within the range", self.date_range)

    @param.depends('date_range')
    def call_operation(self, data_value: pd.Timestamp) -> bool:
        data_value = data_value.to_pydatetime()
        return self.date_range.value[0] <= data_value <= self.date_range.value[1]


class CategoryOperations(Operations):
    category = pn.widgets.MultiChoice(name='Category')

    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.category.options = list(data_series.unique())

    def __panel__(self):
        return pn.Row("is one of", self.category)

    @param.depends('category')
    def call_operation(self, data_value: Any) -> bool:
        return bool(data_value in self.category.value)


class DataType(Enum):
    """
    Maps readable data type names to the pandas data types
    """
    TEXT = 'string'
    INTEGER = 'int64'
    DECIMAL = 'float64'
    BOOLEAN = 'bool'
    DATETIME = 'datetime64[ns]'
    CATEGORY = 'category'


DATATYPE_OPERATIONS_MAP: dict[DataType, Type[Operations]] = {
    DataType.TEXT: TextOperations,
    DataType.INTEGER: IntegerOperations,
    DataType.DECIMAL: FloatOperations,
    DataType.BOOLEAN: BooleanOperations,
    DataType.DATETIME: DateOperations,
    DataType.CATEGORY: CategoryOperations
}
