import operator as op
import re
import datetime
from enum import Enum
from typing import Callable, Type, Any, Union

import numpy
import pandas as pd
import param
from panel import Row, Column


class Operations(param.Parameterized):
    def call_operation(self, data_value: Any) -> bool:
        raise NotImplementedError()


class DefaultOperations(Operations):
    @staticmethod
    def str_equals(data_value: Any, query_value: str) -> bool:
        return str(data_value) == query_value

    OPERATIONS_MAP: dict[str, Callable] = {"equals": str_equals}

    operation = param.Selector(objects=OPERATIONS_MAP)
    query_value = param.String()

    def __panel__(self):
        return Row(self.param.operation, self.param.query_value)

    @param.depends('query_value')
    def call_operation(self, data_value: Any) -> bool:
        return self.operation(data_value, self.query_value)


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
        flag: int = int(ignore_case)
        return re.search(query_value, data_value, flag) is not None

    OPERATIONS_MAP: dict[str, Callable] = {"equals": text_equal, "contains": text_contains,
                                           "starts with": starts_with, "ends with": ends_with,
                                           "regex search": regex_match}

    operation = param.Selector(objects=OPERATIONS_MAP)
    query_value = param.String()
    ignore_case = param.Boolean()

    def __panel__(self):
        return Row(self.param.operation, self.param.query_value, self.param.ignore_case)

    @param.depends('query_value', 'ignore_case')
    def call_operation(self, data_value: str) -> bool:
        return self.operation(data_value, self.query_value, self.ignore_case)


class NumberOperations(Operations):
    OPERATIONS_MAP: dict[str, Callable] = {"equals": op.eq, "is greater than": op.gt,
                                           "is greater than/equals": op.ge, "is less than": op.lt,
                                           "is less than/equals": op.le}

    operation = param.Selector(objects=OPERATIONS_MAP)
    query_value = param.Number()

    def __panel__(self):
        return Row(self.param.operation, self.param.query_value)

    @param.depends('query_value')
    def call_operation(self, data_value: Union[int, float]) -> bool:
        return self.operation(data_value, self.query_value)


class BooleanOperations(Operations):
    OPERATIONS_MAP: dict[str, Callable] = {"equals": op.eq}

    operation = param.Selector(objects=OPERATIONS_MAP)
    query_value = param.Boolean()

    def __panel__(self):
        return Row(self.param.operation, self.param.query_value)

    @param.depends('query_value')
    def call_operation(self, data_value: bool) -> bool:
        return self.operation(data_value, self.query_value)


class DateOperations(Operations):
    @staticmethod
    def inside(data_value: datetime.date, start_date: datetime.date,
               end_date: datetime.date, inclusive: bool) -> bool:
        if inclusive:
            return start_date <= data_value <= end_date
        else:
            return start_date < data_value < end_date

    @staticmethod
    def outside(data_value: datetime.date, start_date: datetime.date,
                end_date: datetime.date, inclusive: bool) -> bool:
        if inclusive:
            return (start_date > data_value) or (end_date < data_value)
        else:
            return (start_date >= data_value) or (end_date <= data_value)

    OPERATIONS_MAP: dict[str, Callable] = {"is inside of": inside, "is outside of": outside}

    operation = param.Selector(objects=OPERATIONS_MAP)
    start_date = param.CalendarDate()
    end_date = param.CalendarDate()
    inclusive = param.Boolean()

    def __panel__(self):
        return Row(self.param.operation, Column(self.param.start_date, self.param.end_date), self.param.inclusive)

    @param.depends('start_date', 'end_date', 'inclusive')
    def call_operation(self, data_value: pd.Timestamp) -> bool:
        data_value = data_value.date()
        return self.operation(data_value, self.start_date, self.end_date, self.inclusive)


class DataType(Enum):
    """
    Maps readable data type names to the pandas data types
    """
    TEXT = 'string'
    INTEGER = 'int64'
    DECIMAL = 'float64'
    BOOLEAN = 'boolean'
    DATETIME = 'datetime64[ns]'
    CATEGORY = 'category'


DATATYPE_OPERATIONS_MAP: dict[DataType, Type[Operations]] = {

    DataType.TEXT: TextOperations,
    DataType.INTEGER: NumberOperations,
    DataType.DECIMAL: NumberOperations,
    DataType.BOOLEAN: BooleanOperations,
    DataType.DATETIME: DateOperations,
    DataType.CATEGORY: DefaultOperations
}
