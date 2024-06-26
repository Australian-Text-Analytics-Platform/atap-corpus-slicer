import re
from enum import Enum
from typing import Callable, Type, Any, Union

import spacy
from spacy.tokens import Doc
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
    query_value = param.String(per_instance=True)

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
    query_value = param.String(per_instance=True)
    ignore_case = param.Boolean(per_instance=True)

    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)


    def __panel__(self):
        return pn.Row(self.param.operation, self.param.query_value, self.param.ignore_case)

    # @param.depends('operation')


    @param.depends('query_value', 'ignore_case')
    def call_operation(self, data_value: Union[str, Doc]) -> bool:
        if isinstance(data_value, Doc):
            data_value = data_value.text_with_ws
        return self.operation(data_value, self.query_value, self.ignore_case)


class IntegerOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.data_range = pn.widgets.EditableRangeSlider(step=1)
        self.data_range.start = data_series.min()
        self.data_range.end = data_series.max()
        self.data_range.value = (self.data_range.start, self.data_range.end)

    def __panel__(self):
        return pn.Row("is within the range", self.data_range)

    def call_operation(self, data_value: int) -> bool:
        return self.data_range.value[0] <= data_value <= self.data_range.value[1]


class FloatOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.data_range = pn.widgets.EditableRangeSlider()
        self.data_range.start = data_series.min()
        self.data_range.end = data_series.max()
        self.data_range.value = (self.data_range.start, self.data_range.end)

    def __panel__(self):
        return pn.Row("is within the range", self.data_range)

    def call_operation(self, data_value: float) -> bool:
        return self.data_range.value[0] <= data_value <= self.data_range.value[1]


class BooleanOperations(Operations):
    query_value = param.Selector(per_instance=True, objects=[True, False])

    def __panel__(self):
        return pn.Row("is equal to", self.param.query_value)

    @param.depends('query_value')
    def call_operation(self, data_value: bool) -> bool:
        return data_value == self.query_value


class DateOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.date_range = pn.widgets.DatetimeRangePicker()
        self.date_range.start = data_series.min()
        self.date_range.end = data_series.max()
        self.date_range.value = (self.date_range.start, self.date_range.end)

    def __panel__(self):
        return pn.Row("is within the range", self.date_range)

    def call_operation(self, data_value: pd.Timestamp) -> bool:
        data_value = data_value.to_pydatetime()
        return self.date_range.value[0] <= data_value <= self.date_range.value[1]


class CategoryOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.category = pn.widgets.MultiChoice(name='Category')
        self.category.options = list(data_series.unique())

    def __panel__(self):
        return pn.Row("is one of", self.category)

    def call_operation(self, data_value: Any) -> bool:
        return bool(data_value in self.category.value)


class SpacyOperations(Operations):
    attribute = param.Selector(per_instance=True)
    search = param.String(per_instance=True)
    ignore_case = param.Boolean(per_instance=True)
    use_regex = param.Boolean(per_instance=True)

    def __init__(self, nlp: spacy.Language, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.nlp: spacy.Language = nlp
        self.attribute_values = pn.widgets.MultiChoice(align="end")
        self.param.attribute.objects = self._get_attr_list()
        self.attribute = self.param.attribute.objects[0]

    def __panel__(self):
        return pn.Row(self.param.attribute, "is one of", self.attribute_values, "and matches", self.param.search, pn.Column(self.param.ignore_case, self.param.use_regex))

    def _get_attr_list(self) -> list[str]:
        if not len(self.data_series):
            return []
        attr_set = set(self.data_series[0]._.attr_vals.keys())
        custom_attr_set = set(self.data_series[0]._.custom_attr_vals.keys())

        return sorted(list(attr_set.union(custom_attr_set)))

    def _get_attr_values(self, attribute: str) -> dict[str, Any]:
        if not len(self.data_series):
            return {}

        attr_vals: set = set()
        doc_attr_vals: set[str]
        for doc in self.data_series:
            doc_attr_vals = doc._.attr_vals.get(attribute)
            if doc_attr_vals is None:
                doc_attr_vals = doc._.custom_attr_vals.get(attribute)
            attr_vals.update(doc_attr_vals)
        attr_dict = {str(attr): attr for attr in attr_vals}

        return attr_dict

    @param.depends('attribute', watch=True)
    def update_tag_values(self):
        self.attribute_values.options = self._get_attr_values(self.attribute)
        self.attribute_values.value = []

    @param.depends('attribute', 'search', 'ignore_case', 'use_regex')
    def call_operation(self, data_value: Doc) -> bool:
        if len(self.attribute_values.value) == 0:
            return True
        regex_flag = 0
        if self.ignore_case:
            regex_flag = re.I
        for token in data_value:
            if len(self.search):
                if self.use_regex:
                    text_match = re.search(self.search, token.text, regex_flag)
                elif self.ignore_case:
                    text_match = self.search.lower() in token.text.lower()
                else:
                    text_match = self.search in token.text
            else:
                text_match = True
            attribute_match = getattr(token, self.attribute, None) in self.attribute_values.value
            custom_attribute_match = getattr(token._, self.attribute, None) in self.attribute_values.value

            if text_match and (attribute_match or custom_attribute_match):
                return True
        return False


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
