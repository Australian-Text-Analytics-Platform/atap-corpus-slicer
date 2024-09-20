import operator
import re
from enum import Enum
from typing import Callable, Type, Any, Union

from spacy import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
import pandas as pd
import panel as pn
from pandas import Series

pn.extension()


class Operations:
    def __init__(self, data_series: Series, **params):
        super().__init__(**params)
        self.data_series: Series = data_series
        self.panel = pn.Row()

    def __panel__(self):
        return self.panel

    def call_operation(self, data_value: Any) -> bool:
        raise NotImplementedError()


class DefaultOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.query_value = pn.widgets.TextInput(name="Search")
        self.panel.objects = ["is equal to", self.query_value]

    def call_operation(self, data_value: Any) -> bool:
        return str(data_value) == self.query_value.value_input


class TextOperations(Operations):
    @staticmethod
    def text_contains(data_value: str, query_value: str, ignore_case: bool, use_regex: bool,
                      count_bound_fn: Callable, count_threshold: int) -> bool:
        if use_regex:
            regex_flag: int = 0
            if ignore_case:
                regex_flag: int = re.I
            count = len(re.findall(query_value, data_value, regex_flag))
        else:
            if ignore_case:
                data_value = data_value.casefold()
                query_value = query_value.casefold()
            count = data_value.count(query_value)

        return count_bound_fn(count, count_threshold)

    @staticmethod
    def text_equal(data_value: str, query_value: str, ignore_case: bool, use_regex: bool) -> bool:
        if use_regex:
            regex_flag: int = 0
            if ignore_case:
                regex_flag = re.I
            regex_match = re.fullmatch(query_value, data_value, regex_flag) is not None
            return regex_match
        else:
            if ignore_case:
                data_value = data_value.casefold()
                query_value = query_value.casefold()
            return data_value == query_value

    @staticmethod
    def starts_with(data_value: str, query_value: str, ignore_case: bool, use_regex: bool) -> bool:
        if use_regex:
            regex_flag: int = 0
            if ignore_case:
                regex_flag: int = re.I
            pattern = re.compile(r'^' + re.escape(query_value), flags=regex_flag)

            return bool(pattern.search(data_value))
        else:
            if ignore_case:
                data_value = data_value.casefold()
                query_value = query_value.casefold()

            return data_value.startswith(query_value)

    @staticmethod
    def ends_with(data_value: str, query_value: str, ignore_case: bool, use_regex: bool) -> bool:
        if use_regex:
            regex_flag: int = 0
            if ignore_case:
                regex_flag: int = re.I
            pattern = re.compile(re.escape(query_value) + r'$', flags=regex_flag)

            return bool(pattern.search(data_value))
        else:
            if ignore_case:
                data_value = data_value.casefold()
                query_value = query_value.casefold()

            return data_value.endswith(query_value)

    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)

        operations_map: dict[str, Callable] = {"contains": self.text_contains, "equals": self.text_equal,
                                               "starts with": self.starts_with, "ends with": self.ends_with}
        self.operation = pn.widgets.Select(name="Operation", options=operations_map)
        self.query_value = pn.widgets.TextInput(name="Search")
        self.count_bound_fn = pn.widgets.Select(options={"at least": operator.ge, "at most": operator.le},
                                                align="end", visible=False, width=100)
        self.count_threshold = pn.widgets.IntInput(name="Occurrences", start=1, value=1,
                                                   align="end", visible=False, width=100)
        self.ignore_case = pn.widgets.Checkbox(name="Ignore case")
        self.use_regex = pn.widgets.Checkbox(name="Regular expression")

        self.panel.objects = [self.operation, self.query_value, self.count_bound_fn, self.count_threshold, pn.Column(self.ignore_case, self.use_regex),
                              pn.bind(self.toggle_count_inputs, self.operation)]

    def toggle_count_inputs(self, *_):
        count_op = self.operation.value == self.text_contains
        self.count_bound_fn.visible = count_op
        self.count_threshold.visible = count_op

    def call_operation(self, data_value: Union[str, Doc]) -> bool:
        if isinstance(data_value, Doc):
            data_value = data_value.text_with_ws
        if self.operation.value == self.text_contains:
            return self.operation.value(data_value, self.query_value.value_input, self.ignore_case.value, self.use_regex.value, self.count_bound_fn.value, self.count_threshold.value)
        return self.operation.value(data_value, self.query_value.value_input, self.ignore_case.value, self.use_regex.value)


class IntegerOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.data_range = pn.widgets.EditableRangeSlider(name="is within the range", step=1)
        self.data_range.start = data_series.min()
        self.data_range.end = data_series.max()
        self.data_range.value = (self.data_range.start, self.data_range.end)

        self.panel.objects = [self.data_range]

    def call_operation(self, data_value: int) -> bool:
        return self.data_range.value[0] <= data_value <= self.data_range.value[1]


class FloatOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.data_range = pn.widgets.EditableRangeSlider(name="is within the range")
        self.data_range.start = data_series.min()
        self.data_range.end = data_series.max()
        self.data_range.value = (self.data_range.start, self.data_range.end)

        self.panel.objects = [self.data_range]

    def call_operation(self, data_value: float) -> bool:
        return self.data_range.value[0] <= data_value <= self.data_range.value[1]


class BooleanOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.query_value = pn.widgets.Select(name="is equal to", options=[True, False])

        self.panel.objects = [self.query_value]

    def call_operation(self, data_value: bool) -> bool:
        return data_value == self.query_value.value


class DateOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.date_range = pn.widgets.DatetimeRangePicker(name="is within the range")
        self.date_range.start = data_series.min()
        self.date_range.end = data_series.max()
        self.date_range.value = (self.date_range.start, self.date_range.end)

        self.panel.objects = [self.date_range]

    def call_operation(self, data_value: pd.Timestamp) -> bool:
        data_value = data_value.to_pydatetime()
        return self.date_range.value[0] <= data_value <= self.date_range.value[1]


class CategoryOperations(Operations):
    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.category = pn.widgets.MultiChoice(name="is one of", options=list(data_series.unique()))

        self.panel.objects = [self.category]

    def call_operation(self, data_value: Any) -> bool:
        return bool(data_value in self.category.value)


class SpacyTokenOperations(Operations):
    attr_renames: dict[str, str] = {"pos_": "part-of-speech", "tag_": "part-of-speech (fine-grained)",
                                    "dep_": "dependency", "lemma_": "lemmatised text", "ent_type_": "entity type",
                                    "lang_": "language", "lower_": "lowercase", "suffix_": "suffix",
                                    "prefix_": "prefix"}

    def __init__(self, data_series: Series, **params):
        super().__init__(data_series, **params)
        self.attribute = pn.widgets.Select(name="Attribute", options=self._get_attr_list())
        self.attribute_values = pn.widgets.MultiChoice(name="is one of", align="end")
        self.search = pn.widgets.TextInput(name="matches")
        self.ignore_case = pn.widgets.Checkbox(name="Ignore case")
        self.use_regex = pn.widgets.Checkbox(name="Regular expression")

        self.panel.objects = [self.attribute, self.attribute_values, self.search,
                              pn.Column(self.ignore_case, self.use_regex),
                              pn.bind(self.update_tag_values, self.attribute)]

    def _get_attr_list(self) -> dict[str, str]:
        if not len(self.data_series):
            return {}
        attr_set = set(self.data_series[0]._.attr_vals.keys())
        sorted_attr = sorted(attr_set)
        sorted_custom_attr = sorted(self.data_series[0]._.custom_attr_vals.keys())

        attr_dict = {}
        for attr in sorted_attr:
            rename = self.attr_renames.get(attr)
            if rename is not None:
                attr_dict[rename] = attr
        for attr in sorted_custom_attr:
            attr_dict[attr] = attr

        return attr_dict

    def _get_attr_values(self, attribute: str) -> dict[str, Any]:
        if (not len(self.data_series)) or (not Doc.has_extension("attr_vals")) or (not Doc.has_extension("custom_attr_vals")):
            return {}

        attr_vals: set = set()
        doc_attr_vals: set[str]
        for doc in self.data_series:
            doc_attr_vals = doc._.attr_vals.get(attribute)
            if doc_attr_vals is None:
                doc_attr_vals = doc._.custom_attr_vals.get(attribute)
            if doc_attr_vals is None:
                continue
            attr_vals.update(doc_attr_vals)
        attr_dict = {str(attr): attr for attr in attr_vals}

        return attr_dict

    def update_tag_values(self, *_):
        self.attribute_values.options = self._get_attr_values(self.attribute.value)
        self.attribute_values.value = []

    def call_operation(self, data_value: Doc) -> bool:
        if len(self.attribute_values.value) == 0:
            return True
        regex_flag = 0
        if self.ignore_case:
            regex_flag = re.I
        search_val = self.search.value_input
        for token in data_value:
            if len(search_val):
                if self.use_regex:
                    text_match = re.search(search_val, token.text, regex_flag)
                elif self.ignore_case:
                    text_match = search_val.casefold() in token.text.casefold()
                else:
                    text_match = search_val in token.text
            else:
                text_match = True
            attribute_match = getattr(token, self.attribute.value, None) in self.attribute_values.value
            custom_attribute_match = getattr(token._, self.attribute.value, None) in self.attribute_values.value

            if text_match and (attribute_match or custom_attribute_match):
                return True
        return False


class SpacyPhraseOperations(Operations):
    attr_renames: dict[str, str] = {"pos": "part-of-speech", "tag": "part-of-speech (fine-grained)",
                                    "dep": "dependency", "lemma": "lemmatised text", "ent_type": "entity type",
                                    "lower": "lowercase", "suffix": "suffix", "prefix": "prefix", "shape": "shape"}

    def __init__(self, data_series: Series, model: Language, **params):
        super().__init__(data_series, **params)
        self.model: Language = model
        self.attribute = pn.widgets.Select(name="Attribute", options=self._get_attr_list())
        self.search = pn.widgets.TextInput(name="matches")
        self.search.param.watch(self._set_search_doc, 'value')
        self.search_doc: Doc = self.model(self.search.value)

        self.panel.objects = [self.attribute, self.search]

    def _get_attr_list(self) -> dict[str, str]:
        if not len(self.data_series):
            return {}
        attr_set = set(self.data_series[0]._.attr_vals.keys())
        sorted_attr = sorted(attr_set)
        sorted_custom_attr = sorted(self.data_series[0]._.custom_attr_vals.keys())

        attr_dict = {}
        for attr in sorted_attr:
            rename = self.attr_renames.get(attr)
            if rename is not None:
                attr_dict[rename] = attr
        for attr in sorted_custom_attr:
            attr_dict[attr] = attr

        return attr_dict

    def _set_search_doc(self, *_):
        self.search_doc = self.model(self.search.value)

    def call_operation(self, data_value: Doc) -> bool:
        matcher: PhraseMatcher = PhraseMatcher(self.model.vocab, self.attribute.value)
        matcher.add('pattern', [self.search_doc])

        return len(matcher(data_value)) != 0


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
