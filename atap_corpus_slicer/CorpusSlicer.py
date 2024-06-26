import logging
import traceback
from logging.handlers import RotatingFileHandler
from os.path import join, dirname, abspath
from typing import Optional, Callable, Any, Type, Union

import panel as pn
import spacy
import param
from atap_corpus.corpus.base import BaseCorpora
from atap_corpus.corpus.corpus import DataFrameCorpus
from pandas import DataFrame, Series
from panel import Row, Column

from panel.theme import Fast

from atap_corpus_loader import CorpusLoader, EventType
from panel.widgets import Tqdm, Button, TextInput
from spacy import Language
from spacy.tokens import Doc
from tqdm import tqdm

from atap_corpus_slicer.Operation import DefaultOperations, SpacyOperations, DATATYPE_OPERATIONS_MAP, Operations, \
    DataType, TextOperations

pn.extension(notifications=True, design=Fast)
tqdm.pandas()


class FilterParams(param.Parameterized):
    negation = param.Boolean(label='Negate', default=False, instantiate=True)
    selected_label = param.Selector(label='Data label', instantiate=True)

    def __init__(self, selected_corpus: DataFrameCorpus, nlp: Optional[spacy.Language] = None, **params):
        super().__init__(**params)
        self.selected_operations = DefaultOperations(Series())
        self.nlp: Optional[spacy.Language] = nlp
        self.spacy_filter: bool = (self.nlp is not None) and selected_corpus.uses_spacy()

        self.remove_filter_button = Button(
            name="Remove",
            button_type="warning", button_style="outline", align="end"
        )

        self.panel = Row()

        self.selected_corpus: DataFrameCorpus = selected_corpus
        self.update_corpus(selected_corpus)

    def __panel__(self):
        return self.panel

    def _update_panel(self):
        objects = []
        if not self.spacy_filter:
            objects.append(self.param.selected_label)
        objects.extend([self.selected_operations, self.param.negation, self.remove_filter_button])
        self.panel.objects = objects

    def update_corpus(self, new_corpus: DataFrameCorpus):
        self.selected_corpus = new_corpus

        label_list: list[str] = []
        if self.selected_corpus is not None:
            df: DataFrame = self.selected_corpus.to_dataframe()
            label_list = df.columns
        self.param.selected_label.objects = label_list
        if len(label_list):
            self.selected_label = label_list[0]

        self._set_operations()

    @param.depends('selected_label', watch=True)
    def _set_operations(self, *_):
        if self.selected_corpus is None:
            return
        df: DataFrame = self.selected_corpus.to_dataframe()
        selected_data_series: Series = df[self.selected_label]
        if self.spacy_filter:
            self.selected_operations = SpacyOperations(self.nlp, selected_data_series)
            self._update_panel()
            return
        elif self.selected_corpus.uses_spacy() and (self.selected_label == self.selected_corpus._COL_DOC):
            self.selected_operations = TextOperations(selected_data_series)
            self._update_panel()
            return

        label_datatype_str: str = str(df.dtypes.get(self.selected_label)).lower()
        try:
            label_datatype: DataType = DataType(label_datatype_str)
            operations_type: Type[Operations] = DATATYPE_OPERATIONS_MAP[label_datatype]
            self.selected_operations = operations_type(selected_data_series)
        except ValueError as e:
            self.selected_operations = DefaultOperations(selected_data_series)

        self._update_panel()

    def resolve_filter(self, data_value: Any) -> bool:
        result: bool = self.selected_operations.call_operation(data_value)
        if self.negation:
            return not result
        return result


class CorpusSlicerParams(param.Parameterized):
    selected_corpus = param.Selector(default=None)

    def __init__(self, nlp: spacy.Language, **params):
        super().__init__(**params)
        self.nlp: spacy.Language = nlp
        self.filters: list = []

        self.add_filter_button = Button(
            name="Add filter",
            button_type="primary", button_style="solid",
            visible=False,
            align='end'
        )
        self.add_filter_button.on_click(self.add_filter)
        self.add_advanced_filter_button = Button(
            name="Add advanced filter",
            button_type="primary", button_style="solid",
            visible=False,
            align='end'
        )
        self.add_advanced_filter_button.on_click(lambda e: self.add_filter(e, True))

        self.control_row = Row(self.param.selected_corpus, self.add_filter_button, self.add_advanced_filter_button)
        self.panel = Column(self.control_row)

    def __panel__(self):
        return pn.panel(self.panel)

    def reset_filters(self):
        self.on_corpus_update()

    def add_filter(self, event=None, advanced: bool = False):
        if advanced:
            new_filter_param = FilterParams(self.selected_corpus, nlp=self.nlp)
        else:
            new_filter_param = FilterParams(self.selected_corpus)
        self.filters.append(new_filter_param)

        new_filter_param.remove_filter_button.on_click(lambda *_, filter_param=new_filter_param, r=new_filter_param.__panel__(): self.remove_filter_row(filter_param, r))

        objects = self.panel.objects
        objects.append(new_filter_param)
        self.panel.objects = objects

    def remove_filter_row(self, filter_param: FilterParams, filter_row: Row, *_):
        self.filters.remove(filter_param)
        objects = [row for row in self.panel.objects if row != filter_row]
        self.panel.objects = objects

    @param.depends('selected_corpus', watch=True)
    def on_corpus_update(self):
        if self.selected_corpus is not None:
            self.add_filter_button.visible = True
            self.add_advanced_filter_button.visible = self.selected_corpus.uses_spacy()
            advanced: bool = True
        else:
            self.add_advanced_filter_button.visible = False
            advanced: bool = False
        self.filters = []
        self.panel.objects = []
        self.panel.objects = [self.control_row]
        self.add_filter(advanced=advanced)


@Language.component("gather_attributes")
def gather_attributes(doc: Doc):
    if not len(doc):
        doc._.attr_vals = {}
        doc._.custom_attr_vals = {}
        return doc

    attribute_values: dict[str, set[str]] = {}
    for attr in dir(doc[0]):
        is_method = callable(getattr(doc[0], attr))
        if (not attr.startswith('_')) and (not is_method):
            attribute_values[attr] = set()
    custom_attribute_values: dict[str, set[str]] = {}
    for custom_attr in dir(doc[0]._):
        is_method = callable(getattr(doc[0]._, custom_attr))
        if not is_method:
            custom_attribute_values[custom_attr] = set()

    for tok in doc:
        for attr in attribute_values.keys():
            value = getattr(tok, attr)
            try:
                attribute_values[attr].add(value)
            except TypeError:
                pass
        for attr in custom_attribute_values.keys():
            value = getattr(tok._, attr)
            try:
                custom_attribute_values[attr].add(value)
            except TypeError:
                pass

    doc._.attr_vals = attribute_values
    doc._.custom_attr_vals = custom_attribute_values
    return doc


class CorpusSlicer(pn.viewable.Viewer):
    LOGGER_NAME: str = "corpus-slicer"

    @staticmethod
    def setup_logger(logger_name: str, run_logger: bool):
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        if not run_logger:
            logger.addHandler(logging.NullHandler())
            return

        formatter = logging.Formatter('%(asctime)s %(levelname)6s - %(name)s:%(lineno)4d %(funcName)20s() - %(message)s')
        log_file_location = abspath(join(dirname(__file__), '..', 'log.txt'))
        # Max size is ~10MB with 1 backup, so a max size of ~20MB for log files
        max_bytes: int = 10000000
        backup_count: int = 1
        file_handler = RotatingFileHandler(log_file_location, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        logger.info('Logger started')

    @staticmethod
    def log(msg: str, level: int):
        logger = logging.getLogger(CorpusSlicer.LOGGER_NAME)
        logger.log(level, msg)

    @staticmethod
    def setup_nlp(nlp: Union[str, spacy.Language]):
        if isinstance(nlp, str):
            nlp = spacy.load(nlp)
        if not isinstance(nlp, spacy.Language):
            raise TypeError(f"Expected supplied nlp to be Spacy Language, instead got {type(nlp)}")

        if Doc.has_extension("attr_vals"):
            Doc.remove_extension("attr_vals")
        if Doc.has_extension("custom_attr_vals"):
            Doc.remove_extension("custom_attr_vals")
        if nlp.has_pipe("gather_attributes"):
            nlp.remove_pipe("gather_attributes")
        Doc.set_extension("attr_vals", default=None)
        Doc.set_extension("custom_attr_vals", default=None)
        nlp.add_pipe("gather_attributes", last=True)

        return nlp

    def __init__(self,
                 root_directory: str = './',
                 nlp: Union[str, spacy.Language] = "en_core_web_sm",
                 run_logger: bool = False,
                 **params):
        super().__init__(**params)

        CorpusSlicer.setup_logger(CorpusSlicer.LOGGER_NAME, run_logger)
        self.nlp = self.setup_nlp(nlp)

        self.progress_bar = Tqdm(visible=False)
        self.progress_bar.pandas()
        self.slicer_params = CorpusSlicerParams(nlp=self.nlp)

        self.run_spacy_button = Button(
            name="Convert to advanced corpus",
            button_type="primary", button_style="solid",
            visible=False,
            align="start"
        )
        self.run_spacy_button.on_click(self.run_spacy)

        self.slice_corpus_button = Button(
            name="Slice",
            button_type="success", button_style="solid",
            height=30, width=100,
            visible=False,
            align="end"
        )
        self.slice_corpus_button.on_click(self.slice_corpus)

        self.sliced_name_field = TextInput(name='Name', placeholder='Enter a name (leave blank to autogenerate)',
                                           visible=False)

        self.slicer_panel = pn.panel(pn.Column(self.slicer_params,
                                               self.progress_bar,
                                               self.run_spacy_button,
                                               Row(self.slice_corpus_button,
                                                   self.sliced_name_field),
                                               height=500))

        self.corpus_loader: CorpusLoader = CorpusLoader(root_directory)
        self.corpora = self.corpus_loader.get_mutable_corpora()

        self.corpus_loader.register_event_callback(EventType.BUILD, self.on_corpora_update)
        self.corpus_loader.register_event_callback(EventType.RENAME, self.on_corpora_update)
        self.corpus_loader.register_event_callback(EventType.DELETE, self.on_corpora_update)
        self.on_corpora_update()
        self.corpus_loader.add_tab("Corpus Slicer", self.slicer_panel)

    def __panel__(self):
        return self.corpus_loader

    def get_corpus_loader(self) -> CorpusLoader:
        return self.corpus_loader

    def get_mutable_corpora(self) -> BaseCorpora:
        return self.corpora

    def display_error(self, error_msg: str):
        self.log(f"Error displayed: {error_msg}", logging.DEBUG)
        pn.state.notifications.error(error_msg, duration=0)

    def display_success(self, success_msg: str):
        self.log(f"Success displayed: {success_msg}", logging.DEBUG)
        pn.state.notifications.success(success_msg, duration=3000)

    def set_corpus_selector_value(self, corpus_dict: dict[str, DataFrameCorpus]):
        formatted_dict: dict[str, DataFrameCorpus] = {}
        for name, corpus in corpus_dict.items():
            label = f"{name} | docs: {len(corpus)}"
            if corpus.parent:
                label += f" | parent: {corpus.parent.name}"
            formatted_dict[label] = corpus
        self.slicer_params.param.selected_corpus.objects = formatted_dict
        if len(corpus_dict):
            self.slicer_params.selected_corpus = list(corpus_dict.values())[-1]
            corpus_exists = True
            corpus_uses_spacy = self.slicer_params.selected_corpus.uses_spacy()
        else:
            self.slicer_params.selected_corpus = None
            corpus_exists = False
            corpus_uses_spacy = False

        self.slice_corpus_button.visible = corpus_exists
        self.sliced_name_field.visible = corpus_exists
        self.run_spacy_button.visible = corpus_exists
        self.run_spacy_button.disabled = corpus_exists and corpus_uses_spacy

        self.slicer_params.on_corpus_update()

    def on_corpora_update(self, *_):
        if self.corpus_loader is None:
            return
        corpus_dict: dict[str, DataFrameCorpus] = {}
        corpora_list: list = self.corpora.items()
        for corpus in corpora_list:
            corpus_dict[corpus.name] = corpus

        self.set_corpus_selector_value(corpus_dict)

    def run_spacy(self, *_):
        try:
            self.run_spacy_button.button_style = "outline"
            if self.slicer_params.selected_corpus is not None:
                self.slicer_params.selected_corpus.run_spacy(self.nlp)
        except Exception as e:
            self.run_spacy_button.button_style = "solid"
            self.log(traceback.format_exc(), logging.ERROR)
            self.display_error(f"Error converting corpus: {e}")
            return

        try:
            self.slicer_params.on_corpus_update()
        except Exception as e:
            self.log(traceback.format_exc(), logging.ERROR)
            self.display_error(f"Error converting corpus: {e}")
        self.run_spacy_button.button_style = "solid"
        self.run_spacy_button.disabled = True
        self.display_success("Corpus converted to advanced successfully")

    def slice_corpus(self, *_):
        new_name = self.sliced_name_field.value_input

        try:
            self.progress_bar.visible = True
            self.slice_corpus_button.button_style = "outline"
            corpus: DataFrameCorpus = self.slicer_params.selected_corpus
            corpus_df: DataFrame = corpus.to_dataframe()

            mask = Series([True] * len(corpus.find_root()))
            for filter_param in self.slicer_params.filters:
                selected_label: str = filter_param.selected_label
                selected_series: Series = corpus_df[selected_label]
                cond_func: Callable = filter_param.resolve_filter

                filter_mask = selected_series.progress_apply(cond_func)
                filter_bool_mask = filter_mask.astype('bool')
                mask = filter_bool_mask & mask
            if len(new_name):
                sliced_corpus = corpus.cloned(mask, new_name)
            else:
                sliced_corpus = corpus.cloned(mask)
            self.corpora.add(sliced_corpus)

            self.on_corpora_update()
            self.slicer_params.reset_filters()

            self.slice_corpus_button.button_style = "solid"
            self.progress_bar.visible = False
            self.sliced_name_field.value = ""
            self.display_success("Corpus sliced successfully")
        except Exception as e:
            self.slice_corpus_button.button_style = "solid"
            self.progress_bar.visible = False
            self.log(traceback.format_exc(), logging.ERROR)
            self.display_error(f"Error slicing corpus: {e}")
