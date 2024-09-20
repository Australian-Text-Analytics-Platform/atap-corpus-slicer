import logging
import traceback
from logging.handlers import RotatingFileHandler
from os.path import join, dirname, abspath
from typing import Callable, Any, Type, Optional, Union

import panel as pn
import param
import spacy
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

from atap_corpus_slicer.Operation import DefaultOperations, SpacyTokenOperations, DATATYPE_OPERATIONS_MAP, Operations, \
    DataType, TextOperations, SpacyPhraseOperations

pn.extension(notifications=True, design=Fast)
tqdm.pandas()


class FilterParams(param.Parameterized):
    negation = param.Boolean(label='Negate', default=False, instantiate=True)
    selected_label = param.Selector(label='Data label', instantiate=True)

    def __init__(self, selected_corpus: DataFrameCorpus, filter_type: str = "simple", model: Optional[Language] = None, **params):
        super().__init__(**params)
        self.model: Optional[Language] = model
        self.selected_operations = DefaultOperations(Series())
        self.filter_type: str = filter_type
        self.spacy_filter: bool = (filter_type != "simple") and selected_corpus.uses_spacy()

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
            # Label selector is unnecessary for spacy filters as they only operate on the document data
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
        if self.filter_type == "spacy_token":
            self.selected_operations = SpacyTokenOperations(selected_data_series)
            self._update_panel()
            return
        elif self.filter_type == "spacy_phrase":
            self.selected_operations = SpacyPhraseOperations(selected_data_series, self.model)
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

    def __init__(self, model: Optional[Language], **params):
        super().__init__(**params)
        self.model: Optional[Language] = model

        self.filters: list = []

        self.add_filter_button = Button(
            name="Add filter",
            button_type="primary", button_style="solid",
            visible=False,
            align='end'
        )
        self.add_filter_button.on_click(self.add_filter)
        self.add_spacy_token_filter_button = Button(
            name="Add token filter",
            button_type="primary", button_style="solid",
            visible=False,
            align='end'
        )
        self.add_spacy_token_filter_button.on_click(lambda e: self.add_filter(e, filter_type="spacy_token"))
        self.add_spacy_phrase_filter_button = Button(
            name="Add phrase filter",
            button_type="primary", button_style="solid",
            visible=False,
            align='end'
        )
        self.add_spacy_phrase_filter_button.on_click(lambda e: self.add_filter(e, filter_type="spacy_phrase", model=self.model))

        self.control_row = Row(self.param.selected_corpus, self.add_filter_button,
                               self.add_spacy_token_filter_button, self.add_spacy_phrase_filter_button)
        self.panel = Column(self.control_row)

    def __panel__(self):
        return pn.panel(self.panel)

    def reset_filters(self):
        self.on_corpus_update()

    def add_filter(self, event=None, filter_type: str = "simple", model: Optional[Language] = None):
        new_filter_param = FilterParams(self.selected_corpus, filter_type=filter_type, model=model)
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
            self.add_spacy_token_filter_button.visible = self.selected_corpus.uses_spacy()
            self.add_spacy_phrase_filter_button.visible = self.selected_corpus.uses_spacy()
        else:
            self.add_filter_button.visible = False
            self.add_spacy_token_filter_button.visible = False
            self.add_spacy_phrase_filter_button.visible = False
        self.filters = []
        self.panel.objects = []
        self.panel.objects = [self.control_row]
        self.add_filter()


class SpacyLabeller:
    @staticmethod
    def label_spacy_attributes(corpus: DataFrameCorpus):
        if not corpus.uses_spacy():
            return
        SpacyLabeller.add_extensions()
        for doc in corpus:
            SpacyLabeller.gather_attributes(doc)

    @staticmethod
    def add_extensions():
        if not Doc.has_extension("attr_vals"):
            Doc.set_extension("attr_vals", default=None)
        if not Doc.has_extension("custom_attr_vals"):
            Doc.set_extension("custom_attr_vals", default=None)

    @staticmethod
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
    """
    A GUI tool for applying filters to a corpus based on its data and metadata
    """
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

    def __init__(self,
                 corpus_loader: Optional[CorpusLoader] = None,
                 run_logger: bool = False,
                 model: Optional[Union[str, Language]] = None,
                 **params):
        """
        CorpusSlicer constructor
        :param corpus_loader: The CorpusLoader that the slicer will be attached to. If None, a default CorpusLoader will be created with no optional features. None by Default.
        :type corpus_loader: Optional[CorpusLoader]
        :param run_logger: If True, a log file will be written to. False by default.
        :type run_logger: bool
        :param model: The spaCy Language or name of the Language that will be used to create a spaCy corpus. If the model argument is not None, the corpus will be converted to a spaCy corpus after being built. If the model argument is a string, then a download of the model through spaCy will be attempted (if not already installed) before loading it as a pipeline. None by default.
        :type model: Optional[Union[str, Language]]
        :param params: Additional parameters that are passed to the Viewer super class
        :type params: Any
        """
        super().__init__(**params)

        CorpusSlicer.setup_logger(CorpusSlicer.LOGGER_NAME, run_logger)

        self.model: Optional[Language]
        if (model is None) or isinstance(model, Language):
            self.model = model
        elif isinstance(model, str):
            try:
                self.model = spacy.load(model)
            except OSError:
                spacy.cli.download(model)
                self.model = spacy.load(model)
        else:
            raise TypeError(f"Expected model argument to be either a spacy Language or a string. Instead got {type(model)}")

        self.progress_bar = Tqdm(visible=False)
        self.slicer_params = CorpusSlicerParams(self.model)

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
                                               Row(self.slice_corpus_button,
                                                   self.sliced_name_field),
                                               height=500))

        if corpus_loader:
            self.corpus_loader: CorpusLoader = corpus_loader
        else:
            self.corpus_loader: CorpusLoader = CorpusLoader(root_directory='.', run_logger=run_logger)
        self.corpora = self.corpus_loader.get_mutable_corpora()

        if self.model is not None:
            self.corpus_loader.register_event_callback(EventType.BUILD, self.corpus_run_spacy)
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

    def corpus_run_spacy(self, corpus):
        # This method would ideally simply call corpus.run_spacy(self.model),
        # but in order to display the progress bar correctly the method must be recreated here.
        try:
            if corpus.uses_spacy():
                return
            run_spacy_on: DataFrameCorpus = corpus.find_root()
            docs = (d for d in run_spacy_on.docs())

            progress_bar = self.corpus_loader.controller.get_build_progress_bar()
            docs_ls = []
            for doc in progress_bar(self.model.pipe(docs), total=len(run_spacy_on), desc="Processing with NLP model", unit="files", leave=True):
                docs_ls.append(doc)
            run_spacy_on._df[run_spacy_on._COL_DOC] = Series(docs_ls)
        except Exception as e:
            self.log(traceback.format_exc(), logging.ERROR)
            self.display_error(f"Error processing files: {e}")

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
        else:
            self.slicer_params.selected_corpus = None
            corpus_exists = False

        self.slice_corpus_button.visible = corpus_exists
        self.sliced_name_field.visible = corpus_exists

        self.slicer_params.on_corpus_update()

    def on_corpora_update(self, corpus=None, *_):
        if self.corpus_loader is None:
            return
        if (corpus is not None) and (corpus.uses_spacy()):
            SpacyLabeller.label_spacy_attributes(corpus)
        corpus_dict: dict[str, DataFrameCorpus] = {}
        corpora_list: list = self.corpora.items()
        for corpus in corpora_list:
            corpus_dict[corpus.name] = corpus

        self.set_corpus_selector_value(corpus_dict)

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
