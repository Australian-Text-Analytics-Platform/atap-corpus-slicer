import logging
import traceback
from io import BytesIO
from logging.handlers import RotatingFileHandler
from os.path import join, dirname, abspath
from typing import Optional, Callable, Any, Type

import numpy as np
import panel as pn
import param
from atap_corpus.corpus.base import BaseCorpora
from atap_corpus.corpus.corpus import DataFrameCorpus
from pandas import DataFrame, Series
from panel import Row, Column

from panel.theme import Fast

from atap_corpus_loader import CorpusLoader
from panel.widgets import FileDownload, Tqdm, Button

from atap_corpus_slicer.Operation import DefaultOperations, DATATYPE_OPERATIONS_MAP, Operations, DataType

pn.extension(notifications=True, design=Fast)


class FilterParams(param.Parameterized):
    negation = param.Boolean(label='Negate', default=False, instantiate=True)
    selected_label = param.Selector(label='Data label', instantiate=True)

    def __init__(self, selected_corpus: DataFrameCorpus, **params):
        super().__init__(**params)
        self.selected_operations = DefaultOperations(Series())

        self.remove_filter_button = Button(
            name="Remove",
            button_type="warning", button_style="outline"
        )

        self.panel = Row()

        self.selected_corpus: DataFrameCorpus = selected_corpus
        self.update_corpus(selected_corpus)

    def __panel__(self):
        return self.panel

    def _update_panel(self):
        self.panel.objects = [self.param.selected_label,
                              self.selected_operations,
                              self.param.negation,
                              self.remove_filter_button]

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
        logger = logging.getLogger(__name__)
        if self.selected_corpus is None:
            return
        df: DataFrame = self.selected_corpus.to_dataframe()
        selected_data_series: Series = df[self.selected_label]
        label_datatype_str: str = str(df.dtypes.get(self.selected_label)).lower()
        try:
            label_datatype: DataType = DataType(label_datatype_str)
            operations_type: Type[Operations] = DATATYPE_OPERATIONS_MAP[label_datatype]
        except ValueError as e:
            logger.debug(f"{type(e)}: {str(e)}")
            self.selected_operations = DefaultOperations(selected_data_series)
            return

        self.selected_operations = operations_type(selected_data_series)
        self._update_panel()

    def resolve_filter(self, data_value: Any) -> bool:
        result: bool = self.selected_operations.call_operation(data_value)
        if self.negation:
            return not result
        return result


class CorpusSlicerParams(param.Parameterized):
    selected_corpus = param.Selector(default=None)

    def __init__(self, **params):
        super().__init__(**params)
        self.filters: list = []

        self.add_filter_button = Button(
            name="Add filter",
            button_type="primary", button_style="solid",
            visible=False,
            align='end'
        )
        self.add_filter_button.on_click(self.add_filter)

        self.control_row = Row(self.param.selected_corpus, self.add_filter_button)
        self.panel = Column(self.control_row)

    def __panel__(self):
        return pn.panel(self.panel)

    def reset_filters(self):
        self.on_corpus_update()

    def add_filter(self, *_):
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
        self.filters = []
        self.panel.objects = [self.control_row]
        self.add_filter()


class CorpusSlicer(pn.viewable.Viewer):
    LOGGER: logging.Logger = None

    @staticmethod
    def setup_logger():
        if CorpusSlicer.LOGGER is not None:
            return
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file_location = abspath(join(dirname(__file__), '..', 'log.txt'))
        # Max size is ~10MB with 1 backup, so a max size of ~20MB for log files
        max_bytes: int = 10000000
        backup_count: int = 1
        file_handler = RotatingFileHandler(log_file_location, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)

        CorpusSlicer.LOGGER = logging.getLogger(__name__)
        CorpusSlicer.LOGGER.setLevel(logging.DEBUG)
        CorpusSlicer.LOGGER.addHandler(file_handler)
        CorpusSlicer.LOGGER.addHandler(console_handler)

        CorpusSlicer.LOGGER.info('Logger started')

    def __init__(self, root_directory: str = './', **params):
        super().__init__(**params)

        CorpusSlicer.setup_logger()

        self.progress_bar = Tqdm(visible=False)
        self.progress_bar.pandas()
        self.slicer_params = CorpusSlicerParams()

        self.slice_corpus_button = Button(
            name="Slice",
            button_type="success", button_style="solid",
            height=30, width=100,
            visible=False,
            align="start"
        )
        self.slice_corpus_button.on_click(self.slice_corpus)

        self.slicer_panel = pn.panel(pn.Column(self.slicer_params,
                                      self.progress_bar,
                                      self.slice_corpus_button,
                                      height=500))

        self.corpus_loader: CorpusLoader = CorpusLoader(root_directory)
        self.corpus_loader.set_build_callback(self.on_corpora_update)
        self.on_corpora_update()
        self.corpus_loader.add_tab("Corpus Slicer", self.slicer_panel)
        self.corpora: BaseCorpora = self.corpus_loader.get_mutable_corpora()

    def __panel__(self):
        return self.corpus_loader

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
            self.slice_corpus_button.visible = True
        self.slicer_params.on_corpus_update()

    def on_corpora_update(self, *_):
        if self.corpus_loader is None:
            return
        corpus_dict: dict[str, DataFrameCorpus] = self.corpus_loader.get_corpora()
        self.set_corpus_selector_value(corpus_dict)

    def slice_corpus(self, *_):
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

            sliced_corpus = corpus.cloned(mask)

            self.corpora.add(sliced_corpus)

            self.on_corpora_update()
            self.slicer_params.reset_filters()

            self.slice_corpus_button.button_style = "solid"
            self.progress_bar.visible = False
        except Exception:
            self.LOGGER.error(traceback.format_exc())
            self.slice_corpus_button.button_style = "solid"
            self.progress_bar.visible = False
