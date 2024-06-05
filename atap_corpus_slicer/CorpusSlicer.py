import logging
from dataclasses import dataclass
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


@dataclass
class Filter:
    negation: bool
    selected_label: str
    operation: Callable


class CorpusSlicerParams(param.Parameterized):
    selected_corpus = param.Selector(default=None)

    negation = param.Boolean(label='Not', default=False)
    selected_label = param.Selector(label='Data label', default=None)

    def __init__(self, **params):
        super().__init__(**params)
        self.selected_operations = DefaultOperations()
        self.num_filters: int = 1

        self.panel = Column(align='center')
        self._update_panel()

    def __panel__(self):
        return pn.panel(self.panel)

    def _update_panel(self):
        self.panel.objects = [
            self.param.selected_corpus,
            Row(self.param.negation,
                self.param.selected_label,
                self.selected_operations)
        ]

    @param.depends('selected_corpus', watch=True)
    def on_corpus_update(self):
        label_list: list[str] = []
        if self.selected_corpus is not None:
            df: DataFrame = self.selected_corpus.to_dataframe()
            label_list = df.columns
        self.param.selected_label.objects = label_list
        if len(label_list):
            self.selected_label = label_list[0]

    @param.depends('selected_label', watch=True)
    def _set_operations(self):
        logger = logging.getLogger(__name__)
        if self.selected_corpus is None:
            return
        df: DataFrame = self.selected_corpus.to_dataframe()
        label_datatype_str: str = str(df.dtypes.get(self.selected_label)).lower()
        try:
            label_datatype: DataType = DataType(label_datatype_str)
            operations_type: Type[Operations] = DATATYPE_OPERATIONS_MAP[label_datatype]
        except ValueError as e:
            logger.debug(f"{type(e)}: {str(e)}")
            self.selected_operations = DefaultOperations()
            return

        self.selected_operations = operations_type()
        logger.debug(f"Operation selected: {type(self.selected_operations)}")
        self._update_panel()

    @param.depends('negation', 'selected_operations.call_operation')
    def resolve_filter(self, data_value: Any) -> bool:
        result: bool = self.selected_operations.call_operation(data_value)
        if self.negation:
            return not result
        return result


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

    def __init__(self, corpora: Optional[BaseCorpora] = None, **params):
        super().__init__(**params)

        CorpusSlicer.setup_logger()

        self.progress_bar = Tqdm(visible=False)
        self.progress_bar.pandas()
        self.slicer_params = CorpusSlicerParams()

        self.corpus_export_button = FileDownload(
            filename=f"sliced-corpus.csv",
            callback=self.export_csv,
            visible=False,
            button_type="primary",
            button_style="solid")

        self.slice_corpus_button = Button(
            name="Slice",
            button_type="success", button_style="solid",
            height=30, width=100,
            align="center"
        )
        self.slice_corpus_button.on_click(self.slice_corpus)

        self.panel = pn.Column(self.slicer_params,
                               self.progress_bar,
                               Row(self.slice_corpus_button, self.corpus_export_button))

        self.sliced_corpus: Optional[DataFrameCorpus] = None
        self.corpus_loader: Optional[CorpusLoader] = None
        if corpora is None:
            self.corpus_loader = CorpusLoader("corpus_data")
            self.corpus_loader.set_build_callback(self.on_corpora_update)
            self.on_corpora_update()
            self.panel.objects = [self.corpus_loader] + self.panel.objects
        else:
            corpus_dict: dict[str, DataFrameCorpus] = {corpus.name: corpus for corpus in corpora.items()}
            self.set_corpus_selector_value(corpus_dict)

    def __panel__(self):
        return pn.panel(self.panel)

    def set_corpus_selector_value(self, corpus_dict: dict[str, DataFrameCorpus]):
        formatted_dict: dict[str, DataFrameCorpus] = {f"{name} | docs: {len(c)} | parent: {c.parent}": c for name, c in corpus_dict.items()}
        self.slicer_params.param.selected_corpus.objects = formatted_dict
        if len(corpus_dict):
            self.slicer_params.selected_corpus = list(corpus_dict.values())[-1]
        self.slicer_params.on_corpus_update()

    def on_corpora_update(self, *_):
        if self.corpus_loader is None:
            return
        corpus_dict: dict[str, DataFrameCorpus] = self.corpus_loader.get_corpora()
        self.set_corpus_selector_value(corpus_dict)

    def slice_corpus(self, *_):
        try:
            corpus: DataFrameCorpus = self.slicer_params.selected_corpus
            corpus_df: DataFrame = corpus.to_dataframe()
            selected_label: str = self.slicer_params.selected_label

            selected_series: Series = corpus_df[selected_label]
            cond_func: Callable = self.slicer_params.resolve_filter

            mask = selected_series.progress_apply(cond_func)
            root_mask = (Series([True] * len(corpus.find_root())) & mask)
            self.sliced_corpus = corpus.cloned(root_mask)

            self.corpus_export_button.visible = True
        except BaseException as e:
            self.LOGGER.debug(f"{type(e)}: {str(e)}")

    def get_sliced_corpus(self) -> Optional[DataFrameCorpus]:
        return self.sliced_corpus

    def export_csv(self) -> Optional[BytesIO]:
        if self.sliced_corpus is None:
            return

        csv_object = BytesIO()
        if len(self.sliced_corpus) == 0:
            return csv_object

        df: DataFrame = self.sliced_corpus.to_dataframe()
        chunks = np.array_split(df.index, min(len(df), 1000))
        self.progress_bar.visible = True
        with self.progress_bar(total=len(df), desc="Exporting to CSV", unit="documents", leave=False) as pbar:
            df.loc[chunks[0]].to_csv(csv_object, mode='w', index=False)
            pbar.update(len(chunks[0]))
            for chunk, subset in enumerate(chunks[1:]):
                df.loc[subset].to_csv(csv_object, header=False, mode='a', index=False)
                pbar.update(len(subset))
        csv_object.seek(0)
        self.progress_bar.visible = False

        return csv_object
