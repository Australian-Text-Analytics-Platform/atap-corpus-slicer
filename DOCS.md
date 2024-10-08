# atap_corpus_slicer Documentation

---

## Docs

### atap_corpus_slicer.CorpusSlicer

A GUI tool for applying filters to a corpus based on its data and metadata

Can be imported using:

```python
from atap_corpus_slicer import CorpusSlicer
```

---

### CorpusSlicer.\_\_init\_\_

CorpusSlicer constructor

Params
- corpus_loader: Optional\[CorpusLoader\] - The CorpusLoader that the slicer will be attached to. If None, a default CorpusLoader will be created with no optional features. None by Default.
- run_logger: bool - If True, a log file will be written to. False by default.
- model: Optional\[Union\[str, Language\]\] - The spaCy Language or name of the Language that will be used to create a spaCy corpus. If the model argument is not None, the corpus will be converted to a spaCy corpus after being built. If the model argument is a string, then a download of the model through spaCy will be attempted (if not already installed) before loading it as a pipeline. None by default.
- params: Any - Additional parameters that are passed to the Viewer super class

Example

```python
nlp = spacy.load('en_core_web_sm')
slicer = CorpusSlicer(run_logger=True, model=nlp)
```

---

### CorpusSlicer.servable

Inherited from panel.viewable.Viewer. Call CorpusSlicer.servable() in a Jupyter notebook context to display the CorpusLoader widget with the CorpusSlicer embedded as a tab.

Example

```python
loader = CorpusSlicer()
loader.servable()
```

---

### CorpusSlicer.get_corpus_loader

Returns the CorpusLoader object used by the CorpusSlicer to build and load the corpus. The CorpusLoader panel is displayed with the CorpusSlicer embedded as a tab.

Returns: CorpusLoader - the CorpusLoader object in which the CorpusSlicer is embedded.

Example

```python
slicer = CorpusSlicer()
loader = slicer.get_corpus_loader()
```

---

### CorpusSlicer.get_mutable_corpora

Returns the corpora object that contains the loaded corpus objects.
This allows adding to the corpora from outside the CorpusSlicer as the object returned is mutable, not a copy.
The Corpora object has a unique name constraint, meaning a corpus object cannot be added to the corpora if another corpus with the same name is already present. The same constraint applies to the rename method of corpus objects added to the corpora.

Returns: TCorpora - the mutable corpora object that contains the loaded corpus objects

Example

```python
slicer = CorpusSlicer()
corpora_object = slicer.get_mutable_corpora()
corpus = corpora_object.get("example")
```


## Example usage

The following snippet could be used as a cell in a Jupyter notebook. Each time the user builds a corpus, the corpus will be piped through the provided spaCy Language.

```python
import spacy
from atap_corpus_slicer import CorpusSlicer
from atap_corpus_loader import CorpusLoader

nlp = spacy.load('en_core_web_sm')
loader = CorpusLoader(root_directory='./corpus_data')
corpus_slicer: CorpusSlicer = CorpusSlicer(corpus_loader=loader, run_logger=True, model=nlp)
corpus_slicer.servable()
```

