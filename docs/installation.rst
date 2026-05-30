Installation
============

Requirements: Python 3.9+

Base install (causal + finance + utils, no deep learning)::

    git clone https://github.com/mofs0/empirlab.git
    cd empirlab
    pip install -e .

With deep learning (PyTorch)::

    pip install -e ".[dl]"

With reinforcement learning (Gymnasium)::

    pip install -e ".[rl]"

With LLM tools (OpenAI + LangChain + FAISS)::

    pip install -e ".[llm]"

Everything::

    pip install -e ".[full]"

Building these docs
-------------------

::

    pip install sphinx furo sphinx-autodoc-typehints
    cd docs
    make html
    open _build/html/index.html
