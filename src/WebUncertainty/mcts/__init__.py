"""Action Uncertainty-Driven MCTS Reasoning (Section 3.2 of the paper).

This package is a pure search/scoring layer: nothing here executes a real
browser action. :mod:`WebUncertainty.mcts.sampler` is the one exception in
spirit — it is the adapter that takes the search's chosen root action and
hands it to a real ``BrowserSession``, exactly once per search.

Modules:

- ``uncertainty`` — ConActU: confidence scores -> (EU, AU) -> reward modulation.
- ``tree`` — the search tree's node/candidate data structures and the PUCT score.
- ``search`` — the select / expand / simulate / backpropagate loop.
- ``sampler`` — commits the search's chosen root action to a real browser.
"""
