"""Prompt construction for the WebUncertainty agent stack.

Every module here builds plain ``(system_prompt, user_prompt)`` strings from
plain Python values (str/float/list/dict) — never from ``PageObservation``,
``BrowserSession``, or any agent/mcts type. Callers in ``agents/`` render a
page or a tree node down to text first (e.g. ``observation.render()``) and
pass that text in. This keeps prompt wording reviewable and testable without
importing a browser or a model client.
"""
