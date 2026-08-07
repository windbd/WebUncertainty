"""Task Uncertainty-Driven Adaptive Planning: the real-browser outer loop.

:class:`~WebUncertainty.planning.adaptive_planner.AdaptivePlanner` is the
only place that owns a live ``BrowserSession`` for this agent stack and
decides how far ahead to plan; it delegates every action's evidential
reasoning to :mod:`WebUncertainty.mcts` and executes exactly the one action
that search settles on each step.
"""
