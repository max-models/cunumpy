"""The scope-profiler tag of cunumpy's own profiling regions.

``cunumpy/__init__.py`` registers it as opt-in, so the regions record nothing
unless the application enables the ``"cunumpy"`` tag (see
:doc:`/guides/profiling`).
"""

#: Tags of every cunumpy region (``@sp.profile("cunumpy.<op>", tags=_PROFILE_TAGS)``).
_PROFILE_TAGS = ("cunumpy",)
