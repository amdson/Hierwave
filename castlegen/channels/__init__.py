"""Channels: the generic layer of notes/channels.tex (plan: notes/channels_demo.md).

A model is a set of channels (a grid of values at one level each, with views
of its domain), a list of factors (tables over views, at offsets), and at
most one certificate per level-1 channel.  Kernels read factors through the
packed descriptors of core.Model.compile; channel sets (ground.py, roots.py)
never see a kernel, and a kernel never sees a channel set."""
from .core import Channel, Factor, Certificate, Model, Kinds, view_of_tags  # noqa: F401
from . import ground, roots, coord  # noqa: F401
