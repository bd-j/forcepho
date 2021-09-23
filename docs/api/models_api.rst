forcepho.model
================
Classes in the :py:mod:`forcepho.model` module are used to transform between
constrained and unconstrained parameter spaces, apply priors, and to organize
and compute posterior probabilities and their gradients for use in sampling or
optimization tasks.

They generally require a :py:class:`forcepho.patches.Patch` instance and a
:py:class:`forcepho.sources.Scene` instance as attributes or inputs. Transforms
and priors are optional.

.. automodule:: forcepho.model
   :members: Posterior, GPUPosterior, BoundedTransform
