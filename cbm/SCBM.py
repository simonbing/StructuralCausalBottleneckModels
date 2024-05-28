"""
Simon Bing, 2023, TU Berlin
"""
from collections.abc import Iterable
import copy
import inspect
import logging

import numpy as np

from cbm import SCBMMechanism, make_iterable

class MacroCausalVar(object):
    """
    Class for macro variables in bottleneck models.
    Only supports jointly Gaussian version for now.

    Attributes:
        parents: list[MacroCausalVar]
            causal parents of a variable
        bottleneck_fcts: list[callable]
            functions that map from respective parents' microstates to bottleneck
            variables
        mechanism: SCBMMechanism
            internal microstate mechanism of the variable
        n: int
            number of internal microstate variables
        value: np.array[sample_size, n]
            sampled values
    """
    def __init__(self, parents, bottleneck_fcts, mechanism, n):
        assert ((isinstance(parents, Iterable) and
                 all(isinstance(parent, MacroCausalVar) for parent in parents))
                or parents is None), "Parents of a variable must be list of " \
                                     "MacroCausalVar objects, or None!"
        self.parents = parents

        assert ((isinstance(bottleneck_fcts, Iterable) and
                 all(inspect.isfunction(bottleneck_fct) for bottleneck_fct in bottleneck_fcts))
                or bottleneck_fcts is None), "Bottleneck functions must be a " \
                                             "list of functions, or None!)"
        self.bottleneck_fcts = bottleneck_fcts # surjective functions
        # (one for each parent) that map micro states to some summary statistic(s)

        assert isinstance(mechanism, SCBMMechanism), "Mechanism must be SCBMMechanism type!"
        self.mechanism = mechanism

        self.n = n

        # Assigned during sampling
        self.value = None


class Intervention(object):
    """
    Intervention class to intervene on SCBMs.
    Only supports hard interventions for now.

    Attributes:
        macro_targets: list[int] or int
            macro variables to target
        micro_targets: list[list[int]] or list[int]
            micro variables in each micro variable to target
        values: list[list] or list
            values that micro variables are set to
    """
    def __init__(self, macro_targets, micro_targets, values):
        # Make all arguments iterable if they are single elements
        macro_targets = make_iterable(macro_targets)
        # If micro_targets is just a list of ints, not a list of lists
        if not any(hasattr(elem, '__len__') for elem in micro_targets):
            micro_targets = [micro_targets]
        if not any(hasattr(elem, '__len__') for elem in values):
            values = [values]

        assert all(len(micro_target) == len(value) for micro_target, value in
                   zip(micro_targets, values)), \
            "Length of all micro_targets and values must match!"

        self.macro_targets = macro_targets
        self.micro_targets = micro_targets
        self.values = values


class SCBM(object):
    """
    Structural Causal Bottleneck Model class.

    Attributes:
        seed: int
            random seed
        rs: RandomState
            random state
        variables: list[MacroCausalVar]
            Macro variables in the correct causal ordering
        intervention_flag: bool
            Indicates whether model has been intervened upon. For sampling.
    """
    def __init__(self, variables, seed=None):
        # Random seed for reproducibility
        self.seed = seed
        self.rs = np.random.RandomState(seed=self.seed)

        self.variables = variables # variables must be defined in correct causal ordering
        assert all(isinstance(var, MacroCausalVar) for var in self.variables), \
            "All SCBM variables must be MacroCausalVar objects!"

        self.intervention_flag = False

    def sample(self, size):
        """
        Draw random samples from the SCBM.

        Args:
            size: int
                Number of samples to draw

        Returns:
            values: list[np.array[size, n_microvars]] with len=n_macrovariables
                Sampled values
        """
        if self.intervention_flag:
            logging.warning("The SCM from which you are sampling has been intervened upon!")

        values = []

        # Loop over variables
        for i, var in enumerate(self.variables):
            # Currently, this sampling works for the langevin gaussian case,
            # might have to adapt it later on

            # Sample from independent Gaussian
            noise = self.rs.multivariate_normal(mean=np.zeros(var.n),
                                                cov=np.eye(var.n), size=size)
            if var.parents is not None:
                # Get bottleneck values
                bottleneck_values = [bottleneck_fct(parent.value) for
                                     parent, bottleneck_fct in
                                     zip(var.parents, var.bottleneck_fcts)]

                value = var.mechanism(noise, *bottleneck_values)
            else: # Leaf nodes
                value = var.mechanism(noise)

            var.value = value

            values.append(value)

        return values

    def intervene(self, iv):
        """
        Perform an intervention on the SCBM.

        Args:
            iv: Intervention
        """
        self.intervention_flag = True

        # Loop over interventions
        for macro_target, target, value in \
                zip(iv.macro_targets, iv.micro_targets, iv.values):
            # Reset sampled values to None
            self.variables[macro_target].value = None
            # Set mechanism
            self.variables[macro_target].mechanism.intervene(target, value)

    def intervent_sample(self, iv, size):
        """
        Intervene and sample from the resulting model. Does not permanently
        alter the underlying model.

        Args:
            iv: Intervention
            size: int
                Number of samples to draw

        Returns:
            values: list[np.array[size, n_microvars]] with len=n_macrovariables
                Sampled values
        """
        # Make copy of SCM to not permanently alter state
        SCBM_copy = copy.deepcopy(self)
        # Intervene
        SCBM_copy.intervene(iv)
        # Sample
        return SCBM_copy.sample(size)
