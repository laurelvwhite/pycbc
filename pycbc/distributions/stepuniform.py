# Copyright (C) 2016  Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
This modules provides classes for evaluating distributions that are uniform in two steps..
"""

import numpy
from pycbc.distributions import bounded

class StepUniform(bounded.BoundedDist):
    """
    A uniform distribution on the given parameters, with 50% of the
    distribution located between the first bound and the step and 50%
    located between the step and the second bound. The parameters are
    independent of each other. Instances of this class can be called like
    a function.

    Parameters
    ----------
    \**params :
        The keyword arguments should provide the names of parameters and their
        corresponding bounds, as either tuples or a `boundaries.Bounds`
        instance.
    step:
        An integer or a float value betwen the provided bounds where the
        distribution should be split.

    Attributes
    ----------
    name : 'stepuniform'
        The name of this distribution.

    Attributes
    ----------
    params : list of strings
        The list of parameter names.
    bounds : dict
        A dictionary of the parameter names and their bounds.
    norm : float
        The normalization of the multi-dimensional pdf.
    step : float
        The point where the distribution is split..

    Examples
    --------
    Create a 2 dimensional step-uniform distribution:

    >>> dist = distributions.StepUniform(mass1=(10.,50.), mass2=(10.,50.), step=20.)

    Generate some random values:

    >>> dist.rvs(size=3)
        array([(36.90885758394699, 51.294212757995254),
               (39.109058546060346, 13.36220145743631),
               (34.49594465315212, 47.531953033719454)],
              dtype=[('mass1', '<f8'), ('mass2', '<f8')])

    """
    name = 'stepuniform'
    def __init__(self, step=None, **params):
        super(StepUniform, self).__init__(**params)
        # compute the norm and save
        # temporarily suppress numpy divide by 0 warning
        self._step = step
        numpy.seterr(divide='ignore')
        self._norm1 = (0.5/(step-bnd[0]) for bnd in self._bounds.values())
        self._norm2 = (0.5/(bnd[1]-step) for bnd in self._bounds.values())
        numpy.seterr(divide='warn')

    @property
    def norm(self):
        return self._norm1 and self._norm2

    def _pdf(self, **kwargs):
        """Returns the pdf at the given values. The keyword arguments must
        contain all of parameters in self's params. Unrecognized arguments are
        ignored.
        """
        if kwargs in self:
            print self, kwargs
            return self._norm1
        else:
            return 0.

    def rvs(self, size=1, param=None):
        """Gives a set of random values drawn from this distribution.

        Parameters
        ----------
        size : {1, int}
            The number of values to generate; default is 1.
        param : {None, string}
            If provided, will just return values for the given parameter.
            Otherwise, returns random values for each parameter.

        Returns
        -------
        structured array
            The random values in a numpy structured array. If a param was
            specified, the array will only have an element corresponding to the
            given parameter. Otherwise, the array will have an element for each
            parameter in self's params.
        """
        if param is not None:
            dtype = [(param, float)]
        else:
            dtype = [(p, float) for p in self.params]
        if size%2 == 0:
           arr1 = numpy.zeros(size/2, dtype=dtype)
           arr2 = numpy.zeros(size/2, dtype=dtype)
           for (p,_) in dtype:
               arr1[p] = numpy.random.uniform(self._bounds[p][0],
                                        self._step,
                                        size=size/2)
               arr2[p] = numpy.random.uniform(self._step,
                                        self._bounds[p][1],
                                        size=size/2)
               arr = numpy.append(arr1, arr2)
        else:
           arr1 = numpy.zeros(1, dtype=dtype)
           random = numpy.random.randint(2)
           if random == 0:
              for (p,_) in dtype:
                 arr1[p] = numpy.random.uniform(self.bounds[p][0],
                                        self._step,
                                        size=1)
           elif random == 1:
              for (p,_) in dtype:
                 arr1[p] = numpy.random.uniform(self._step,
                                        self.bounds[p][1],
                                        size=1)
           arr2 = numpy.zeros((size-1)/2, dtype=dtype)
           arr3 = numpy.zeros((size-1)/2, dtype=dtype)
           if size-1 != 0:
              for (p,_) in dtype:
                  arr1[p] = numpy.random.uniform(self._bounds[p][0],
                                        self._step,
                                        size=(size-1)/2)
                  arr2[p] = numpy.random.uniform(self._step,
                                        self._bounds[p][1],
                                        size=(size-1)/2)
            arr = numpy.append(arr1, arr2, arr3)
         return arr

    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a distribution based on a configuration file. The parameters
        for the distribution are retrieved from the section titled
        "[`section`-`variable_args`]" in the config file.

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            ``VARARGS_DELIM``. These must appear in the "tag" part
            of the section header.

        Returns
        -------
        Uniform
            A distribution instance from the pycbc.inference.prior module.
        """
        return super(StepUniform, cls).from_config(cp, section, variable_args,
                     bounds_required=True)


__all__ = ['StepUniform']
