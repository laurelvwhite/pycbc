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

class SimpleStepUniform(bounded.BoundedDist):
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
    name = 'simplestepuniform'
    def __init__(lwrbnd=None, uprbnd=None, step=None):
        super(SimpleStepUniform, self).__init__(**params)
        self._step = step
        self._lwrbnd = lwrbnd
        self._uprbnd = uprbnd
        self._norm1 = 0.5/(self._step-self._lwrbnd)
        self._norm2 = 0.5/(self._uprbnd - self._step)

    @property
    def norm(self):
        return self._norm1 and self._norm2

    def _pdf(value=None):
        """Returns the pdf at the given value.
        """
        self._value = value
        if self._value < self._step:
            return self._norm1
        else:
            return self._norm2

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
            arr = numpy.zeros(size, dtype=dtype)
            arr1 = numpy.zeros(size/2, dtype=dtype)
            arr2 = numpy.zeros(size/2, dtype=dtype)
            for (p,_) in dtype:
               arr1[p] = numpy.random.uniform(self._bounds[p][0],
                                        self._step,
                                        size=size/2)
               arr2[p] = numpy.random.uniform(self._step,
                                        self._bounds[p][1],
                                        size=size/2)
               arr[p] = numpy.append(arr1[p], arr2[p]) 
            return arr
        else:
            arr1 = numpy.zeros(1, dtype=dtype)
            for (p,_) in dtype:
              random = numpy.random.randint(2)
              if random == 0:
                 arr1[p] = numpy.random.uniform(self.bounds[p][0],
                                        self._step,
                                        size=1)
              elif random == 1:
                 arr1[p] = numpy.random.uniform(self._step,
                                        self.bounds[p][1],
                                        size=1)
            if (size-1) > 0:
               arr2 = numpy.zeros((size-1)/2, dtype=dtype)
               arr3 = numpy.zeros((size-1)/2, dtype=dtype)
               for (p,_) in dtype:
                  arr2[p] = numpy.random.uniform(self._bounds[p][0],
                                        self._step,
                                        size=(size-1)/2)
                  arr3[p] = numpy.random.uniform(self._step,
                                        self._bounds[p][1],
                                        size=(size-1)/2)
               arr = numpy.append(arr1, arr2)
               arr = numpy.append(arr, arr3)
               return arr
            else:
               return arr1

__all__ = ['StepUniform']
