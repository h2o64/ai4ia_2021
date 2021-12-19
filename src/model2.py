# MIT License

# Copyright (c) 2021 Louis Grenioux

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Libraries
from scipy import signal, optimize, fftpack
from typing import List
import json
import numpy as np
import os

# IMPORT THE MODEL API FROM WHICH YOUR MODEL MUST INHERITATE : 
try:
    from model_api import ModelApi
except:pass
try:
    from utilities.model_api import ModelApi
except:pass
try:
    from sources.utilities.model_api import ModelApi
except:pass

class RC2Model(ModelApi):

    def __init__(self, init_params=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)):
        # Save kwargs
        self.model_kwargs = {
            'init_params' : init_params,
        }
        # Initial parameters for the RC filter
        self.init_params = np.array(init_params)

    def filter_low_pass_order2(self, f, a):
      z = 2.0 * np.pi * f * 1j
      return ((a[0] + a[1] * z) / (a[2] + a[3] * z + a[4] * (z*z)))

    def apply_filter(self, X, filter_args, f=None, a=None):
      if f is None:
        f = fftpack.fftfreq(X.shape[0])
      if a is None:
        a = fftpack.fft(X)
      a_out = a * self.filter_low_pass_order2(f, filter_args)
      return fftpack.ifft(a_out)
    
    def pred(self, X, params, f=None, a=None):
      return params[5] * self.apply_filter(X, params[:5], f=f, a=a).real + params[6]

    def fit(self, xs: List[np.ndarray], ys: List[List[np.ndarray]], timeout=36000):
        self.num_outputs = len(ys)
        self.best_params = np.empty((self.num_outputs, self.init_params.shape[0]))
        for j in range(self.num_outputs):
          # Run the optimizer on MSE
          f = fftpack.fftfreq(xs[0].shape[0])
          a = fftpack.fft(xs[0])
          ret_rc = optimize.minimize(lambda params : np.mean(np.square(self.pred(xs[0], params, f=f, a=a) - ys[j])), self.init_params, method='Powell')
          # Save the best parameters
          self.best_params[j] = ret_rc.x

    @classmethod
    def get_sagemaker_estimator_class(self):
        from sagemaker.sklearn import SKLearn
        return SKLearn

    def predict_timeseries(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([self.pred(x, self.best_params[j]) for j in range(self.num_outputs)]).T

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, 'model_kwargs.json')
        with open(path, 'w') as f:
            json.dump(self.model_kwargs, f)

        path = os.path.join(model_dir, 'model.npy')
        with open(path, 'wb') as f:
          np.save(f, self.best_params)
        
    @classmethod
    def load(cls, model_dir: str):
        path = os.path.join(model_dir, 'model_kwargs.json')
        with open(path, 'r') as f:
            model_kwargs = json.load(f)
            
        my_model = cls(**model_kwargs)

        path = os.path.join(model_dir, 'model.npy')
        with open(path, 'rb') as f:
            my_model.best_params = np.load(f)
            my_model.num_outputs = my_model.best_params.shape[0]

        return my_model

    @classmethod
    def create_model(cls, gpu_available: bool = False, **kwargs):
        return cls(**kwargs)

    @property
    def description(self):
        team_name = 'SimpleModels'
        email = 'louis.grenioux@polytechnique.edu'
        model_name = 'RC2'
        affiliation = 'Télécom SudParis'
        description = 'Order 2 low pass filter'
        technology_stack = 'scipy'
        other_remarks = ''

        return dict(team_name=team_name,
                    email=email,
                    model_name=model_name,
                    description=description,
                    technology_stack=technology_stack,
                    other_remarks=other_remarks,
                    affiliation=affiliation)
