# Copyright 2023 Matthew Lock.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Simple MPC controller for the Crazyflie 2.1 in Webots. """

from casadi import *
from casadi import Opti as Opti
import time

import numpy as np

class MPC():
    
    def __init__(
            self, model, 
            x0 : np.array, 
            x_ref_states : np.array,
            x_upper_bound : np.array,
            x_lower_bound : np.array,
            u_upper_bound : np.array,
            u_lower_bound : np.array,
            N = 10, 
            u_n = 3,
            logger = None
        ):
        """ 
        Initialize the MPC controller.
        
        Parameters
        ----------
        model : dynamic model
            The dynamic model of the system.    
        x0 : np.array
            Initial state of the system.
        x_ref : np.array
            Reference trajectory.
        x_ref_states : np.array
            Reference trajectory states (with respect to the state vector)
        x_upper_bound : np.array
            Upper bound on the states.
        x_lower_bound : np.array
            Lower bound on the states.
        u_upper_bound : np.array    
            Upper bound on the inputs.
        u_lower_bound : np.array
            Lower bound on the inputs.
        N : int
            Prediction horizon length.
        u_n : int
            Number of inputs.            
        """
        
        assert isinstance(x0, np.ndarray), "x0 must be a numpy array."
        
        self.logger = logger
        
        self.model = model
        self.x0 = x0
        self.N = N
        self.x_n = x0.shape[0]  # number of states
        self.last_sol = None
        
        ## Create optimization problem
        self.opistack           = Opti('conic')
        self.x                  = self.opistack.variable(self.x_n, N+1)
        self.u                  = self.opistack.variable(u_n, N)
        self.x_ref_slack        = self.opistack.variable(len(x_ref_states), N)
        self.x_ref_parameter    = self.opistack.parameter(len(x_ref_states), N+1)
        self.x_0_parameter      = self.opistack.parameter(self.x_n, 1)
        
        # # Add constraints
        self.opistack.subject_to(self.x[:,0] == self.x_0_parameter)                                 # Initial state constraint
        self.opistack.subject_to(self.opistack.bounded(x_lower_bound, self.x, x_upper_bound))       # State constraints
        self.opistack.subject_to(self.opistack.bounded(u_lower_bound, self.u, u_upper_bound))       # Input constraints
        
        for i in range(N):
            self.opistack.subject_to(self.x[:,i+1] == self.model(self.x[:,i], self.u[:,i]))         # Dynamics constraints
            slack_var = 0
            for j in range(len(x_ref_states)):
                self.opistack.subject_to(self.x_ref_slack[slack_var,i] == self.x[j,i+1] - self.x_ref_parameter[j,i+1])
                slack_var += 1
                
        self.opistack.minimize(
            sumsqr(self.x_ref_slack) + 0*sumsqr(self.u) 
        )
        jit_options = {"flags": ["-O3"], "verbose": True}
        # options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        solver_ipopt_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        # solver_ipopt_opts = {
        #     'ipopt.print_level': 0, 
        #     'print_time': 0, 
        #     'ipopt.sb': 'yes',
        #     "jit": True, "compiler": "shell", "jit_options": jit_options
        # }
        self.opistack.solver('qpoases')
        

    def set_initial_state(self, x0):
        """ Set the initial state of the system. """
        self.opistack.set_value(self.x_0_parameter, x0)
        
    def set_reference_trajectory(self, x_ref):
        """ Set the reference trajectory. """
        self.opistack.set_value(self.x_ref_parameter, x_ref)
        
    def warm_start(self, solution):
        """ Warm start the solver. """
        self.opistack.set_initial(solution.value_variables())
        
    def solve(self, verbose=False):
        """ Solve the optimization problem. """
        if self.last_sol is not None:
            # self.logger.info("Warm start!!")
            self.warm_start(self.last_sol)     
        start_time = time.time()
        self.last_sol = self.opistack.solve()
        solve_time = time.time() - start_time
        if verbose and self.logger is not None:
            solve_hz = 1/solve_time
            self.logger.info("MPC solve time: {}".format(solve_time))
            self.logger.info("MPC solve frequency: {}".format(solve_hz))
            # self.logger.info("COST : {}".format(self.last_sol.value(self.opistack.f)))
        x = self.last_sol.value(self.x)
        u = self.last_sol.value(self.u)
        
        return x, u