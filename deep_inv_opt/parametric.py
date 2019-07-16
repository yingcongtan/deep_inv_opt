# Copyright (C) to Yingcong Tan, Andrew Delong, Daria Terekhov. All Rights Reserved.

import numpy as np
import torch
from .util import as_tensor
from .util import as_numpy
from .util import as_str
from .util import build_tensor
from .linprog import squared_error
from .linprog import linprog
from .linprog import InfeasibleConstraintsError
from .linprog import UnboundedConstraintsError
from .linprog import _eps_schedule_decay
from .linprog import _eps_schedule_const


class ParametricLP:
    '''
    ParametricLP is a base class for defining a parametric
    linear program. The subclass is responsible for implementing
    the generate(u, w) function as depicted in the 1D example below.

        class Simple1D(io.ParametricLP):

            """min c(u) s.t. a(u) x <= b(u)"""
        
            def generate(self, u, w):
                w0, w1, w2, w3, w4, w5 = w

                c = [[w0 + w1*u]]
                A = [[w2 + w3*u]]
                b = [[w4 + w5*u]]
                
                return c, A, b, None, None   # None for A_eq or b_eq
            
        f = Simple1D([ 1.0, 0.0,   # c(u) = [ 1.0 + 0.0*u]
                      -1.0, 0.0,   # A(u) = [-1.0 + 0.0*u]
                       1.0, 1.0])  # b(u) = [ 1.0 + 1.0*u]

        params = f(u=2)          # params is [1.0, -1.0, -3.0, None, None]
        x = io.linprog(*params)  # x is -3.0

    '''
    def __init__(self, weights):
        self.weights = as_tensor(weights)
        self.weights.requires_grad_()
                
    def zero_grads(self):
        if self.weights.grad is not None:
            self.weights.grad.detach_()
            self.weights.grad.data.zero_()
            
    def generate(self, u, w):
        raise NotImplementedError("Should be implemented by subclass")
    
    def __call__(self, u):
        # If called with a single scalar value, must be in a list to become a tensor
        if isinstance(u, (int, float)):
            u = [u]
            
        u = as_tensor(u)
        assert len(u) == 1, "Parametric LP currently must be called with a single 1xK u value"
            
        # Call build_tensor on each 
        return list(map(build_tensor, self.generate(u, self.weights)))
    
    def __str__(self):
        return "weights=%s" % as_str(self.weights, flat=True)

      
def inverse_parametric_linprog(u, x, f,
                               lr=10.0,
                               loss=squared_error,
                               eps=1e-5,
                               eps_decay=False,
                               max_steps=10,
                               callback=None,
                               solver=linprog):
    """Solves the inverse parametric linear programming problem.
    
    Feature tensor u should be NxK where N is the number of training points and K is the
    dimensionality of the features of the parametric LP.
    
    Target tensor x should be NxM where M is the dimensionality of the decision variables.
    
    Parametric LP f should be a subclass of ParametricLP. Each row of u will be run through f.

    Here is how you would run it on a Simple1D example from the ParametricLP documentation:

        u = io.tensor([[1],
                       [2],
                       [3],
                       [4]])

        x = io.tensor([[-2],  # Data generated from min x s.t. -x <= 1+u
                       [-3],
                       [-4],
                       [-5]])

        f = Simple1D([ 1.0, 0.0,   # c(u) = [ 1.0 + 0.0*u]
                      -1.0, 0.0,   # A(u) = [-1.0 + 0.0*u]
                       0.7, 1.0])  # b(u) = [ 1.0 + 1.0*u]

        lr = [0, 0, 0, 0, 1, 1]    # learn w4 and w5 only, for b(u)
        io.inverse_parametric_linprog(u, x, f, lr=lr, max_steps=10,                           # Train f.weights 10 steps
                                      callback=io.inverse_parametric_linprog_step_printer(),  # Print progress
                                      solver=io.custom_linprog(eps=0.0001))                   # Solve to high precision
    """

    assert len(u) == len(x)
    if not isinstance(lr, (list, tuple)):
        lr = [lr]   # Broadcast same learning rate to all weights by default
    lr = as_tensor(lr)
    u = u.detach()  # Detach u and x just in case they accidentally requires_grad
    x = x.detach()
    
    def run_forward_loss(eps):
        # Run the forward solver on each individual set of parameters and collect the results
        x_predicts = []
        curr_loss = []
        for ui, xi in zip(u, x):
            # Generate LP coefficients given the current feature set
            c, A_ub, b_ub, A_eq, b_eq = f(ui)
            
            # Solve the forward problem, and compute the loss
            xi_pred = solver(c, A_ub, b_ub, A_eq, b_eq, eps=eps).t()
            xi_loss = torch.sum(loss(c, xi.view(1, -1).t(), xi_pred.t()), 0)
            
            # Record the result 
            curr_loss.append(xi_loss)
            x_predicts.append(xi_pred)
            
        # Stack the results into matricies, for convenience
        curr_loss = torch.mean(torch.cat(curr_loss))
        x_predicts = torch.cat(x_predicts)

        return curr_loss, x_predicts
    
    if eps_decay:
        eps_schedule = _eps_schedule_decay(0.1, eps, max_steps)
    else:
        eps_schedule = _eps_schedule_const(eps)
    
    # Outer loop trains the coefficients defining the LP
    step_size = 1.0
    for step in range(1, max_steps+1):
        f.zero_grads()

        # Compute the current loss and backpropagate it to the initial parameters
        curr_loss, x_predicts = run_forward_loss(eps_schedule(step-1))
        curr_loss.backward()
        
        # Detach the parameters once gradient for this iteration is computed
        with torch.no_grad():

            # Report current state
            if callback:
                callback(step, u, x, f, x_predicts, curr_loss.item())

            # Compute direction of descent
            w0 = f.weights.data.clone()
            dw = f.weights.grad.mul(-lr)
            
            # TODO: Turn this into proper line search. Right now it heuristically accepts any decrement (beta, but no alpha).
            beta = 0.5
            step_size = min(1.0, step_size/beta)  # Give the step size a boost compared to last time
            while step_size > 1e-16:
                # Make step in parameter space
                f.weights.data[:] = w0 + dw*step_size
                
                # Evaluate loss at that point, and accept the step if loss is an improvement.
                try:
                    trial_loss, _ = run_forward_loss(eps_schedule(step-1))
                    if trial_loss < curr_loss:
                        break
                except InfeasibleConstraintsError:
                    pass  # linprog_feasible failed
                except UnboundedConstraintsError:
                    pass  # linprog_feasible failed
                except RuntimeError as err:
                    msg, *_ = err.args
                    if "Lapack Error" not in msg:  # torch.inverse probably failed
                        raise

                # Otherwise reduce step size by factor beta
                step_size *= beta
            else:
                # If step size got to floating point zero, then give up
                f.weights.data[:] = w0
                print("Stopping early due to step_size < 1e-16.")
                break
        # if curr_loss <eps:
        #     break

    # Report final state
    if callback:
        curr_loss, x_predicts = run_forward_loss(eps)
        callback(None, u, x, f, x_predicts, curr_loss.item())
    
    return f, curr_loss, x_predicts


def inverse_parametric_linprog_step_printer(want_details=False):
    """Returns a callback that prints the state of each inverse_parametric_linprog step."""

    def callback(step, u, x, f, x_predicts, loss):
        step_str = "done" if step is None else "%04d" % step
        print("inverse_parametric_linprog[%s]: loss=%.6f %s" % (step_str, loss, f))
        if want_details:
            for ui, xi in zip(u, x_predicts):
                print("    x=%s f(%s)=%s " % (as_str(xi), as_str(ui), " ".join(as_str(*f(ui), flat=True))))

    return callback