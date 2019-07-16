# Copyright (C) to Yingcong Tan, Andrew Delong, Daria Terekhov. All Rights Reserved.

import numpy as np
import torch
from .util import as_tensor
from .util import as_numpy


_LINPROG_CENTRAL_PATH = []


def _collect_linprog_central_path(step, c, A_ub, b_ub, A_eq, b_eq, x, dx, t):
    global _LINPROG_CENTRAL_PATH
    if step == 0:
        _LINPROG_CENTRAL_PATH = []
    _LINPROG_CENTRAL_PATH.append((step, as_numpy(x), as_numpy(dx)))
    return _LINPROG_CENTRAL_PATH


def _arr2str(x, indent=0):
    return np.array2string(x, precision=4, max_line_width=100, threshold=5, floatmode='fixed', suppress_small=True, prefix=' '*indent)


def _eps_schedule_decay(start, end, steps):
    k = np.log(end/start)/steps
    return lambda step: np.maximum(end, start*np.exp(k*step))


def _eps_schedule_const(eps):
    return lambda step: eps


def linprog_step_printer():
    """Returns a callback that prints the state of each linprog step."""

    def callback(step, c, A_ub, b_ub, A_eq, b_eq, x, dx, t):
        c = as_numpy(c).ravel()
        x = as_numpy(x).ravel()
        cx = np.dot(x, c)
        step_str = "done" if step is None else "%04d" % step
        print("linprog[%s]: x=%s cx=%s t=%.2f" % (step_str, _arr2str(x), _arr2str(cx), t))

    return callback


def inverse_linprog_step_printer(want_constraints=False):
    """Returns a callback that prints the state of each inverse_linprog step."""

    def callback(step,
                 c, A_ub, b_ub, A_eq, b_eq,
                 dc, dA_ub, db_ub, dA_eq, db_eq,
                 x_target, x_predict, loss):

        c = as_numpy(c).ravel()
        x = as_numpy(x_predict).ravel()
        step_str = "done" if step is None else "%04d" % step
        print("inverse_linprog[%s]: c=%s x*=%s loss=%.5f" % (step_str, _arr2str(c), _arr2str(x), loss))
        if want_constraints:
            print("   A_ub: %s" % _arr2str(as_numpy(A_ub), 9))
            print("   b_ub: %s" % _arr2str(as_numpy(b_ub), 9))
            if A_eq is not None:
                print("   A_eq: %s" % _arr2str(as_numpy(A_eq), 9))
                print("   b_eq: %s" % _arr2str(as_numpy(b_eq), 9))

    return callback


def _line_search(f, x, dx, f_x, grad_f_x, alpha=0.3, beta=0.25):
    # Algorithm 9.2 from the BV book
    # Use "not a < b" rather than "a > b" so that loop continues while a is nan
    step_size = 1.0
    while not f(x + step_size * dx) <= f_x + (alpha*step_size) * (grad_f_x.t() @ dx):
        step_size *= beta
        if step_size == 0.0:
            raise RuntimeError("line search encountered a step size of zero!")
    return step_size

# Allows _linprog_ip to raise an error that specifically
# identifies that matrix inverse failed (probably non-singular)
# Useful for backing up in line search.
class MatrixInverseError(RuntimeError):
    pass

def _linprog_ip(x, c, A_ub, b_ub, A_eq, b_eq,
                 max_steps, t0, mu, eps, callback):
    """Returns a solution to the given LP using the interior point method."""
    assert (A_eq is None) == (b_eq is None)

    # TODO: Unfortunately PyTorch doesn't handle zero-length tensor dimensions correctly yet (unlike numpy)
    #       so we cannot simply sete A_eq to a (0, m)-tensor and let the remaining code go through.
    #       Instead we must constantly check if A_eq is None and do something special.

    m = len(c)        # Number of variables
    n_ub = len(A_ub)  # Number inequality constraints
    n_eq = len(A_eq) if A_eq is not None else 0  # Number equality constraints
    
    # Parameters of the algorithm
    t = t0

    # Objective function of barrier method f(x) = t * c'x - \sum_i log(-ai'x +bi)
    # Note that x inside the lambda does not refer to the _linprog_ip argument x already in scope,
    # but the t value is captured by reference from _linprog_ip and will be whatever the
    # current value of t is whenever it's invoked
    f = lambda x: t*(c.t() @ x) - torch.sum(torch.log(-(A_ub @ x - b_ub)))

    # Initial callback at step 0
    if callback:
        if callback(0, c, A_ub, b_ub, A_eq, b_eq, x, None, t) == True:
            return x

    # Run interior point inner loop
    for step in range(1, max_steps+1):

        # Newton step to next point on central path
        d = 1 / (b_ub - A_ub @ x)
        grad_phi = A_ub.t() @ d                # Gradient of barrier phi(x)
        hess_phi = A_ub.t() @ ((d**2) * A_ub)  # Hessian of barrier phi(x)
                                               # (Equiv to A.t() @ torch.diagflat(d**2) @ A)
        grad_obj = t*c  # Gradient of t c'x
        hess_obj = 0    # Hessian of t c'x
        
        grad_f = grad_obj + grad_phi  # Gradient of f(x) = tc'x + phi(x)
        hess_f = hess_obj + hess_phi  # Hessian of f(x) = tc'x + phi(x)
        if A_eq is not None:
            # Newton WITH equality constraints            
            # M = [[ H  A' ],
            #      [ A  0  ]]
            # where H is Hessian of the entire objective f(x) = t*c'x + phi(x)
            # (10.19 in BV book)
            M = torch.cat((torch.cat((hess_f, A_eq.t()), 1),
                           torch.cat((A_eq, torch.zeros(n_eq, n_eq, dtype=A_eq.dtype)), 1)), 0)
            # v =  [[     -g ],
            #       [ b - Ax ]],
            # where g is gradient of the entire objective f(x) = t*c'x + phi(x)
            # (10.19 in BV book)
            v = torch.cat((-grad_f, b_eq - A_eq @ x), 0)
        else:
            # Newton WITHOUT equality constraints
            M = hess_f
            v = -grad_f
        
        # Compute Newton step for infeasible points
        try:
            # dxw = torch.inverse(M) @ v      # Direction of Newton step with slack variables w
            dxw = torch.solve(v,M).solution
        except RuntimeError as e:
            if any("Lapack Error" in arg for arg in e.args):
                raise MatrixInverseError(*e.args)
            raise
        dx = dxw[:m]                    # Strip off slack variables w

        # Checking stopping criterion of newton's method
        lamb = v.t() @ dxw   
        if lamb*0.5 <= eps:  # Stopping criterion of newton's method
            t *= mu     # Update t
            
            # Report current state. If callback returns True, terminate
            if callback:
                if callback(step, c, A_ub, b_ub, A_eq, b_eq, x, dx, t) == True:
                    break
        else:           # Update x based on gradient if stopping criterion is not met
            step_size = _line_search(f, x, dx, f(x), grad_f)
            x = x + step_size*dx
            if max(abs(x)>1e30):
                raise UnboundedConstraintsError

        # Report current state. If callback returns True, terminate
        if callback:
            if callback(step, c, A_ub, b_ub, A_eq, b_eq, x, dx, t) == True:
                break

        # Stop when accuracy below eps
        if n_ub/t < eps:
            break

        step += 1
        
    # Report final state
    if callback:
        callback(None, c, A_ub, b_ub, A_eq, b_eq, x, None, t)

    return x

def _infeaStartNewton(x_init, c, A_ub, b_ub, A_eq, b_eq,
                      max_steps, t0, eps, callback):

    #     rewrite ineq constraints using log barrier function:
    #     min   t_0* c'X - Sum log(b_ub - A_ub@X)
    #     s.t.  A_eq@X == b_eq 

    def _infeaStartNewton_line_search(  rN, rP, rD, 
                                        x, xDual, dx, dxDual,
                                        alpha=0.1, beta=0.3):
        # Algorithm 10.2 from the BV book
        # backtrack line search to ensure that newton's step imporves the primal/dual residuals
        step_size = 1.0
        while not (1-alpha*step_size)*rN(rP(x), rD(xDual)) >= rN(rP(x+step_size*dx), rD(xDual+step_size*dxDual)):
            step_size = step_size*beta
            if step_size == 0.0:
                raise RuntimeError("line search encountered a step size of zero!")
        # Additional line search added for our algorithm, to ensure inequality constratins are satisfied
        while all(A_ub@(x+step_size*dx)<= b_ub) is not True:
            step_size = step_size*beta
            if step_size == 0.0:
                raise RuntimeError("line search encountered a step size of zero!")
        return step_size

    def is_feasible(x):
        return torch.allclose(x[-1], torch.zeros((1,1),dtype = torch.double), atol=1e-5)

    """Returns a solution to the given LP using the interior point method."""

    m = len(c)        # Number of variables
    n_ub = len(A_ub)  # Number inequality constraints
    n_eq = len(A_eq)     
    t = t0
    # Objective function of barrier method f(x) = t * c'x - \sum_i log(-ai'x +bi)
    # Note that x inside the lambda does not refer to the _linprog_ip argument x already in scope,
    # but the grad_f value is captured by reference from _linprog_ip and will be whatever the
    # current value of grad_f is whenever it's invoked

    # Functions for computing primal, dual residual
    rP = lambda x: A_eq@x-b_eq
    rD = lambda xDual: grad_f + A_eq.t()@xDual
    rN = lambda rprimal, rdual: torch.norm(torch.cat((rprimal, rdual), 0))

    # initialize dual var
    xDual = torch.ones((n_eq,1), dtype = torch.double)
    x = x_init

    # Run Newton's Method loop
    for step in range(1, max_steps+1):
        # print("step", step)

        # [H_f, A']  [dx]      =  [-g_f - A'xDual]
        # [A  , 0 ]  [dxDual]  =  [ b-A@X]

        # Newton step to next point on central path
        d = 1 / (b_ub - A_ub @ x)
        grad_phi = A_ub.t() @ d                # Gradient of barrier phi(x)
        hess_phi = A_ub.t() @ ((d**2) * A_ub)  # Hessian of barrier phi(x)
                                               # (Equiv to A.t() @ torch.diagflat(d**2) @ A)
        grad_obj = t*c  # Gradient of t c'x
        hess_obj = 0    # Hessian of t c'x

        grad_f = grad_obj + grad_phi  # Gradient of f(x) = tc'x + phi(x)
        hess_f = hess_obj + hess_phi  # Hessian of f(x) = tc'x + phi(x)

        # Newton WITH equality constraints            
        # M = [[ H  A' ],
        #      [ A  0  ]]
        # where H is Hessian of the entire objective f(x) = t*c'x + phi(x)
        # (10.19 in BV book)
        M = torch.cat((torch.cat((hess_f, A_eq.t()), 1),
                       torch.cat((A_eq, torch.zeros(n_eq, n_eq, dtype=A_eq.dtype)), 1)), 0)
        # v =  [[-g - A'xDual] 
        #       [ b - Ax ]],
        # where g is gradient of the entire objective f(x) = t*c'x + phi(x)
        # (10.19 in BV book)
        v = torch.cat((- grad_f - A_eq.t()@xDual, b_eq - A_eq @ x), 0)


        # Compute Newton step for infeasible points
        try:
            # dxw = torch.inverse(M) @ v      # Direction of Newton step with slack variables w
            dxw = torch.solve(v,M).solution
        except RuntimeError as e:
            if any("Lapack Error" in arg for arg in e.args):
                raise MatrixInverseError(*e.args)
            raise
        dx = dxw[:m]                    # gradient of primal variables
        dxDual = dxw[m:]            # gradient of dual variables


        step_size = _infeaStartNewton_line_search(rN, rP, rD, 
                                                  x, xDual, dx, dxDual)
        x = x + step_size*dx
        xDual = xDual + step_size*dxDual

        if max(abs(x)>1e30):
            raise UnboundedConstraintsError


        # Stop when finding a feasible solution
        # the objective is to find a strictl feasible solution
        # we terminate immediately after proving feasbility
        # Thus, no need to update t = t*mu as normal IPM
        if is_feasible(x) == True:
            break

        # if rN(rP(x), rD(xDual))<= eps:
        #     break
        step += 1

    #report final state
    if callback:
        callback(None, c, A_ub, b_ub, A_eq, b_eq, x, None, t)
    
    return x

# Allows linprog_feasible to raise an error that specifically
# identifies that an infeasible instance was detected and that no feasible
# point will be returned. Useful for backing up in line search.
class InfeasibleConstraintsError(RuntimeError):
    pass
class UnboundedConstraintsError(RuntimeError):
    pass

def linprog_feasible(c, A_ub, b_ub, A_eq=None, b_eq=None, 
                     max_steps=100,t0=1.0, mu=2.0, eps=0.001,
                     callback=None):

    """Returns a strictly feasible point for the given LP."""
    assert max_steps >= 1
    assert (A_eq is None) == (b_eq is None)

    # If slack variable < 0 and the equality constraints are satisfied, then we can stop.
    def is_feasible(x, A_ub, b_ub, A_eq, b_eq):
        return all([all(A_ub@x <=b_ub),  # ineuqality constraint feasibility
                    (A_eq is None or torch.allclose(A_eq @ x - b_eq, torch.zeros_like(b_eq), atol=1e-5) )])  # ineuqality constraint feasibility

    # Terminates _linprog_ip as soon as a feasible point is found
    def check_feasible(step, c, A_ub, b_ub, A_eq, b_eq, x, dx, t):
        if callback is not None:
            callback(step, c, A_ub, b_ub, A_eq, b_eq, x, dx, t)  # Forward args to user callback if applicable
        return is_feasible(x, A_ub, b_ub, A_eq, b_eq)

    n_ub, m = A_ub.shape

    if A_eq is not None:
        #     min   0
        #     s.t.  A_ub@X <= b_ub
        #           A_eq@X == b_eq 
            
        #     equivalently:
        #     min   0
        #     s.t.  A_ub@X -s <= b_ub 
        #           A_eq@X    == b_eq
        #                   s == 0
            
        #     update:
        #             c = [[0]
        #                  [1]]
        #             x = [[X]
        #                  [S]]
        #             A_ub = [A_ub,  -1]
        #             A_eq = [[A_ub, 0]   # A_eq@X == b_eq
        #                     [ 0,   1]]  # s=0
        #             b_eq = [0]          # s=0
                       
        n_eq, _ = A_eq.shape 
        assert m == _, "expected same number of columns in A_ub and A_eq"

        # Construct x_init=(x0, s0) such that x_init is strictly feasible wrt inequalities of the max-infeasibility LP
        x0 = torch.zeros((m, 1), dtype=torch.double)
        s0 = torch.min(b_ub).view(-1, 1) * -1 + 1 # Equivalent to max(A_ub @ x0 - b_ub) when x0=0
        x_init = torch.cat((x0, s0), 0)

        # Construct the max-infeasibility LP
        c = torch.cat ( (c, torch.zeros((1,1), dtype=torch.double) ), 0)
        # c = torch.cat((torch.zeros((m,1), dtype=torch.double), torch.zeros((1,1), dtype=torch.double)),0)
        A_ub = torch.cat((A_ub, -torch.ones((n_ub, 1), dtype=torch.double)), 1)
        A_eq = torch.cat((A_eq, torch.zeros((n_eq, 1), dtype=torch.double)), 1) 
        A_eq = torch.cat((A_eq, torch.cat((torch.zeros((1,m),dtype=torch.double),
                                           torch.ones((1,1),dtype=torch.double) ),1)  
                        ),0)
        b_eq = torch.cat((b_eq, torch.zeros((1,1), dtype=torch.double)),0)
        
        x_center = _infeaStartNewton(x_init, c, A_ub, b_ub, A_eq, b_eq, 
                                t0=t0, max_steps=max_steps, eps=eps, 
                                callback=check_feasible)
    else:
        n_eq, _ = A_eq.shape if A_eq is not None else (0, m)
        assert m == _, "expected same number of columns in A_ub and A_eq"

        # Construct x_init=(x0, s0) such that x_init is strictly feasible wrt inequalities of the max-infeasibility LP
        x0 = torch.zeros((m, 1), dtype=torch.double)
        s0 = torch.min(b_ub).view(-1, 1) * -1.5 + 1 # Equivalent to max(A_ub @ x0 - b_ub) when x0=0
        x_init = torch.cat((x0, s0), 0)

        # Construct the max-infeasibility LP
        c = torch.cat((torch.zeros((m, 1), dtype=torch.double),
                    torch.ones((1, 1), dtype=torch.double)), 0)
        A_ub = torch.cat((A_ub, -torch.ones((n_ub, 1), dtype=torch.double)), 1)
        A_eq = torch.cat((A_eq, torch.zeros((n_eq, 1), dtype=torch.double)), 1) if A_eq is not None else None

        # Hack to prevent infeasibility problem instance from being unbounded (nonsingular Hessian)
        s_lower_bound = 10.0*(1.0 + torch.max(torch.abs(A_ub.detach()))).view(1, 1)   # Make lower bound dependent on scale of constraint coefficients
        A_ub = torch.cat((A_ub, -c.t()), 0)
        b_ub = torch.cat((b_ub, s_lower_bound), 0)

        # Solve the max-infeasible LP with interior point
        x_center = _linprog_ip(x_init, c, A_ub, b_ub, A_eq, b_eq,
                            max_steps=max_steps, t0=t0, mu=mu, eps=eps,
                            callback=check_feasible)

    # print("x_center", x_center)
    # Check strict feasibility
    if not is_feasible(x_center, A_ub, b_ub, A_eq, b_eq):
        raise InfeasibleConstraintsError("Constraints were not strictly feasible, or strictly feasible point not found in allotted steps.")

    # Return strictly feasible point for original LP
    return x_center[:-1]

def linprog(c, A_ub, b_ub, A_eq=None, b_eq=None,
            max_steps=100, t0=1.0, mu=2.0, eps=1e-5,
            callback=None):
    """Returns a solution to the given LP using the interior point method."""    

    x_init = linprog_feasible(c, A_ub, b_ub, A_eq, b_eq, 
                    max_steps=max_steps, t0=t0, mu=mu, eps=eps,)  # A_eq, b_eq deliberately omitted, since subsequent _linprog_ip is infeasible start Newton

    x = _linprog_ip(x_init, c, A_ub, b_ub, A_eq, b_eq,
                    max_steps=max_steps, t0=t0, mu=mu, eps=eps,
                    callback=callback)

    # Sanity check that equality constraints are satisfied.
    # If the system is properly infeasible, it should have been caught by _infeaStartNewton raising an exception, so this is an internal check.
    if A_eq is not None:
        assert torch.allclose(A_eq @ x - b_eq, torch.zeros_like(b_eq), atol=1e-5), "linprog failed to satisfy equality constraints, but also failed to detect infeasibility, so there's something wrong"

    return x


def custom_linprog(**custom_linprog_kwargs):
    """Returns a callable linprog() with custom default settngs.

    Useful for specifying a new default eps, for example.
    """
    def callback(*args, **kwargs):
        kwargs.update(custom_linprog_kwargs)
        return linprog(*args, **kwargs)
    return callback


def squared_error(c, x_target, x_predict):
    return (x_target - x_predict)**2


def abs_duality_gap(c, x_target, x_predict):
    return torch.abs(c.t() @ (x_target - x_predict))


def inverse_linprog(x_target, c, A_ub, b_ub, A_eq=None, b_eq=None,
                    max_steps=100,
                    eps=1e-5, #defaul value for the end of eps_decay
                    eps_decay=False,
                    learn_rate_c=10.0,
                    learn_rate_ub=0.0,
                    learn_rate_eq=0.0,
                    loss=squared_error,
                    callback=None,
                    normalize=True,
                    return_loss=False,
                    solver=linprog):
    """Solves the inverse linear programming problem."""

    # TODO: Unfortunately PyTorch doesn't handle zero-length tensor dimensions correctly yet (unlike numpy)
    #       so we cannot simply replace A_eq with a (0, m) and let the remaining code go through.
    #       Instead we must constantly check if A_eq is None.
    
    c = c.clone()
    A_ub = A_ub.clone()
    b_ub = b_ub.clone()
    A_eq = A_eq.clone() if A_eq is not None else None
    b_eq = b_eq.clone() if b_eq is not None else None

    solver_callback = _collect_linprog_central_path if callback else None

    # Project c to have unit norm. TODO also normalize constraint coefficients?
    if normalize and len(c) > 1:
        c /= sum(abs(c))
    
    def reset_grads():
        for p in (c, A_ub, b_ub, A_eq, b_eq):
            if p is not None:
                p.requires_grad_()         # Flag the parameters as needing a gradient
                if p.grad is not None:
                    p.grad.data.zero_()    # But also zero out inplace any values leftover from the previous gradient

    def detach_params():
        for p in (c, A_ub, b_ub, A_eq, b_eq):
            if p is not None:
                p.detach_()                # Stop tracing computations made with these parameters, essentially fixing them

    def run_forward_loss(c, A_ub, b_ub, A_eq, b_eq, eps, solver_callback):
        # Solve for x given fixed problem coefficients
        x_predict = solver(c, A_ub, b_ub, A_eq, b_eq, eps = eps, callback=solver_callback)
        # Compute error with respect to training target
        curr_loss = torch.mean(torch.sum(loss(c, x_target, x_predict), 0))

        # Currently the loss is just the error.
        # In future it may include priors on c and constraints.
        return curr_loss, x_predict

    # compute the eps at each step which decay from 0.1 (at step 0) to eps (at final step)
    if eps_decay:
        eps_schedule = _eps_schedule_decay(0.1, eps, max_steps)
    else:
        eps_schedule = _eps_schedule_const(eps)

    # Outer loop trains the coefficients defining the LP
    step_size = 0.05
    for step in range(1, max_steps+1):
        reset_grads()  # Reset the gradients so that they don't accumulate from previous outer loop iteration
        
        # Compute the current loss and backpropagate it to the initial parameters
        curr_loss, x_predict = run_forward_loss(c, A_ub, b_ub, A_eq, b_eq,
                                                eps_schedule(step-1), solver_callback)
                                                # steps start at 0
        curr_loss.backward()
        if step == 1:
            inital_loss = as_numpy(curr_loss)
        # Detach the parameters once gradient for this iteration is computed
        detach_params()
        
        with torch.no_grad():

            dc = -learn_rate_c*c.grad
            dA_ub = -learn_rate_ub*A_ub.grad
            db_ub = -learn_rate_ub*b_ub.grad
            dA_eq = -learn_rate_eq*A_eq.grad if A_eq is not None else None
            db_eq = -learn_rate_eq*b_eq.grad if b_eq is not None else None
            
            # Linesearch which heuristically accepts any decrement.
            beta = 0.5
            step_size = min(1.0, step_size/beta)  # Give the step size a boost compared to last time
            while step_size > 1e-8:
                # Create step in parameter space
                _c = c + step_size*dc
                _A_ub = A_ub + step_size*dA_ub
                _b_ub = b_ub + step_size*db_ub
                _A_eq = A_eq + step_size*dA_eq if A_eq is not None else None
                _b_eq = b_eq + step_size*db_eq if A_eq is not None else None
                if normalize and len(_c) > 1:
                    _c /= sum(abs(_c))
                
                # Evaluate loss at that point, and accept the step if loss is an improvement.
                try:
                    trial_loss, trial_x_predict = run_forward_loss(_c, _A_ub, _b_ub, _A_eq, _b_eq, eps_schedule(step-1), None)
                    if trial_loss < curr_loss:
                        break
                except InfeasibleConstraintsError:
                    pass  # linprog_feasible failed
                except UnboundedConstraintsError:
                    pass  # unbounded solution
                except RuntimeError as err:
                    msg, *_ = err.args
                    if "Lapack Error" not in msg:  # torch.inverse probably failed
                        raise

                # Otherwise reduce step size by factor beta
                step_size *= beta
            else:
                # If step size got to floating point zero, then give up
                break

            # Report current state
            if callback:
                callback(step, c, A_ub, b_ub, A_eq, b_eq, dc, dA_ub, db_ub, dA_eq, db_eq, x_target, x_predict, curr_loss.item())
            
            # Take a learning step
            c = _c
            A_ub = _A_ub
            b_ub = _b_ub
            A_eq = _A_eq
            b_eq = _b_eq
            
            # Project c to have unit norm. 
            if normalize and len(c) > 1:
                c /= sum(abs(c))
    
    # Report final state
    if callback:
        callback(None, c, A_ub, b_ub, A_eq, b_eq, None, None, None, None, None, x_target, x_predict, curr_loss.item())
    
    # Only return as many parameters as the user specified.
    if return_loss == True:
        # using low eps value to compute loss and x_final with high accuracy
        final_loss, x_final = run_forward_loss(c, A_ub, b_ub, A_eq, b_eq, 1e-7, solver_callback) 
        if A_eq is None:
            return c, A_ub, b_ub, final_loss, x_final, inital_loss
        else:
            return c, A_ub, b_ub, A_eq, b_eq, final_loss, x_final,inital_loss
    else:
        if A_eq is None:
            return c, A_ub, b_ub
        else:
            return c, A_ub, b_ub, A_eq, b_eq

