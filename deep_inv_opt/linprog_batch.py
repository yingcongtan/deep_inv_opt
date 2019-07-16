import torch
import numpy as np
import deep_inv_opt as di
import deep_inv_opt.plot as dip
import matplotlib.pyplot as plt
from .util import as_tensor
from .util import as_numpy
from .util import as_str
from .util import build_tensor
from .linprog import squared_error
from .linprog import _eps_schedule_decay
from .linprog import _eps_schedule_const

def _T(x):
    return torch.transpose(x, -2, -1)



# Raised when either the primal or dual is detected to be infeasible
class InfeasibleOrUnboundedError(RuntimeError):
    pass


def _infeaStartNewton_batch(x_init, c, A_ub, b_ub, 
                     A_eq, b_eq, 
                     t0=1.0, max_steps = 100, eps=1e-3, callback=None):
    
    
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
        # print("step_size after residual line search", step_size)
        # Additional line search added for our algorithm, to ensure inequality constratins are satisfied
        
        while torch.all(A_ub@(x+step_size*dx)<= b_ub) !=1:
            step_size = step_size*beta
            if step_size == 0.0:
                raise RuntimeError("line search encountered a step size of zero!")
        # print("step_size after feasibility line search", step_size)

        return step_size
    
    
    
    k, n_ub, m  = A_ub.shape
    n_eq = A_eq.shape[1]

    xDual = torch.ones((k, n_eq,1), dtype = torch.double)
    x = x_init
    t = t0
    

    rP = lambda x: A_eq@x-b_eq
    rD = lambda xDual: grad_f + _T(A_eq)@xDual
    rN = lambda rprimal, rdual: torch.norm(torch.cat((rprimal, rdual), 1),p=2)

    for step in range(1, max_steps+1):

        # [H_f, A']  [dx]              =  [-g_f - A'xDual]
        # [A  , 0 ]  [xDual + dxDual]  =  [ b-A@X]

        # Newton step to next point on central path
        d = 1 / (b_ub - A_ub @ x)
        grad_phi = _T(A_ub) @ d                # Gradient of barrier phi(x)
        hess_phi = _T(A_ub) @ ((d**2) * A_ub)  # Hessian of barrier phi(x)
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
        M = torch.cat((torch.cat((hess_f, _T(A_eq)), 2),
                       torch.cat((A_eq, torch.zeros(k, n_eq, n_eq, dtype=A_eq.dtype)), 2)), 1)
        # v =  [[-g - A'xDual],dtype
        #       [ b - Ax ]],
        # where g is gradient of the entire objective f(x) = t*c'x + phi(x)
        # (10.19 in BV book)
        v = torch.cat((- grad_f - _T(A_eq)@xDual, b_eq - A_eq @ x), 1)


        # Compute Newton step for infeasible points
        try:
            # dxw = torch.inverse(M) @ v      # Direction of Newton step with slack variables w
            dxw = torch.solve(v,M).solution
        except RuntimeError as e:
            if any("Lapack Error" in arg for arg in e.args):
                raise MatrixInverseError(*e.args)
            raise
        dx = dxw[:,:m]                    # gradient of primal variables
        dxDual = dxw[:,m:]            # gradient of dual variables

        # print("step", step)
        # print("before update primal/dual var")
        # print("\tprimal var\t", x)
        # print("\tgradient primal var\t", dx)
        # print("\tdual var\n\t", xDual)
        # print("\tgradient dual var\t", dxDual)
        # print("\tPrimal Residual\t:", rP(x))
        # print("\tDual Residual\t:", rD(xDual))
        # print("\trNorm\n\t", rN(rP(x), rD(xDual)))
        # print("\tb_ub-A_ub@x\n\t", (b_ub-A_ub@x))

        step_size = 1
        beta = 0.3
        alpha = 0.1
                
        step_size = _infeaStartNewton_line_search(rN, rP, rD, 
                                                  x, xDual, dx, dxDual)

        x = x+step_size*dx
        xDual = xDual+step_size*dxDual
        # print("After take newton's step")
        # print("\tprimal var\t", x)
        # print("\tdual var\t", xDual)    
        # print("\tPrimal Residual\t:", rP(x))
        # print("\tDual Residual\t:", rD(xDual))
        # print("\trNorm\t", rN(rP(x), rD(xDual)))
        # print("\tb_ub-A_ub@x\n\t", (b_ub-A_ub@x))
        # print()
        
        if all([torch.allclose(i[-1], torch.zeros((1,1),dtype = torch.double), atol=1e-3) for i in x]):
            break
        # if rN(rP(x), rD(xDual))<= epsilon:
        #     break
            
    return x


def _linprog_ip_batch(x, c, A_ub, b_ub, A_eq, b_eq,
                     t0=1.0, mu=2.0, eps=1e-3, max_steps=100):
    
    def _line_search(f, x, dx, f_x, grad_f_x, alpha=0.3, beta=0.25):
        # Algorithm 9.2 from the BV book
        # Use "not a < b" rather than "a > b" so that loop continues while a is nan
        step_size = 1.0
        while all(f(x + step_size * dx) <= f_x + (alpha*step_size) * (_T(grad_f_x) @ dx).view(-1,1)) is not True:
            # print("f(x + step_size * dx)",f(x + step_size * dx))
            # print("f_x + (alpha*step_size) * (_T(grad_f_x) @ dx).view(-1,1)",f_x + (alpha*step_size) * (_T(grad_f_x) @ dx).view(-1,1))
            # print("f(x + step_size * dx) <= f_x + (alpha*step_size) * (_T(grad_f_x) @ dx).view(-1,1)",f(x + step_size * dx) <= f_x + (alpha*step_size) * (_T(grad_f_x) @ dx).view(-1,1))
            step_size *= beta
            if step_size == 0.0:
                raise RuntimeError("line search encountered a step size of zero!")
        return step_size
    k,n_ub, m = A_ub.shape
    dtype = A_ub.dtype
    n_eq = A_eq.shape[1]
    t = torch.ones((k,1,1), dtype=dtype)
    t = t0
    # print("x.shape", x.shape)

    Tc = _T(c)

    # print("d", d)
    # print("grad_phi", grad_phi)
    # print("hess_phi", hess_phi)

    f = lambda x: t*(Tc @ x).view(-1,1) - torch.sum(torch.log(-(A_ub @ x - b_ub)),1).view(-1,1)

    for step in range(1, max_steps+1):
        d = 1 / (b_ub - A_ub @ x)
        grad_phi = _T(A_ub)@ d                # Gradient of barrier phi(x)
        hess_phi = _T(A_ub) @ ((d**2) * A_ub) 
        
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
            M = torch.cat((torch.cat((hess_f, _T(A_eq)), 2),
                           torch.cat((A_eq, torch.zeros(k, n_eq, n_eq, dtype=A_eq.dtype)), 2)), 1)
            # v =  [[     -g ],
            #       [ b - Ax ]],
            # where g is gradient of the entire objective f(x) = t*c'x + phi(x)
            # (10.19 in BV book)
            v = torch.cat((-grad_f, b_eq - A_eq @ x), 1)
        else:
            # Newton WITHOUT equality constraints
            M = hess_f
            v = -grad_f
            
        # print("M.shape",M.shape)
        # print("v.shape",v.shape)
    
        try:
        #     dxw = torch.inverse(M) @ v      # Direction of Newton step with slack variables w
            dxw = torch.solve(v,M).solution
            # print("dxw.shape",dxw.shape)
        except RuntimeError as e:
            if any("Lapack Error" in arg for arg in e.args):
                raise MatrixInverseError(*e.args)
            raise
        dx = dxw[:,:m]  

        # Checking stopping criterion of newton's method
        lamb = _T(v) @ dxw   
        if max(lamb*0.5) <= eps:  # Stopping criterion of newton's method
            t *= mu     # Update t
            
        else:           # Update x based on gradient if stopping criterion is not met
            step_size = _line_search(f, x, dx, f(x), grad_f)
            x = x + step_size*dx
            if torch.max(torch.abs(x)>1e30):
                raise UnboundedConstraintsError

        # Stop when accuracy below eps
        if n_ub/t < eps:
            break

        step += 1

    return x      


def linprog_feasible_batch(c, A_ub, b_ub, A_eq=None, b_eq=None, t0=1.0,
                     max_steps=200, eps=0.001,
                     callback=None):
    k, n_ub, m = A_ub.shape

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

        n_eq = A_eq.shape [1]

        # Construct x_init=(x0, s0) such that x_init is strictly feasible wrt inequalities of the max-infeasibility LP
        x0 = torch.zeros((k, m, 1), dtype=torch.double)
        temp = torch.min(b_ub) * -1 + 1 # Equivalent to max(A_ub @ x0 - b_ub) when x0=0
        s0 = torch.DoubleTensor(k, 1, 1 ).fill_(temp)
        x_init = torch.cat((x0, s0), 1)

        # Construct the max-infeasibility LP
        c = torch.cat ( (c, torch.zeros((k, 1,1), dtype=torch.double) ), 1)
        A_ub = torch.cat((A_ub, -torch.ones((k, n_ub, 1), dtype=torch.double)), 2)
        A_eq = torch.cat((A_eq, torch.zeros((k, n_eq, 1), dtype=torch.double)), 2) 
        A_eq = torch.cat((A_eq, torch.cat((torch.zeros((k,1,m),dtype=torch.double),
                                           torch.ones((k,1,1),dtype=torch.double) ),2)  
                        ),1)
        b_eq = torch.cat((b_eq, torch.zeros((k, 1,1), dtype=torch.double)),1)

        x_center = _infeaStartNewton_batch(x_init, c, A_ub, b_ub, A_eq, b_eq)
    else:
        #     min   s
        #     s.t.  A_ub@X -s <= b_ub

        #     update:
        #             c = [[0]
        #                  [1]]
        #             x = [[X]
        #                  [s]]
        #             A_ub = [A_ub, -1]


        c = torch.cat((torch.zeros((k, m,1), dtype=torch.double), torch.zeros((k, 1, 1), dtype=torch.double)),1)
        x0 = torch.zeros((k, m, 1), dtype=torch.double)
        temp = torch.min(b_ub) * -1 + 1 # Equivalent to max(A_ub @ x0 - b_ub) when x0=0
        s0 = torch.DoubleTensor(k, 1, 1 ).fill_(temp)
        x_init = torch.cat((x0, s0), 1)  

        A_ub = torch.cat((A_ub, -torch.ones((k, n_ub, 1), dtype = torch.double) ),2)

        # Solve the max-infeasible LP with interior point
        x_center = _linprog_ip_batch(x_init, c, A_ub, b_ub, A_eq=None, b_eq=None, 
                                    t0=t0, max_steps=max_steps, eps=eps, 
                                    callback=check_feasible)
        
    return x_center[:,:-1]

def linprog_batch(c, A_ub, b_ub, A_eq=None, b_eq=None,
            max_steps=100, t0=1.0, mu=2.0, eps=1e-5,
            callback=None):
    """Returns a solution to the given LP using the interior point method."""    


    x_init = linprog_feasible_batch(c, A_ub, b_ub, A_eq, b_eq)  # A_eq, b_eq deliberately omitted, since subsequent _linprog_ip is infeasible start Newton

    # print("x_init", x_init)
    x = _linprog_ip_batch(x_init, c, A_ub, b_ub, A_eq, b_eq,
                    max_steps=max_steps, t0=t0, mu=mu, eps=eps)

    # Sanity check that equality constraints are satisfied.
    # If the system is properly infeasible, it should have been caught by _infeaStartNewton raising an exception, so this is an internal check.
    if A_eq is not None:
        assert torch.allclose(A_eq @ x - b_eq, torch.zeros_like(b_eq), atol=1e-5), "linprog failed to satisfy equality constraints, but also failed to detect infeasibility, so there's something wrong"

    return x


def inverse_parametric_linprog_batch(u, x, f,
                                     loss=squared_error,
                                     lr=10.0,
                                     eps=1e-5,
                                     eps_decay = False,
                                     max_steps=100,
                                     callback = None):
    lr = as_tensor(lr)  # Convert learning rate to tensor, possible with per-weight learning rates
    assert lr.numel() in (1, len(f.weights)), "Expected learning rate to specify either 1 or len(f.weights) elements"
    # assert len(u) == len(x)
    u = u.detach()  # Detach u and x just in case they accidentally requires_grad
    x = x.detach()
    
    def run_forward_loss(eps = eps):
        # Generate LP coefficients given the current feature set
        c, A_ub, b_ub, A_eq, b_eq = f(u)
        
        # Solve the forward problem, and compute the loss.
        # If any instance fails, raise an error so that the line search can immediately back off.
        x_predict= linprog_batch(c, A_ub, b_ub, A_eq, b_eq, eps=eps)
        curr_loss = torch.mean(torch.sum(loss(c, x, x_predict),1))

        return curr_loss, x_predict
    if eps_decay:
        eps_schedule = _eps_schedule_decay(0.1, eps, max_steps)
    else:
        eps_schedule = _eps_schedule_const(eps)

    # Outer loop trains the coefficients defining the LP
    step_size = 1.0
    for step in range(1, max_steps+1):
        f.zero_grads()

        # Compute the current loss and backpropagate it to the initial parameters
        curr_loss, x_predict = run_forward_loss(eps_schedule(step-1))
        curr_loss.backward()
        
        # Detach the parameters once gradient for this iteration is computed
        with torch.no_grad():
            # Report current state
            if callback:
                callback(step, u, x, f, x_predict, curr_loss.item())

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
                    trial_loss, trial_x_predict = run_forward_loss(eps_schedule(step-1))
                    if trial_loss < curr_loss:
                        print("curr_loss",curr_loss)
                        curr_loss = trial_loss
                        x_predict = trial_x_predict
                        break
                except InfeasibleOrUnboundedError:
                    pass  # linprog_feasible failed
                except RuntimeError:
                    raise

                # Otherwise reduce step size by factor beta
                step_size *= beta
            else:
                # If step size got to floating point zero, then give up
                f.weights.data[:] = w0
                #print("Stopping early due to step_size < 1e-16.")
                break
        # if curr_loss <eps:
        #     break
    # Report final state
    if callback:
        curr_loss, x_predict = run_forward_loss(eps)
        callback(None, u, x, f, x_predict, curr_loss.item())

    return curr_loss.detach(), x_predict.detach(), f.weights.detach()

def inverse_parametric_linprog_batch_step_printer(want_details=False):
    """Returns a callback that prints the state of each inverse_parametric_linprog step."""

    def callback(step, u, x, f, x_predicts, loss):
        step_str = "done" if step is None else "%04d" % step
        print("inverse_parametric_linprog[%s]: loss=%.6f %s" % (step_str, loss, f))
        
    return callback

def custom_linprog_batch(**custom_linprog_kwargs):
    """Returns a callable linprog() with custom default settngs.

    Useful for specifying a new default eps, for example.
    """
    def callback(*args, **kwargs):
        kwargs.update(custom_linprog_kwargs)
        return linprog_batch(*args, **kwargs)
    return callback