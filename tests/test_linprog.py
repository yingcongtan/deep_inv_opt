# Copyright (C) to Yingcong Tan, Andrew Delong, Daria Terekhov. All Rights Reserved.

import unittest
import numpy as np
import deep_inv_opt as io


class test_linprog(unittest.TestCase):

    def assertVectorEqual(self, x, y, places=3):
        x = np.array(x).ravel()
        y = np.array(y).ravel()
        np.testing.assert_almost_equal(x, y, decimal=places)

    def assertStrictlyFeasible(self, x, A_ub, b_ub, A_eq=None, b_eq=None):
        # Check strictly feasible with respect to INEQUALITIES
        for a, b in zip(A_ub, b_ub):
            self.assertLess(a @ x, b)

        # Check feasible with respect to EQUALITIES
        if A_eq is not None:
            for a, b in zip(A_eq, b_eq):
                np.testing.assert_almost_equal(a @ x, b)


    def test_linprog_feasible(self):

        # Sample problem instance
        A_ub = io.tensor([[-2.0, -5.0],
                          [-2.0,  3.0],
                          [-2.0, -1.0],
                          [ 2.0,  1.0]])
        b_ub = io.tensor([[-10.0],
                          [  6.0],
                          [ -4.0],
                          [ 10.0]])
        A_eq = io.tensor([[-0.05, 1.0]])
        b_eq = io.tensor([[1.0]])

        # Compute a feasible point
        x = io.linprog_feasible(A_ub, b_ub, A_eq, b_eq)
        self.assertStrictlyFeasible(x, A_ub, b_ub, A_eq, b_eq)
        
        # Check that infeasibility is detected due to EQUALITIES
        A_eq2 = io.tensor([[0.0, 1.0],
                           [1.0, 0.0]])
        b_eq2 = io.tensor([[5.0],
                           [5.0]])
        with self.assertRaises(RuntimeError):
            x = io.linprog_feasible(A_ub, b_ub, A_eq2, b_eq2)

        # Check that infeasibility is detected due to INEQUALITIES
        A_ub2 = io.tensor([[-1.0, -1.0],
                           [ 1.0,  1.0]])
        b_ub2 = io.tensor([[-1.0],
                           [ 1.0]])
        with self.assertRaises(RuntimeError):
            x = io.linprog_feasible(A_ub2, b_ub2, A_eq, b_eq)

        # Check that a feasible point is detected under multiple EQUALITIES
        A_ub3 = io.tensor([[-1.0, 0.0],   # x1 >= 0
                           [ 0.0,-1.0]])  # x2 >= 0
        b_ub3 = io.tensor([[0.0],
                           [0.0]])
        x = io.linprog_feasible(A_ub3, b_ub3, A_eq2, b_eq2)
        self.assertVectorEqual(x, [5, 5])


    def test_linprog_feasible_unbounded(self):

        # Basic instance: x1 >= 1 and x2 >= 3
        A_ub = io.tensor([[-1.0,  0.0],
                          [ 0.0, -1.0]])
        b_ub = io.tensor([[-1.0],
                          [-3.0]])
        A_eq = io.tensor([[-1.0, 1.0]])
        b_eq = io.tensor([[0.5]])

        # (0, 0) is infeasible
        x = io.linprog_feasible(A_ub, b_ub)
        self.assertStrictlyFeasible(x, A_ub, b_ub)

        # (0, 0) is feasible
        x = io.linprog_feasible(A_ub, -b_ub)
        self.assertStrictlyFeasible(x, A_ub, -b_ub)

        # Large coefficients
        x = io.linprog_feasible(10000*A_ub, 10000*b_ub)
        self.assertStrictlyFeasible(x, 10000*A_ub, 10000*b_ub)

        # Equality constraints
        x = io.linprog_feasible(A_ub, b_ub, A_eq, b_eq)
        self.assertStrictlyFeasible(x, A_ub, b_ub, A_eq, b_eq)


    def test_linprog(self):

        # Sample problem instance
        c = io.tensor([[-1.0],
                       [ 0.0]])
        A_ub = io.tensor([[-2.0, -5.0],
                          [-2.0,  3.0],
                          [ 2.0,  1.0]])
        b_ub = io.tensor([[-10.0],
                          [  6.0],
                          [ 10.0]])
        A_eq = io.tensor([[0.0, 1.0]])
        b_eq = io.tensor([[1.0]])

        # Check INEQUALITIES only
        x = io.linprog(c, A_ub, b_ub)
        self.assertVectorEqual(x, [5.0, 0.0])

        # Check INEQUALITIES and EQUALITIES
        x = io.linprog(c, A_ub, b_ub, A_eq, b_eq)
        self.assertVectorEqual(x, [4.5, 1.0])

        # Check that linprog doesn't blow up when c is normal to constraints
        for ai in A_ub:
            x = io.linprog(-ai.view(-1, 1), A_ub, b_ub)


    def test_inverse_linprog(self):

        # Sample problem instance
        c = io.tensor([[1.0],
                       [1.0]])
        A_ub = io.tensor([[-2.0, -5.0],
                          [-2.0,  3.0],
                          [ 2.0,  1.0]])
        b_ub = io.tensor([[-10.0],
                          [  6.0],
                          [ 10.0]])
        A_eq = io.tensor([[0.0, 1.0]])
        b_eq = io.tensor([[1.0]])

        x_target = io.tensor([[5.0],
                              [0.0]])

        # Check that solution before training is (0.0, 2.0)
        x = io.linprog(c, A_ub, b_ub)
        self.assertVectorEqual(x, [0.0, 2.0])

        # Check INEQUALITIES only
        params = io.inverse_linprog(x_target, c, A_ub, b_ub)
        x = io.linprog(*params)
        self.assertVectorEqual(x, [5.0, 0.0])

        # Check INEQUALITIES and EQUALITIES
        params = io.inverse_linprog(x_target, c, A_ub, b_ub, A_eq, b_eq, eps_decay=True)  # This one needs eps_decay to work
        x = io.linprog(*params)
        self.assertVectorEqual(x, [4.5, 1.0])

        # Check ABS DUALITY GAP with INEQUALITIES and EQUALITIES
        params = io.inverse_linprog(x_target, c, A_ub, b_ub, A_eq, b_eq, loss=io.abs_duality_gap)
        x = io.linprog(*params)
        self.assertVectorEqual(x, [4.5, 1.0])  # Currently linprog doesn't get 3 decimals correct, but ok

        c_inv, *constraints = io.inverse_linprog(x_target, c, A_ub, b_ub, A_eq, b_eq, loss=io.abs_duality_gap)
        x = io.linprog(c_inv, *constraints)
        self.assertAlmostEqual((c_inv.t() @ (x - x_target)).item(), 0)

        # Check MULTIPOINT finds solution with minimal average error,
        # regardless if initial c has support of one target point
        # (Note this doesn't work for abs_duality_gap yet; reasons unknown)
        x_target = io.tensor([[0.0, 5.0, 4.9, 4.8],
                              [2.0, 0.0, 0.1, 0.2]])
        params = io.inverse_linprog(x_target, c, A_ub, b_ub)
        x = io.linprog(*params)
        self.assertVectorEqual(x, [4.85, 0.06], places=2)  # linprog's line-search manages to get IPM to generate a non-vertex

if __name__ == '__main__':
    unittest.main()
