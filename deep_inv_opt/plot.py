# Copyright (C) to Yingcong Tan, Andrew Delong, Daria Terekhov. All Rights Reserved.

import numpy as np
import matplotlib.pyplot as plt
from .util import as_numpy
from .util import enumerate_polytope_vertices
from .linprog import _collect_linprog_central_path


def _plot_line2d(ax, a, b, c, *args, **kwargs):
    # Plot the line a*x + b*y = c
    if abs(a) < abs(b):
        x = np.linspace(-1000, 1000, 2)
        y = (c - a*x) / b
    else:
        y = np.linspace(-1000, 1000, 2)
        x = (c - b*y) / a
    ax.plot(x, y, *args, **kwargs)


def _new_linprog_figure():
    fig = plt.figure(dpi=80, figsize=(5, 5))
    ax = fig.add_subplot(111)
    return fig, ax


def _plot_linprog_constraints(ax, A_ub, b_ub, A_eq, b_eq):
    # Plot inequality constraints
    for a, b in zip(A_ub, b_ub):
        _plot_line2d(ax, a[0], a[1], b, '-', color=[0, 0, 0])

    # Plot equality constraints
    if A_eq is not None:
        for a, b in zip(A_eq, b_eq):
            _plot_line2d(ax, a[0], a[1], b, ':', color=[.5, .5, .5])


def _plot_linprog_c(ax, c):
    # Plot c as an arrow
    cx, cy = 0.1, 0.1
    c0, c1 = c.ravel()
    ax.arrow(cx, cy, c0, c1, head_width=0.05, color='b', length_includes_head=True)
    ax.text(cx + 0.08, cy - 0.1, "c=(%.3f, %.3f)" % (c0, c1), color='b')
    return cx, cy


def _plot_linprog_x(ax, x):
    if x is not None:
        ax.plot(x[0], x[1], 'or', markersize=8)
        ax.text(x[0] + 0.3, x[1] + 0.0, "(%.3f, %.3f)" % (x[0], x[1]), color='r')


def plot_linprog(c, A_ub, b_ub, A_eq=None, b_eq=None, xylim=None, title=None, axes=None):
    """Plots the constraints and c vector of an LP."""
    c, A_ub, b_ub, A_eq, b_eq = as_numpy(c, A_ub, b_ub, A_eq, b_eq)
    assert len(c) == 2

    # If no axes specified, assume user wants new figure
    if axes is None:
        _, axes = _new_linprog_figure()

    # Configure the plot
    if xylim:
        xlim, ylim = xylim
        axes.set_xlim(*xlim)
        axes.set_ylim(*ylim)
    if title:
        axes.set_title(title)
    axes.set_aspect('equal', 'box')

    # Plot the constraints and c vector
    _plot_linprog_constraints(axes, A_ub, b_ub, A_eq, b_eq)
    _plot_linprog_c(axes, c)
    return axes


def _plot_linprog_step(step, c, A_ub, b_ub, A_eq, b_eq, x, dx, t, xylim=None, title=None):
    c, A_ub, b_ub, A_eq, b_eq, x, dx = as_numpy(c, A_ub, b_ub, A_eq, b_eq, x, dx)
    if title is None: title = 'final' if step is None else 'step %d' % step
    if xylim is None: xylim = ((-1, 10), (-1, 10))

    fig, ax = _new_linprog_figure()
    plot_linprog(c, A_ub, b_ub, A_eq, b_eq, xylim, title=title, axes=ax)
    _plot_linprog_x(ax, x)
    return fig, ax


def _plot_linprog_path(step, c, A_ub, b_ub, A_eq, b_eq, x, dx, t, xylim=None, title=None):
    assert len(c) == 2

    # Collect this point into LINPROG_CENTRAL_PATH
    central_path = _collect_linprog_central_path(step, c, A_ub, b_ub, A_eq, b_eq, x, dx, t)

    # Only plot after steps are complete
    if step is not None:
        return

    # Create plot with constraints and c vector
    c, A_ub, b_ub, A_eq, b_eq, x, dx = as_numpy(c, A_ub, b_ub, A_eq, b_eq, x, dx)
    fig, ax = _plot_linprog_step(None, c, A_ub, b_ub, A_eq, b_eq, x, None, t, xylim=xylim, title=title)

    # Plot central path
    for _, x_path, _ in central_path:
        ax.plot(x_path[0], x_path[1], 'xr', alpha=0.3)
    
    # Plot final x
    _plot_linprog_x(ax, x)

    return fig, ax
    

def _plot_inverse_linprog_step(step,
                               c, A_ub, b_ub, A_eq, b_eq,
                               dc, dA_ub, db_ub, dA_eq, db_eq,
                               x_target, x_predict, loss,
                               xylim=None, title=None):
    
    # Convert from torch to numpy, to ensure detached form torch compute graph
    c, A_ub, b_ub, A_eq, b_eq = as_numpy(c, A_ub, b_ub, A_eq, b_eq)
    dc, dA_ub, db_ub, dA_eq, db_eq = as_numpy(dc, dA_ub, db_ub, dA_eq, db_eq)
    x_target, x_predict = as_numpy(x_target, x_predict)

    # At each step of inverse, plot the entire most recent CENTRAL_PATH on single plot
    if title is None:
        title = 'final' if step is None else 'step %d' % step
    fig, ax = _plot_linprog_path(None, c, A_ub, b_ub, A_eq, b_eq, x_predict, None, None, xylim=xylim, title=title)
    
    # Plot x_target
    ax.plot(x_target[0], x_target[1], 'ok', markersize=10, fillstyle='none')
    
    # Plot dc as arrow
    #if dc is not None:
    #    dc0, dc1 = dc.ravel()
    #    ax.arrow(cx+c0, cy+c1, dc0, dc1, head_width=0.05, color='g', length_includes_head=True)
    #    ax.text(cx+c0+dc0+0.05, cy+c1+dc1+0.05, "dc=(%.3f, %.3f)" % (dc0, dc1), color='g')

    return fig, ax


def linprog_step_plotter(xylim=None, title=None):
    """Returns a callback that plots each step of linprog."""

    def callback(*args, **kwargs):
        kwargs.setdefault('xylim', xylim)
        kwargs.setdefault('title', title)
        _plot_linprog_step(*args, **kwargs)

    return callback


def linprog_path_plotter(xylim=None, title=None):
    """Returns a callback that plots the central path of linprog."""

    def callback(*args, **kwargs):
        kwargs.setdefault('xylim', xylim)
        kwargs.setdefault('title', title)
        _plot_linprog_path(*args, **kwargs)

    return callback


def inverse_linprog_step_plotter(xylim=None, title=None, frequency=1):
    """Returns a callback that plots each step of inverse_linprog."""

    def callback(step, *args, **kwargs):
        if step is None or (step-1) % frequency == 0:
            kwargs.setdefault('xylim', xylim)
            kwargs.setdefault('title', title)
            _plot_inverse_linprog_step(step, *args, **kwargs)

    return callback


def inverse_linprog_plotter(xylim=None, title=None):
    """Returns a callback that plots the initial and final step of inverse_linprog."""
    return inverse_linprog_step_plotter(xylim=xylim, title=title, frequency=100000000)


def plot_linear_program(c, A_ub, b_ub, A_eq=None, b_eq=None, color='k', alpha=1.0, linestyle='-', vertex_color=None, cxy=None, ax=None):
    """Plots a linear program cost vector c, the convex polytope represented
    by A_ub and b_ub, and optionally equality constraints represented by A_eq and b_eq.
    If vertex_style is given, also plots the vertices where inequality constraints meet.
    
    The main different from linprog_step_plotter is that the inequality constraints are not
    plotted as separate lines, but rather processed into a closed polygon.

    If no axis `ax` is specified, the current axis is used
    """
    c, A_ub, b_ub, A_eq, b_eq = as_numpy(c, A_ub, b_ub, A_eq, b_eq)
    if ax is None:
        ax = plt.gca()

    # Draw cost vector arrow
    if c is not None:
        assert len(c) == 2, "Only 2D supported"
        cx, cy = cxy if cxy is not None else (3, 3)
        c0, c1 = c.ravel() / np.sum(c**2)**0.5  
        ax.arrow(cx, cy, c0, c1, color=color, alpha=alpha, head_width=0.05, length_includes_head=True)  

    if A_ub is not None:
        n, m = A_ub.shape
        assert m == 2, "Only 2D supported"
        # Plot inequality constratints
        vertices, intersect = enumerate_polytope_vertices(A_ub, b_ub)
        for i in range(n):
            pt_ind = [j for j in range(len(intersect)) if i in intersect[j]] 
            x = vertices[pt_ind,0]
            y = vertices[pt_ind,1]
            ax.plot(x, y, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.5)

        # Plot the vertices of the inequality constraints.
        if vertex_color is not None:
            ax.plot(vertices[:,0], vertices[:,1], '+', color=vertex_color, alpha=alpha)

    # Plot equality constraints
    if A_eq is not None:
        for a, b in zip(A_eq, b_eq):
            _plot_line2d(ax, a[0], a[1], b, ':', color=color, alpha=alpha, linestyle=linestyle)


def plot_targets(x_targets, *args, **kwargs):
    # If not axis specified, use current axis
    ax = kwargs.get('ax', None)
    if ax is None:
        ax = plt.gca()

    # If no alpha specified, just draw the targets as markers in a single call.
    # This is typical in non-parametric plots.
    alpha = kwargs.get('alpha', None)
    if alpha is None:
        ax.plot(as_numpy(x_targets[:,0]), as_numpy(x_targets[:,1]), *args, **kwargs)    
        return

    # Special mode emulates the fading effect for parametric, so that
    # the optima (x*) are shaded to match their constraints
    # (i.e. light shaded target matches light shaded constraints)
    del kwargs['alpha']
    color = kwargs['color']
    if color == 'k':
        color = [0, 0, 0]
    color = np.array(color, dtype=np.float64)
    white = np.array([1, 1, 1], dtype=np.float64)
    for i, xi in enumerate(x_targets):
        alpha_i = alpha + (1-alpha)*(i/(len(x_targets)-1) if len(x_targets) > 1 else 1)
        kwargs['color'] = color * alpha_i + (1-alpha_i) * white 
        ax.plot(as_numpy(x_targets[i,0]), as_numpy(x_targets[i,1]), *args, **kwargs)


