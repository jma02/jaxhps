"""Shared HTML fragments for the scattering reports."""

import numpy as np


def sphere_mesh(center, radius, n=40):
    th = np.linspace(0, np.pi, n)
    ph = np.linspace(0, 2 * np.pi, 2 * n)
    th, ph = np.meshgrid(th, ph)
    x = center[0] + radius * np.sin(th) * np.cos(ph)
    y = center[1] + radius * np.sin(th) * np.sin(ph)
    z = center[2] + radius * np.cos(th)
    return x, y, z


def figure_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)


def table_html(headers, rows):
    """Render report-authored cells, which may contain math or HTML."""
    header = "".join(f"<th>{cell}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><tr>{header}</tr>{body}</table>"
