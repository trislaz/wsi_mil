"""
Code largement inspiré de sklearn.metrics.
Adapté pour générer les roc_curve à partir des outputs de trislaz/deepMIL
"""
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, auc
import numpy as np

class RocCurveDisplay:
    """ROC Curve visualization.

    It is recommend to use :func:`~sklearn.metrics.plot_roc_curve` to create a
    visualizer. All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    fpr : ndarray
        False positive rate.

    tpr : ndarray
        True positive rate.

    roc_auc : float
        Area under ROC curve.

    estimator_name : str
        Name of estimator.

    Attributes
    ----------
    line_ : matplotlib Artist
        ROC Curve.

    ax_ : matplotlib Axes
        Axes with ROC Curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    Examples
    --------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([0, 0, 1, 1])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    >>> roc_auc = metrics.auc(fpr, tpr)
    >>> display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,\
                                          estimator_name='example estimator')
    >>> display.plot()  # doctest: +SKIP
    >>> plt.show()      # doctest: +SKIP
    """

    def __init__(self, fpr, tpr, roc_auc, estimator_name):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.estimator_name = estimator_name

    def plot(self, ax=None, name=None, **kwargs):
        """Plot visualization

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use the name of the
            estimator.

        Returns
        -------
        display : :class:`~sklearn.metrics.plot.RocCurveDisplay`
            Object that stores computed values.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        name = self.estimator_name if name is None else name

        line_kwargs = {
            'label': "{} (AUC = {:0.2f})".format(name, self.roc_auc), 
            'lw':0.25

        }
        line_kwargs.update(**kwargs)

        self.line_ = ax.plot(self.fpr, self.tpr, **line_kwargs)[0]
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc='lower right')

        self.ax_ = ax
        self.figure_ = ax.figure
        return self

## Greatly inspired from sklearn
def plot_roc_curve(y_proba, y_true, y_pred, sample_weight=None,
                   drop_intermediate=True, name=None, ax=None, **kwargs):
    """plot_roc_curve.

    :param y_proba: proba of the predicted value
    :param y_true: true label. list or array of size [n]. 1 dimensional.
    :param y_pred: predicted label
    :param sample_weight:
    :param drop_intermediate:
    :param name:
    :param ax:
    :param kwargs:
    """

    oe=OrdinalEncoder()
    oe.fit(np.array(y_true).reshape(-1, 1))
    y_true = oe.transform(np.array(y_true).reshape(-1,1))
    y_pred = oe.transform(np.array(y_pred).reshape(-1,1))
    y_proba = [(1-x) if y_pred[o].item() == 0 else x for o,x in enumerate(y_proba)]
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=None,
                            sample_weight=sample_weight,
                            drop_intermediate=drop_intermediate)
    roc_auc = auc(fpr, tpr)
    viz = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name
    )
    return viz.plot(ax=ax, name=name, **kwargs)
