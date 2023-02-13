import shap


class ShapValues:

    def __init__(self, model, X):
        self.model = model
        self.X = X

        self.shap_values = None
        self.shap_scatter = None

        self.get_explainer()

    def get_explainer(self):

        explainer = shap.Explainer(self.model)

        self.shap_scatter = explainer(self.X)
        self.shap_values = explainer.shap_values(self.X)

    def summary_plot(self, **kwargs):

        if "plot_type" in kwargs:
            return shap.summary_plot(self.shap_values, self.X, **kwargs)
        else:
            return shap.summary_plot(self.shap_values[1], self.X, **kwargs)

    def dependence_plot(self, **kwargs):
        return shap.dependence_plot(
            **kwargs, shap_values=self.shap_values[1], features=self.X
        )

    def scatter_plot(self, features: list, dependence=False, **kwargs):
        
        if dependence:
            return shap.plots.scatter(
                self.shap_scatter[:, features[0], 1],
                self.shap_scatter[:, features[1], 1],
                **kwargs
            )
        else:
            return shap.plots.scatter(
                self.shap_scatter[:, features[0], 1],
                **kwargs
            )
        
