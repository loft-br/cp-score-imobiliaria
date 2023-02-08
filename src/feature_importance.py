import shap


class ShapValues:

    def __init__(self, model, X, **kwargs):
        self.model = model
        self.X = X

        self.shap_values = None
        self.get_explainer()

    def get_explainer(self):

        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(self.X)

    def summary_plot(self, **kwargs):

        if 'plot_type' in kwargs:
            return shap.summary_plot(self.shap_values, self.X, **kwargs)
        else:
            return shap.summary_plot(self.shap_values[1], self.X, **kwargs)

    def dependance_plot(self, **kwargs):
        return shap.dependence_plot(
            shap_values=self.shap_values[1], features=self.X, **kwargs
        )