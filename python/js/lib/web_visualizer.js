var widgets = require("@jupyter-widgets/base");
var _ = require("lodash");

// See web_visualizer.py for the kernel counterpart to this file.

// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including
//
//  - `_view_name`
//  - `_view_module`
//  - `_view_module_version`
//
//  - `_model_name`
//  - `_model_module`
//  - `_model_module_version`
//
//  when different from the base class.

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
var WebVisualizerModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: "WebVisualizerModel",
    _view_name: "WebVisualizerView",
    _model_module: "open3d",
    _view_module: "open3d",
    _model_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
    _view_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
    value: "Hello World!",
  }),
});

// Custom View. Renders the widget model.
var WebVisualizerView = widgets.DOMWidgetView.extend({
  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.value_changed();

    // Observe changes in the value traitlet in Python, and define
    // a custom callback.
    this.model.on("change:value", this.value_changed, this);
  },

  value_changed: function () {
    this.el.textContent = this.model.get("value");
  },
});

module.exports = {
  WebVisualizerModel: WebVisualizerModel,
  WebVisualizerView: WebVisualizerView,
};
