// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

var widgets = require("@jupyter-widgets/base");

var JVisualizerModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _view_name: "JVisualizerView",
    _view_module: "open3d",
    _view_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
  }),
});

var JVisualizerView = widgets.DOMWidgetView.extend({
  // Render the view.
  render: function () {
    this.email_input = document.createElement("input");
    this.email_input.type = "email";
    this.email_input.value = this.model.get("value");
    this.email_input.disabled = this.model.get("disabled");

    // The `el` property is the DOM element associated with the view
    this.el.appendChild(this.email_input);

    // Python -> JavaScript update
    this.model.on("change:value", this.value_changed, this);
    this.model.on("change:disabled", this.disabled_changed, this);

    // JavaScript -> Python update
    this.email_input.onchange = this.input_changed.bind(this);
  },

  value_changed: function () {
    this.email_input.value = this.model.get("value");
  },

  disabled_changed: function () {
    this.email_input.disabled = this.model.get("disabled");
  },

  input_changed: function () {
    this.model.set("value", this.email_input.value);
    this.model.save_changes();
  },
});

// Since we have `import`, cannot use common.js's module.exports, use ES6's way
export { JVisualizerModel, JVisualizerView };
