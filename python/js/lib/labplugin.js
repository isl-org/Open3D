var plugin = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
  id: 'open3d:plugin',
  requires: [base.IJupyterWidgetRegistry],
  activate: (app, widgets) => {
    widgets.registerWidget({
      name: 'open3d',
      version: plugin.version,
      exports: plugin
    });
  },
  autoStart: true
};
