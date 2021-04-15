// Note: This file contains example code from official documentation:
// https://github.com/jupyter-widgets/widget-cookiecutter

// This file contains the javascript that is run when the notebook is loaded.
// It contains some requirejs configuration and the `load_ipython_extension`
// which is required for any notebook extension.

// Configure requirejs
if (window.require) {
    window.require.config({
        map: {
            "*": {
                open3d: "nbextensions/open3d/index"
            }
        }
    });
}

// Export the required load_ipython_extension
module.exports = {
    load_ipython_extension: function() {}
};
