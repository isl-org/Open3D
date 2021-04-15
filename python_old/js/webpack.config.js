// Note: This file contains example code from official documentation:
// https://github.com/jupyter-widgets/widget-cookiecutter

var path = require("path");
var version = require("./package.json").version;

// Custom webpack rules are generally the same for all webpack bundles, hence
// stored in a separate local variable.
var rules = [
    { test: /\.css$/, use: ["style-loader", "css-loader"] },
    { test: /\.less$/, use: ["style-loader", "css-loader", "less-loader"] },
    {
        test: /\.(png|svg|jpg|gif)$/,
        use: ["file-loader"]
    }
];

module.exports = [
    {
        // Notebook extension
        //
        // This bundle only contains the part of the JavaScript that is run on
        // load of the notebook. This section generally only performs
        // some configuration for requirejs, and provides the legacy
        // "load_ipython_extension" function which is required for any notebook
        // extension.
        //
        entry: "./extension.js",
        output: {
            filename: "extension.js",
            path: path.resolve(__dirname, "..", "open3d", "static"),
            libraryTarget: "amd"
        }
    },
    {
        // Bundle for the notebook containing the custom widget views and models
        //
        // This bundle contains the implementation for the custom widget views and
        // custom widget.
        // It must be an amd module
        //
        entry: "./index.js",
        output: {
            filename: "index.js",
            path: path.resolve(__dirname, "..", "open3d", "static"),
            libraryTarget: "amd"
        },
        devtool: "source-map",
        module: {
            rules: rules
        },
        externals: ["@jupyter-widgets/base"]
    }
];
