var path = require("path");
var version = require("./package.json").version;

// Custom webpack rules are generally the same for all webpack bundles, hence
// stored in a separate local variable.
var rules = [{ test: /\.css$/, use: ["style-loader", "css-loader"] }];

module.exports = (env, argv) => {
  var devtool = argv.mode === "development" ? "source-map" : false;
  return [
    {
      // Notebook extension
      //
      // This bundle only contains the part of the JavaScript that is run on
      // load of the notebook. This section generally only performs
      // some configuration for requirejs, and provides the legacy
      // "load_ipython_extension" function which is required for any notebook
      // extension.
      //
      entry: "./lib/extension.js",
      output: {
        filename: "extension.js",
        path: path.resolve(__dirname, "..", "open3d", "nbextension"),
        libraryTarget: "amd",
        publicPath: "", // publicPath is set in extension.js
      },
      devtool,
    },
    {
      // Bundle for the notebook containing the custom widget views and models
      //
      // This bundle contains the implementation for the custom widget views and
      // custom widget.
      // It must be an amd module
      //
      entry: "./lib/index.js",
      output: {
        filename: "index.js",
        path: path.resolve(__dirname, "..", "open3d", "nbextension"),
        libraryTarget: "amd",
        publicPath: "",
      },
      devtool,
      module: {
        rules: rules,
      },
      externals: ["@jupyter-widgets/base"],
    },
    {
      // Embeddable open3d bundle
      //
      // This bundle is generally almost identical to the notebook bundle
      // containing the custom widget views and models.
      //
      // The only difference is in the configuration of the webpack public path
      // for the static assets.
      //
      // It will be automatically distributed by unpkg to work with the static
      // widget embedder.
      //
      // The target bundle is always `dist/index.js`, which is the path required
      // by the custom widget embedder.
      //
      entry: "./lib/embed.js",
      output: {
        filename: "index.js",
        path: path.resolve(__dirname, "dist"),
        libraryTarget: "amd",
        publicPath: "https://unpkg.com/open3d@" + version + "/dist/",
      },
      devtool,
      module: {
        rules: rules,
      },
      externals: ["@jupyter-widgets/base"],
    },
  ];
};
