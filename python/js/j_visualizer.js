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
var _ = require("lodash");
var THREE = require("three");
var OrbitControls = require("three-orbit-controls")(THREE);

// Webpack automatically resolves path for assets
import disc_path from "./assets/disc.png";

var JVisualizerModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name: "JVisualizerModel",
        _model_module: "open3d",
        _model_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
        _view_name: "JVisualizerView",
        _view_module: "open3d",
        _view_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
        geometry_jsons: []
    })
});

var JVisualizerView = widgets.DOMWidgetView.extend({
    camera: null,
    control: null,
    scene: null,
    renderer: null,
    geometry: null,

    initialize: function() {
        // TODO: fix frozen issue when creating multiple views
        widgets.DOMWidgetView.prototype.initialize.apply(this, arguments);
    },

    remove: function() {
        widgets.DOMWidgetView.prototype.remove.apply(this, arguments);
    },

    initEnvironment: function() {
        console.log("[called] initEnvironment");

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ alpha: true });
        this.renderer.setSize(800, 600);

        // Collect vertices from point clouds
        // TODO: handle different geometries
        // TODO: handle multiple geometries, some with color, some without
        let vertices = [];
        let colors = [];
        var geometry_idx;
        for (
            geometry_idx = 0;
            geometry_idx < this.model.get("geometry_jsons").length;
            geometry_idx++
        ) {
            let geometry_json = this.model.get("geometry_jsons")[geometry_idx];
            if (geometry_json["type"] != "PointCloud") {
                throw "Only support PointCloud in JS currently.";
            }
            // TODO: speed up
            for (var i = 0; i < geometry_json["points"].length; i++) {
                vertices.push(geometry_json["points"][i]);
            }
            for (var i = 0; i < geometry_json["colors"].length; i++) {
                colors.push(geometry_json["colors"][i]);
            }
        }

        // Geometry
        this.geometry = new THREE.BufferGeometry();
        this.geometry.addAttribute(
            "position",
            new THREE.Float32BufferAttribute(vertices, 3)
        );
        if (colors.length != 0) {
            this.geometry.addAttribute(
                "color",
                new THREE.Float32BufferAttribute(colors, 3)
            );
        }

        // Material
        let sprite = new THREE.TextureLoader().load(disc_path);
        let material = new THREE.PointsMaterial({
            size: 10,
            sizeAttenuation: false,
            map: sprite,
            alphaTest: 0.5,
            transparent: true
        });
        if (colors.length != 0) {
            material.vertexColors = THREE.VertexColors;
        } else {
            material.color.setRGB(0, 0, 0);
        }
        let particles = new THREE.Points(this.geometry, material);

        // Scene
        this.scene = new THREE.Scene();
        this.scene.add(particles);
        this.scene.background = null;

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            35, // Field of view
            800 / 600, // Aspect ratio
            0.1, // Near plane
            10000 // Far plane
        );
        this.camera.position.set(0, 0, 1);
        this.camera.lookAt(this.scene.position);

        // Control
        this.controls = new OrbitControls(
            this.camera,
            this.renderer.domElement
        );
    },

    animate: function() {
        // console.log("[called] animate")
        let self = this;
        requestAnimationFrame(function() {
            self.animate();
        });
        this.controls.update();
        this.renderCanvas();
    },

    renderCanvas: function() {
        // console.log("[called] renderCanvas")
        this.renderer.render(this.scene, this.camera);
    },

    // Called at IPython.display, i.e. at JVisualizer.show()
    render: function() {
        console.log("[called] render");
        this.setupEventListeners();
        this.valueChanged();
        this.model.on("change:geometry_jsons", this.valueChanged, this);

        this.initEnvironment();
        this.el.appendChild(this.renderer.domElement);
        this.animate();
    },

    valueChanged: function() {
        console.log("[called] valueChanged");
        this.initEnvironment();
    },

    // TODO: consider using this mechanism to send geometry data
    setupEventListeners: function() {
        this.listenTo(
            this.model,
            "msg:custom",
            this.onCustomMessage.bind(this)
        );
    },

    onCustomMessage: function(content, buffers) {
        switch (content.type) {
            // case "dog":
            //     console.log("dog received with name " + content.name);
            //     break;
            default:
                console.error("ERROR: invalid custom message", content);
        }
    }
});

// Since we have `import`, cannot use common.js's module.exports, use ES6's way
export { JVisualizerModel, JVisualizerView };
