import ipywidgets as widgets
from traitlets import Unicode, validate, TraitError
from IPython.display import display

# See js/lib/web_visualizer.js for the frontend counterpart to this file.


@widgets.register
class WebVisualizer(widgets.DOMWidget):
    """An example widget."""

    # Name of the widget view class in front-end
    _view_name = Unicode('WebVisualizerView').tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode('WebVisualizerModel').tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode('open3d').tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode('open3d').tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)
    # Version of the front-end module containing widget model
    _model_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)

    # Widget specific property.
    # Widget properties are defined as traitlets. Any property tagged with `sync=True`
    # is automatically synced to the frontend *any* time it changes in Python.
    # It is synced back to Python from the frontend *any* time the model is touched.
    window_uid = Unicode("window_UNDEFINED", help="Window UID").tag(sync=True)

    def show(self):
        display(self)

    @validate('window_uid')
    def _valid_window_uid(self, proposal):
        if proposal['value'][:7] != "window_":
            raise TraitError('window_uid must be "window_xxx".')
        return proposal['value']
