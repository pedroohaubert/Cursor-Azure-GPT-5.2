"""The app module, containing the app factory function."""

from flask import Flask
from rich.traceback import install as install_rich_traceback

from . import commands
from .blueprint import blueprint, init_registry


def create_app(config_object="app.settings"):
    """Create application factory, as explained here: http://flask.pocoo.org/docs/patterns/appfactories/.

    :param config_object: The configuration object to use.
    """
    app = Flask(__name__.split(".")[0])
    app.config.from_object(config_object)
    configure_logging(app)
    configure_registry(app)
    register_commands(app)
    register_blueprints(app)
    return app


def configure_registry(app):
    """Initialize the model registry."""
    init_registry(app.config["MODEL_CONFIG_PATH"])


def register_blueprints(app):
    """Register Flask blueprints."""
    app.register_blueprint(blueprint)
    return None


def register_commands(app):
    """Register Click commands."""
    app.cli.add_command(commands.test)
    app.cli.add_command(commands.lint)


def configure_logging(app):
    """Configure logging."""
    install_rich_traceback()
