"""Application settings: parameters.yaml (defaults) merged with override.yaml (the user's changes)."""
import copy
import os
import sys
import threading
import time

import yaml

from tsutils import Singleton
from tsutils import utilities


CONFIG_REFRESH_INTERVAL_SECONDS = 10


class Config(Singleton.Singleton):
    """A Singleton object with all configuration data
    """
    _default_config_filename:  str = None
    _override_config_filename: str = None
    _default_data: dict = None   # Data as read from parameters.yaml
    _override_data: dict = None  # Data as read from override.yaml
    _current_data: dict = None   # Merged data of parameters.yaml and override.yaml
    _initialized: bool = False
    _refresh_period = CONFIG_REFRESH_INTERVAL_SECONDS
    _refresh_periodically = False
    _write_lock = threading.Lock()

    def __init__(self, default_config_filename: str = 'parameters.yaml',
                 override_config_filename: str = 'override.yaml'):

        if self._initialized:
            return

        self._default_config_filename = default_config_filename
        self._override_config_filename = override_config_filename
        self.read_config_from_files()
        self._initialized = True

    def refresh(self):
        while True:
            time.sleep(CONFIG_REFRESH_INTERVAL_SECONDS)
            self.read_config_from_files()

    def read_config_from_files(self):
        """Read config data from yaml files
        """
        try:
            with open(self._default_config_filename, mode='r', encoding='utf-8') as default_config_file:
                self._default_data = yaml.load(stream=default_config_file, Loader=yaml.SafeLoader)
        except FileNotFoundError as err:
            print(f'Failed to find yaml file: {self._default_config_filename}.')
            print(f'Error: {err}')
            sys.exit(1)

        # override.yaml holds only what the user changed. A missing, empty or damaged file must
        # not stop the program from starting: the defaults are used and it is written again on
        # the next change.
        self._override_data = {}
        try:
            with open(self._override_config_filename, mode='r', encoding='utf-8') as override_config_file:
                loaded = yaml.load(stream=override_config_file, Loader=yaml.SafeLoader)
            if isinstance(loaded, dict):
                self._override_data = loaded
        except FileNotFoundError:
            pass
        except (yaml.YAMLError, OSError, UnicodeDecodeError) as err:
            print(f'[WARN] {self._override_config_filename} could not be read ({err}); using the defaults.')

        # Merge default and override values
        self._current_data = copy.deepcopy(self._default_data)
        self._current_data = utilities.merge(self._current_data, self._override_data)

    def add_override_value(self, input_dict: dict):
        """Override a default configuration parameter value
        """
        if not isinstance(input_dict, dict):
            print(f'Expected input value to be a dict. Instead received: {input_dict}')
            return
        with self._write_lock:
            # Update the override values
            self._override_data = utilities.merge(self._override_data, copy.deepcopy(input_dict))
            # update the current values
            self._current_data = utilities.merge(self._current_data, copy.deepcopy(input_dict))
            self._write_override_file()

    def _write_override_file(self):
        """Write override.yaml through a temporary file, so a crash or a full disk while
        writing cannot leave a half-written settings file behind."""
        temp_name = f'{self._override_config_filename}.tmp'
        with open(file=temp_name, mode="w", encoding='utf-8') as override_file:
            yaml.dump(self._override_data, override_file, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        os.replace(temp_name, self._override_config_filename)

    @staticmethod
    def load_alter_save(section_name, property_name, value):
        """Change one value and save it to the override file."""
        Config().add_override_value({section_name: {property_name: value}})

    @property
    def data(self) -> dict:
        """Get all configuration data read from yaml file"""
        return self._current_data

    @property
    def default_data(self) -> dict:
        """The values of parameters.yaml alone (what "reset to default" returns to)."""
        return self._default_data

    @property
    def config_override_file(self) -> str:
        """Name of the yaml file used to override configuration"""
        return self._override_config_filename

    @property
    def config_file(self) -> str:
        """Name of the yaml file with default configuration"""
        return self._default_config_filename
