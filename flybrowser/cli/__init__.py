# Copyright 2026 Firefly Software Solutions Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlyBrowser CLI tools.

This module provides command-line tools for:
- Unified CLI (flybrowser command)
- Interactive REPL for browser automation
- Project setup and configuration
- Service management
- Cluster deployment
- Administrative commands

CLI Entry Points:
    flybrowser           Unified CLI (recommended)
    flybrowser-setup     Setup and configuration
    flybrowser-serve     API server
    flybrowser-cluster   Cluster management
    flybrowser-admin     Administrative tasks
"""

from flybrowser.cli.setup import setup_wizard, generate_config

__all__ = [
    "setup_wizard",
    "generate_config",
]

