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

"""
Search Provider Implementations.

This module provides concrete implementations of search providers:
    - SerperProvider: Serper.dev API (recommended, best value)
    - GoogleProvider: Google Custom Search API
    - BingProvider: Bing Web Search API
"""

from flybrowser.agents.tools.search.providers.serper_provider import SerperProvider
from flybrowser.agents.tools.search.providers.google_provider import GoogleProvider
from flybrowser.agents.tools.search.providers.bing_provider import BingProvider

__all__ = [
    "SerperProvider",
    "GoogleProvider",
    "BingProvider",
]
