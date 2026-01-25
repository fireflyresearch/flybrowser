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

"""Utility functions and helpers for FlyBrowser."""

from flybrowser.utils.text_normalizer import normalize_text, normalize_data, is_normalized
from flybrowser.utils.page_utils import (
    is_blank_page,
    is_flybrowser_blank_page,
    is_completion_page,
    is_flybrowser_completion_page,
    is_flybrowser_internal_page,
)

__all__ = [
    "normalize_text",
    "normalize_data",
    "is_normalized",
    "is_blank_page",
    "is_flybrowser_blank_page",
    "is_completion_page",
    "is_flybrowser_completion_page",
    "is_flybrowser_internal_page",
]

