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
Data Comparison Workflow
========================

Extracts information from two different pages on the same site (Wikipedia
articles about related countries) and compares the structured data to
surface differences.  Demonstrates same-site multi-page extraction, data
normalization, diff logic, and thorough LLM usage reporting.

Environment variables:
    ANTHROPIC_API_KEY          - API key for the configured LLM provider
    FLYBROWSER_LLM_PROVIDER   - LLM provider (default: "anthropic")
    FLYBROWSER_LLM_MODEL      - LLM model   (default: "claude-sonnet-4-5-20250929")
"""

import asyncio
import json
import os
from datetime import datetime, timezone

from flybrowser import FlyBrowser


PAGE_A = {
    "label": "United States",
    "url": "https://en.wikipedia.org/wiki/United_States",
}
PAGE_B = {
    "label": "Canada",
    "url": "https://en.wikipedia.org/wiki/Canada",
}

EXTRACT_QUERY = (
    "From the infobox and article, extract: official name, capital city, "
    "largest city, population (approximate), area in km2, official language(s), "
    "currency, government type, and GDP (nominal, approximate)."
)


async def extract_country_data(browser: FlyBrowser, page: dict) -> dict:
    """Navigate to a Wikipedia country article and extract key facts."""
    print(f"  Extracting data for: {page['label']}")
    await browser.goto(page["url"])

    result = await browser.extract(EXTRACT_QUERY)

    if result.success:
        print(f"    OK  ({result.llm_usage.total_tokens} tokens, "
              f"{result.execution.duration_seconds:.1f}s)")
        result.pprint()
        return result.data if isinstance(result.data, dict) else {"raw": result.data}

    print(f"    FAILED: {result.error}")
    return {}


def compare_datasets(label_a: str, data_a: dict, label_b: str, data_b: dict) -> list[dict]:
    """
    Compare two dictionaries field-by-field and return a list of differences.
    Fields present in one but missing in the other are noted as well.
    """
    diffs: list[dict] = []
    all_keys = sorted(set(list(data_a.keys()) + list(data_b.keys())))

    for key in all_keys:
        val_a = data_a.get(key)
        val_b = data_b.get(key)

        if val_a is None and val_b is not None:
            diffs.append({"field": key, label_a: "(missing)", label_b: str(val_b)})
        elif val_b is None and val_a is not None:
            diffs.append({"field": key, label_a: str(val_a), label_b: "(missing)"})
        elif str(val_a).strip().lower() != str(val_b).strip().lower():
            diffs.append({"field": key, label_a: str(val_a), label_b: str(val_b)})

    return diffs


def print_comparison_table(label_a: str, label_b: str, diffs: list[dict]) -> None:
    """Render the comparison as a fixed-width table."""
    col_w = 30
    print("\n" + "=" * (col_w * 3 + 6))
    print(f"  {'Field':<{col_w}} {label_a:<{col_w}} {label_b:<{col_w}}")
    print("  " + "-" * (col_w * 3 + 2))

    for d in diffs:
        field = d["field"][:col_w - 1]
        va = str(d.get(label_a, ""))[:col_w - 1]
        vb = str(d.get(label_b, ""))[:col_w - 1]
        print(f"  {field:<{col_w}} {va:<{col_w}} {vb:<{col_w}}")

    print("=" * (col_w * 3 + 6))


async def main() -> None:
    """Run the data comparison workflow."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    print("=" * 64)
    print("Data Comparison Workflow")
    print(f"Comparing: {PAGE_A['label']} vs {PAGE_B['label']}")
    print(f"Provider: {provider}  |  Model: {model}")
    print("=" * 64)

    data_a: dict = {}
    data_b: dict = {}

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        data_a = await extract_country_data(browser, PAGE_A)
        data_b = await extract_country_data(browser, PAGE_B)

        # Take a final screenshot for the audit trail
        screenshot = await browser.screenshot()
        print(f"\n  Screenshot captured ({len(screenshot.get('data_base64', ''))} bytes)")

        usage = browser.get_usage_summary()

    # Compare the two datasets
    if data_a and data_b:
        diffs = compare_datasets(PAGE_A["label"], data_a, PAGE_B["label"], data_b)
        print_comparison_table(PAGE_A["label"], PAGE_B["label"], diffs)
        print(f"\nTotal differences found: {len(diffs)}")
    else:
        print("\nComparison skipped -- one or both extractions failed.")
        diffs = []

    # Persist results
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "total_tokens": usage.get("total_tokens", 0),
        "page_a": {"label": PAGE_A["label"], "url": PAGE_A["url"], "data": data_a},
        "page_b": {"label": PAGE_B["label"], "url": PAGE_B["url"], "data": data_b},
        "differences": diffs,
    }
    filename = f"data_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nResults saved to: {filename}")
    print(f"Total LLM tokens: {usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
