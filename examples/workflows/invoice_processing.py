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
Document / Table Extraction Workflow
=====================================

Navigates to Wikipedia pages containing structured tables (list of
countries by GDP and by population), extracts the tabular data, normalizes
it, and produces a formatted structured report.  Demonstrates multi-table
extraction from real public pages, data normalization, cross-referencing
between datasets, and comprehensive LLM usage tracking.

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


GDP_URL = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
POP_URL = "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population"

TOP_N = 10  # How many countries to include in the final report


async def extract_gdp_table(browser: FlyBrowser) -> list[dict]:
    """Navigate to the GDP page and extract the top countries by GDP."""
    print("  [GDP] Navigating...")
    await browser.goto(GDP_URL)

    result = await browser.extract(
        f"Extract the top {TOP_N} countries by nominal GDP from the main table. "
        "For each country include: rank, country name, and GDP value in "
        "US dollars (as shown in the table)."
    )

    if result.success:
        print(f"  [GDP] Extracted ({result.llm_usage.total_tokens} tokens, "
              f"{result.execution.duration_seconds:.1f}s)")
        result.pprint()
        data = result.data if isinstance(result.data, list) else [result.data]
        return data[:TOP_N]

    print(f"  [GDP] Extraction failed: {result.error}")
    return []


async def extract_population_table(browser: FlyBrowser) -> list[dict]:
    """Navigate to the population page and extract the top countries."""
    print("  [POP] Navigating...")
    await browser.goto(POP_URL)

    result = await browser.extract(
        f"Extract the top {TOP_N} countries by population from the main table. "
        "For each country include: rank, country name, and population figure."
    )

    if result.success:
        print(f"  [POP] Extracted ({result.llm_usage.total_tokens} tokens, "
              f"{result.execution.duration_seconds:.1f}s)")
        result.pprint()
        data = result.data if isinstance(result.data, list) else [result.data]
        return data[:TOP_N]

    print(f"  [POP] Extraction failed: {result.error}")
    return []


def cross_reference(gdp_rows: list[dict], pop_rows: list[dict]) -> list[dict]:
    """
    Merge GDP and population data by country name.  Returns a list of
    records with country, gdp, and population fields where both are known.
    """
    def normalize_name(entry: dict) -> str:
        name = entry.get("country") or entry.get("country_name") or entry.get("name") or ""
        return name.strip().lower()

    gdp_lookup: dict[str, dict] = {}
    for row in gdp_rows:
        key = normalize_name(row)
        if key:
            gdp_lookup[key] = row

    merged: list[dict] = []
    for row in pop_rows:
        key = normalize_name(row)
        if key in gdp_lookup:
            merged.append({
                "country": row.get("country") or row.get("country_name") or row.get("name"),
                "population": row.get("population") or row.get("population_figure"),
                "gdp": gdp_lookup[key].get("gdp") or gdp_lookup[key].get("gdp_value"),
            })

    return merged


def print_structured_report(
    gdp_rows: list[dict], pop_rows: list[dict], merged: list[dict]
) -> None:
    """Render the final structured report."""
    print("\n" + "=" * 64)
    print("STRUCTURED DATA EXTRACTION REPORT")
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 64)

    print(f"\nSource 1 -- GDP (top {TOP_N}):")
    for row in gdp_rows:
        name = row.get("country") or row.get("country_name") or row.get("name") or "?"
        gdp = row.get("gdp") or row.get("gdp_value") or "?"
        print(f"  {name}: {gdp}")

    print(f"\nSource 2 -- Population (top {TOP_N}):")
    for row in pop_rows:
        name = row.get("country") or row.get("country_name") or row.get("name") or "?"
        pop = row.get("population") or row.get("population_figure") or "?"
        print(f"  {name}: {pop}")

    print(f"\nCross-Referenced (countries in both lists):")
    if merged:
        for m in merged:
            print(f"  {m['country']}: GDP={m['gdp']}, Pop={m['population']}")
    else:
        print("  (no overlapping countries found)")

    print("\n" + "=" * 64)


async def main() -> None:
    """Run the document extraction workflow."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    print("=" * 64)
    print("Document / Table Extraction Workflow")
    print(f"Provider: {provider}  |  Model: {model}")
    print("=" * 64)

    gdp_rows: list[dict] = []
    pop_rows: list[dict] = []

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        gdp_rows = await extract_gdp_table(browser)
        pop_rows = await extract_population_table(browser)

        # Verify we ended on a Wikipedia page
        obs = await browser.observe("Is this a Wikipedia list article?")
        if obs.success:
            print(f"\n  Page verification: {obs.data}")

        usage = browser.get_usage_summary()

    # Cross-reference and report
    merged = cross_reference(gdp_rows, pop_rows)
    print_structured_report(gdp_rows, pop_rows, merged)

    # Persist
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "total_tokens": usage.get("total_tokens", 0),
        "gdp_data": {"url": GDP_URL, "rows": gdp_rows},
        "population_data": {"url": POP_URL, "rows": pop_rows},
        "cross_referenced": merged,
    }
    filename = f"table_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nData saved to: {filename}")
    print(f"Total LLM tokens: {usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
