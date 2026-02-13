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
Competitive Research Workflow
=============================

Navigates to Wikipedia to research multiple related technology topics,
extracts structured information from each, and compiles a comparative
research report.  Demonstrates multi-page navigation, repeated extraction,
data aggregation, and comprehensive error handling.

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


TOPICS = [
    ("Python (programming language)", "https://en.wikipedia.org/wiki/Python_(programming_language)"),
    ("Rust (programming language)", "https://en.wikipedia.org/wiki/Rust_(programming_language)"),
    ("Go (programming language)", "https://en.wikipedia.org/wiki/Go_(programming_language)"),
]

EXTRACT_QUERY = (
    "Extract the following about this programming language: "
    "name, initial release year, original designer(s), typing discipline, "
    "paradigm(s), latest stable version, and a one-sentence description."
)


async def research_topic(browser: FlyBrowser, name: str, url: str) -> dict:
    """Navigate to a Wikipedia article and extract structured data."""
    print(f"  Researching: {name}")
    await browser.goto(url)

    result = await browser.extract(EXTRACT_QUERY)

    if result.success:
        print(f"    Extracted successfully ({result.llm_usage.total_tokens} tokens)")
        result.pprint()
        return {"topic": name, "url": url, "data": result.data, "error": None}

    print(f"    Extraction failed: {result.error}")
    return {"topic": name, "url": url, "data": None, "error": result.error}


async def compile_report(entries: list[dict]) -> str:
    """Build a plain-text comparative report from extracted entries."""
    lines = [
        "=" * 64,
        "COMPETITIVE RESEARCH REPORT  --  Programming Languages",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Topics researched: {len(entries)}",
        "=" * 64,
    ]
    for entry in entries:
        lines.append(f"\n--- {entry['topic']} ---")
        if entry["data"]:
            if isinstance(entry["data"], dict):
                for key, value in entry["data"].items():
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {entry['data']}")
        else:
            lines.append(f"  [ERROR] {entry['error']}")
    lines.append("\n" + "=" * 64)
    return "\n".join(lines)


async def main() -> None:
    """Run the full competitive research workflow."""
    provider = os.getenv("FLYBROWSER_LLM_PROVIDER", "anthropic")
    model = os.getenv("FLYBROWSER_LLM_MODEL", "claude-sonnet-4-5-20250929")

    print("=" * 64)
    print("Competitive Research Workflow")
    print(f"Provider: {provider}  |  Model: {model}")
    print("=" * 64)

    results: list[dict] = []
    total_tokens = 0

    async with FlyBrowser(llm_provider=provider, llm_model=model, headless=True) as browser:
        for name, url in TOPICS:
            entry = await research_topic(browser, name, url)
            results.append(entry)

        # Use observe to confirm we are still on a valid Wikipedia page
        obs = await browser.observe("Is this a Wikipedia article page?")
        if obs.success:
            print(f"\n  Final page observation: {obs.data}")

        usage = browser.get_usage_summary()
        total_tokens = usage.get("total_tokens", 0)

    # Compile and display the report
    report = await compile_report(results)
    print(report)

    # Persist the raw data to JSON
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "total_tokens": total_tokens,
        "topics": results,
    }
    filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nRaw data saved to: {filename}")

    # Summary statistics
    succeeded = sum(1 for r in results if r["data"] is not None)
    print(f"\nSummary: {succeeded}/{len(results)} topics extracted successfully")
    print(f"Total LLM tokens used: {total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
