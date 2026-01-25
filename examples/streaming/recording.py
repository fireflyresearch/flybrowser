"""
Example: Content Recording

Demonstrates recording browser sessions for tutorials, demos, or evidence.
This matches the recording examples from the README.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
from pathlib import Path
from flybrowser import FlyBrowser


async def basic_recording_example():
    """Basic recording of a browser session."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Navigate to page
        await browser.goto("https://news.ycombinator.com")
        
        # Start recording
        print("Starting recording...")
        await browser.start_recording(codec="h265", quality="high")
        
        # Perform actions to record
        await browser.act("scroll down slowly")
        await asyncio.sleep(3)
        
        posts = await browser.extract("Get the top 5 post titles")
        print(f"Extracted {len(posts.data)} posts")
        
        await browser.act("scroll down more")
        await asyncio.sleep(3)
        
        # Stop recording
        recording = await browser.stop_recording()
        
        print(f"\nRecording complete!")
        print(f"  Recording ID: {recording['recording_id']}")
        print(f"  Duration: {recording.get('duration_seconds', 'N/A')}s")
        print(f"  File size: {recording.get('file_size_mb', 'N/A')} MB")


async def tutorial_recording():
    """Record a tutorial walkthrough."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=False,  # Show browser for tutorial
    ) as browser:
        # Start recording immediately
        await browser.start_recording(
            codec="h264",
            quality="high",
            include_audio=False
        )
        
        print("Recording tutorial: How to search Hacker News")
        
        # Step 1: Navigate
        await browser.goto("https://news.ycombinator.com")
        await asyncio.sleep(2)
        
        # Step 2: Click search
        await browser.act("click the search link at the top")
        await asyncio.sleep(2)
        
        # Step 3: Search
        await browser.act("type 'AI' in the search box and press Enter")
        await asyncio.sleep(3)
        
        # Step 4: Review results
        await browser.act("scroll through the search results")
        await asyncio.sleep(3)
        
        # Stop recording
        recording = await browser.stop_recording()
        
        print(f"\nTutorial recording saved!")
        print(f"  ID: {recording['recording_id']}")
        print(f"  Path: {recording.get('file_path', 'N/A')}")


async def demo_with_checkpoints():
    """Record a demo with pause/resume capability."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto("https://news.ycombinator.com")
        
        # Recording session 1
        print("Recording Part 1: Browse front page")
        await browser.start_recording(quality="medium")
        
        await browser.act("scroll down to see more stories")
        await asyncio.sleep(3)
        
        recording1 = await browser.stop_recording()
        print(f"  Part 1 saved: {recording1['recording_id']}")
        
        # Pause, then recording session 2
        await asyncio.sleep(1)
        
        print("\nRecording Part 2: View comments")
        await browser.start_recording(quality="medium")
        
        await browser.act("click on the first story's comment link")
        await asyncio.sleep(2)
        
        await browser.act("scroll through the comments")
        await asyncio.sleep(3)
        
        recording2 = await browser.stop_recording()
        print(f"  Part 2 saved: {recording2['recording_id']}")
        
        return [recording1, recording2]


async def recording_with_extraction():
    """Record while extracting data."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Start recording
        await browser.start_recording(
            codec="h265",  # Better compression
            quality="medium"
        )
        
        print("Recording + Data Extraction Demo")
        
        # Navigate
        await browser.goto("https://news.ycombinator.com")
        await asyncio.sleep(2)
        
        # Extract while recording
        stories = await browser.extract(
            "Get all story titles and scores",
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "score": {"type": "integer"}
                    }
                }
            }
        )
        
        print(f"Extracted {len(stories.data)} stories while recording")
        
        # Continue recording
        await browser.act("scroll down")
        await asyncio.sleep(2)
        
        # Stop recording
        recording = await browser.stop_recording()
        
        print(f"\nRecording: {recording['recording_id']}")
        print(f"Data extracted: {len(stories.data)} items")
        
        return {
            "recording": recording,
            "data": stories.data
        }


async def main():
    """Main entry point."""
    print("=" * 60)
    print("Recording Examples")
    print("=" * 60)
    
    print("\n--- Basic Recording ---")
    await basic_recording_example()
    
    print("\n--- Tutorial Recording ---")
    await tutorial_recording()
    
    print("\n--- Recording with Checkpoints ---")
    await demo_with_checkpoints()
    
    print("\n--- Recording + Extraction ---")
    await recording_with_extraction()


if __name__ == "__main__":
    asyncio.run(main())
