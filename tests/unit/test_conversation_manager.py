# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ConversationManager, TokenEstimator, TokenBudgetManager, and chunking."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flybrowser.llm.token_budget import (
    TokenEstimator,
    TokenBudgetManager,
    ContentType,
    TokenEstimate,
    BudgetAllocation,
)
from flybrowser.llm.chunking import (
    Chunk,
    TextChunker,
    HTMLChunker,
    JSONChunker,
    SmartChunker,
    get_chunker,
)
from flybrowser.llm.conversation import (
    ConversationManager,
    ConversationMessage,
    ConversationHistory,
    MessageRole,
    AccumulationPhase,
    AccumulationContext,
)


class TestTokenEstimator:
    """Tests for TokenEstimator."""

    def test_estimate_text(self):
        """Test token estimation for plain text."""
        text = "Hello world, this is a simple test."
        estimate = TokenEstimator.estimate(text)
        
        assert estimate.tokens > 0
        assert estimate.content_type == ContentType.TEXT
        assert estimate.confidence >= 0.8
        assert estimate.raw_size == len(text)

    def test_estimate_html(self):
        """Test token estimation for HTML content."""
        html = "<div><p>Hello</p><span>World</span></div>"
        estimate = TokenEstimator.estimate(html)
        
        assert estimate.tokens > 0
        assert estimate.content_type == ContentType.HTML
        # HTML has lower confidence due to tag overhead variability
        assert estimate.confidence >= 0.8

    def test_estimate_json(self):
        """Test token estimation for JSON content."""
        data = {"name": "test", "value": 123, "items": [1, 2, 3]}
        estimate = TokenEstimator.estimate(data)
        
        assert estimate.tokens > 0
        assert estimate.content_type == ContentType.JSON
        assert estimate.confidence >= 0.8

    def test_estimate_json_string(self):
        """Test token estimation for JSON string."""
        json_str = '{"key": "value", "number": 42}'
        estimate = TokenEstimator.estimate(json_str)
        
        assert estimate.tokens > 0
        assert estimate.content_type == ContentType.JSON

    def test_estimate_code(self):
        """Test token estimation for code content."""
        code = """
def hello_world():
    return "Hello, World!"

class MyClass:
    def __init__(self):
        self.value = 0
"""
        estimate = TokenEstimator.estimate(code)
        
        assert estimate.tokens > 0
        assert estimate.content_type == ContentType.CODE

    def test_estimate_image_bytes(self):
        """Test token estimation for image bytes."""
        # Simulate small image
        image_bytes = b"x" * 50000  # ~50KB
        estimate = TokenEstimator.estimate(image_bytes)
        
        assert estimate.tokens > 0
        assert estimate.content_type == ContentType.IMAGE
        # Image estimation is less precise
        assert estimate.confidence < 0.9

    def test_estimate_with_explicit_type(self):
        """Test that explicit content type overrides detection."""
        # This looks like JSON but we force TEXT type
        json_str = '{"key": "value"}'
        estimate = TokenEstimator.estimate(json_str, ContentType.TEXT)
        
        assert estimate.content_type == ContentType.TEXT

    def test_estimate_messages(self):
        """Test token estimation for message list."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        tokens = TokenEstimator.estimate_messages(messages)
        
        assert tokens > 0
        # Should be roughly sum of content tokens + overhead per message

    def test_with_buffer(self):
        """Test token estimate with safety buffer."""
        estimate = TokenEstimate(
            tokens=100,
            content_type=ContentType.TEXT,
            confidence=0.8,
            raw_size=400,
        )
        
        buffered = estimate.with_buffer
        # Lower confidence should result in higher buffer
        assert buffered > 100
        assert buffered <= 150  # At most 50% increase


class TestTokenBudgetManager:
    """Tests for TokenBudgetManager."""

    def test_default_initialization(self):
        """Test default budget manager initialization."""
        manager = TokenBudgetManager()
        
        assert manager.context_window == 128000
        assert manager.max_output_tokens == 8192
        assert manager.safety_margin == 0.1

    def test_available_for_input(self):
        """Test available tokens calculation."""
        manager = TokenBudgetManager(
            context_window=10000,
            max_output_tokens=2000,
            safety_margin=0.1,
        )
        
        # Available = 10000 - 2000 - 1000 (10% safety) = 7000
        assert manager.available_for_input == 7000

    def test_can_fit(self):
        """Test content fitting check."""
        manager = TokenBudgetManager(context_window=10000, max_output_tokens=2000)
        
        # Short content should fit
        assert manager.can_fit("Hello world")
        
        # Very long content should not fit
        long_content = "x" * 100000  # ~25K tokens
        assert not manager.can_fit(long_content)

    def test_allocate(self):
        """Test budget allocation."""
        manager = TokenBudgetManager(context_window=10000, max_output_tokens=2000)
        
        allocation = manager.allocate(
            system_prompt="You are helpful.",
            conversation_history=[],
            current_content="What is 2+2?",
        )
        
        assert isinstance(allocation, BudgetAllocation)
        assert allocation.system_prompt > 0
        assert allocation.current_message > 0
        assert allocation.response_reserve == 2000

    def test_would_exceed_budget(self):
        """Test budget overflow detection."""
        manager = TokenBudgetManager(context_window=1000, max_output_tokens=200)
        
        # Small content should not exceed
        exceeds, overflow = manager.would_exceed_budget(
            system_prompt="Hi",
            conversation_history=[],
            current_content="Hello",
        )
        assert not exceeds
        assert overflow == 0
        
        # Large content should exceed
        large = "x" * 10000
        exceeds, overflow = manager.would_exceed_budget(
            system_prompt="",
            conversation_history=[],
            current_content=large,
        )
        assert exceeds
        assert overflow > 0

    def test_record_usage(self):
        """Test token usage recording."""
        manager = TokenBudgetManager(context_window=10000, max_output_tokens=2000)
        
        manager.record_usage(100)
        manager.record_usage(50)
        
        assert manager.total_used == 150
        
    def test_reset(self):
        """Test budget reset."""
        manager = TokenBudgetManager(context_window=10000, max_output_tokens=2000)
        manager.record_usage(500)
        
        manager.reset()
        
        assert manager.total_used == 0

    def test_get_stats(self):
        """Test statistics retrieval."""
        manager = TokenBudgetManager(context_window=10000, max_output_tokens=2000)
        manager.record_usage(100)
        manager.record_usage(200)
        
        stats = manager.get_stats()
        
        assert stats["context_window"] == 10000
        assert stats["used_tokens"] == 300
        assert stats["message_count"] == 2

    def test_calculate_chunk_size(self):
        """Test chunk size calculation."""
        manager = TokenBudgetManager(context_window=10000, max_output_tokens=2000)
        
        chunk_size = manager.calculate_chunk_size(5000, reserved_for_other=1000)
        
        # Should be 60% of (available - reserved)
        assert chunk_size > 1000
        assert chunk_size < manager.available_for_input


class TestTextChunker:
    """Tests for TextChunker."""

    def test_chunk_small_content(self):
        """Test that small content returns single chunk."""
        chunker = TextChunker()
        text = "This is a short text."
        
        chunks = chunker.chunk(text, max_tokens_per_chunk=100)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].is_first
        assert chunks[0].is_last

    def test_chunk_by_paragraphs(self):
        """Test chunking at paragraph boundaries."""
        chunker = TextChunker()
        text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content."""
        
        chunks = chunker.chunk(text, max_tokens_per_chunk=20)
        
        assert len(chunks) >= 2
        # Each chunk should be complete (no cut-off sentences)
        for chunk in chunks:
            assert chunk.content.strip()

    def test_chunk_long_paragraph(self):
        """Test chunking a single long paragraph."""
        chunker = TextChunker()
        text = "Word " * 500  # Long single paragraph
        
        chunks = chunker.chunk(text, max_tokens_per_chunk=50)
        
        assert len(chunks) > 1

    def test_chunk_indices(self):
        """Test chunk index tracking."""
        chunker = TextChunker()
        text = """First part.\n\nSecond part.\n\nThird part."""
        
        chunks = chunker.chunk(text, max_tokens_per_chunk=5)
        
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.total_chunks == len(chunks)

    def test_format_header(self):
        """Test chunk header formatting."""
        chunk = Chunk(
            content="Test",
            index=1,
            total_chunks=3,
            estimated_tokens=5,
        )
        
        header = chunk.format_header()
        
        assert "2/3" in header  # 1-indexed


class TestHTMLChunker:
    """Tests for HTMLChunker."""

    def test_chunk_html(self):
        """Test HTML chunking preserves structure."""
        chunker = HTMLChunker()
        html = """
<div>
    <p>First paragraph</p>
    <p>Second paragraph</p>
</div>
<section>
    <p>Third paragraph</p>
</section>
"""
        chunks = chunker.chunk(html, max_tokens_per_chunk=30)
        
        assert len(chunks) >= 1
        # Chunks should contain valid HTML fragments
        for chunk in chunks:
            assert chunk.content.strip()

    def test_chunk_large_html(self):
        """Test chunking large HTML content."""
        chunker = HTMLChunker()
        # Generate large HTML
        html = "".join([f"<div><p>Paragraph {i} content here.</p></div>" for i in range(100)])
        
        chunks = chunker.chunk(html, max_tokens_per_chunk=100)
        
        assert len(chunks) > 1


class TestJSONChunker:
    """Tests for JSONChunker."""

    def test_chunk_array(self):
        """Test chunking JSON arrays."""
        chunker = JSONChunker()
        data = [{"id": i, "value": f"item_{i}"} for i in range(50)]
        json_str = json.dumps(data)
        
        chunks = chunker.chunk(json_str, max_tokens_per_chunk=100)
        
        assert len(chunks) > 1
        # Each chunk should be valid JSON
        for chunk in chunks:
            parsed = json.loads(chunk.content)
            assert isinstance(parsed, list)

    def test_chunk_object(self):
        """Test chunking JSON objects."""
        chunker = JSONChunker()
        data = {f"key_{i}": f"value_{i}" * 10 for i in range(50)}
        json_str = json.dumps(data)
        
        chunks = chunker.chunk(json_str, max_tokens_per_chunk=100)
        
        assert len(chunks) >= 1
        # Each chunk should be valid JSON
        for chunk in chunks:
            parsed = json.loads(chunk.content)
            assert isinstance(parsed, dict)

    def test_invalid_json_fallback(self):
        """Test fallback for invalid JSON."""
        chunker = JSONChunker()
        invalid_json = "This is not { valid JSON at all"
        
        # Should not raise, falls back to text chunking
        chunks = chunker.chunk(invalid_json, max_tokens_per_chunk=10)
        
        assert len(chunks) >= 1


class TestSmartChunker:
    """Tests for SmartChunker."""

    def test_auto_detect_html(self):
        """Test auto-detection of HTML content."""
        chunker = SmartChunker()
        html = "<html><body><div>Content</div></body></html>"
        
        chunks = chunker.chunk(html, max_tokens_per_chunk=50)
        
        assert len(chunks) >= 1

    def test_auto_detect_json(self):
        """Test auto-detection of JSON content."""
        chunker = SmartChunker()
        json_str = '{"key": "value", "items": [1, 2, 3]}'
        
        chunks = chunker.chunk(json_str, max_tokens_per_chunk=50)
        
        assert len(chunks) >= 1

    def test_auto_detect_text(self):
        """Test auto-detection of plain text."""
        chunker = SmartChunker()
        text = "This is just plain text without any special formatting."
        
        chunks = chunker.chunk(text, max_tokens_per_chunk=50)
        
        assert len(chunks) >= 1


class TestGetChunker:
    """Tests for get_chunker factory function."""

    def test_get_html_chunker(self):
        """Test getting HTML chunker."""
        chunker = get_chunker(ContentType.HTML)
        assert isinstance(chunker, HTMLChunker)

    def test_get_json_chunker(self):
        """Test getting JSON chunker."""
        chunker = get_chunker(ContentType.JSON)
        assert isinstance(chunker, JSONChunker)

    def test_get_text_chunker(self):
        """Test getting text chunker."""
        chunker = get_chunker(ContentType.TEXT)
        assert isinstance(chunker, TextChunker)

    def test_get_smart_chunker_default(self):
        """Test getting smart chunker for unknown type."""
        chunker = get_chunker(None)
        assert isinstance(chunker, SmartChunker)


class TestConversationMessage:
    """Tests for ConversationMessage."""

    def test_system_message(self):
        """Test creating system message."""
        msg = ConversationMessage.system("You are helpful.")
        
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are helpful."
        assert msg.tokens > 0

    def test_user_message(self):
        """Test creating user message."""
        msg = ConversationMessage.user("Hello!")
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"

    def test_assistant_message(self):
        """Test creating assistant message."""
        msg = ConversationMessage.assistant("Hi there!")
        
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"

    def test_to_api_format(self):
        """Test converting to API format."""
        msg = ConversationMessage.user("Hello!")
        api_msg = msg.to_api_format()
        
        assert api_msg["role"] == "user"
        assert api_msg["content"] == "Hello!"


class TestConversationHistory:
    """Tests for ConversationHistory."""

    def test_add_messages(self):
        """Test adding messages to history."""
        history = ConversationHistory()
        
        history.add(ConversationMessage.system("Be helpful."))
        history.add(ConversationMessage.user("Hello!"))
        history.add(ConversationMessage.assistant("Hi!"))
        
        assert history.message_count == 2  # System not counted
        assert history.system_message is not None

    def test_total_tokens(self):
        """Test total token counting."""
        history = ConversationHistory()
        
        history.add(ConversationMessage.user("Hello!"))
        history.add(ConversationMessage.assistant("Hi!"))
        
        assert history.total_tokens > 0

    def test_get_messages_for_api(self):
        """Test getting messages in API format."""
        history = ConversationHistory()
        history.add(ConversationMessage.system("Be helpful."))
        history.add(ConversationMessage.user("Hello!"))
        
        messages = history.get_messages_for_api()
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_prune_to_fit(self):
        """Test pruning history to fit budget."""
        history = ConversationHistory()
        
        # Add many messages with enough content to exceed token budget
        for i in range(20):
            history.add(ConversationMessage.user(f"This is a longer message number {i} with more content to consume tokens"))
            history.add(ConversationMessage.assistant(f"This is a longer response number {i} with more content to consume tokens"))
        
        initial_count = history.message_count
        initial_tokens = history.total_tokens
        
        # Verify we have enough messages to prune
        assert initial_count == 40  # 20 * 2 messages
        assert initial_tokens > 50  # Should exceed our target budget
        
        # Prune to very small budget (50 tokens)
        removed = history.prune_to_fit(max_tokens=50, keep_recent=2)
        
        # Should have removed messages since initial tokens > 50 and we keep only 2
        assert removed > 0, f"Expected to remove messages, but removed={removed}, initial_tokens={initial_tokens}"
        assert history.message_count < initial_count
        assert history.message_count >= 2  # Keep recent

    def test_clear(self):
        """Test clearing history (keeps system)."""
        history = ConversationHistory()
        history.add(ConversationMessage.system("Be helpful."))
        history.add(ConversationMessage.user("Hello!"))
        
        history.clear()
        
        assert history.message_count == 0
        assert history.system_message is not None

    def test_reset(self):
        """Test full reset."""
        history = ConversationHistory()
        history.add(ConversationMessage.system("Be helpful."))
        history.add(ConversationMessage.user("Hello!"))
        
        history.reset()
        
        assert history.message_count == 0
        assert history.system_message is None


class TestAccumulationContext:
    """Tests for AccumulationContext."""

    def test_is_complete_single(self):
        """Test completion status for single phase."""
        ctx = AccumulationContext(phase=AccumulationPhase.SINGLE)
        assert ctx.is_complete

    def test_is_complete_accumulating(self):
        """Test completion status during accumulation."""
        ctx = AccumulationContext(
            phase=AccumulationPhase.ACCUMULATING,
            total_chunks=3,
            processed_chunks=2,
        )
        assert not ctx.is_complete

    def test_progress(self):
        """Test progress calculation."""
        ctx = AccumulationContext(
            phase=AccumulationPhase.ACCUMULATING,
            total_chunks=4,
            processed_chunks=2,
        )
        assert ctx.progress == 0.5


class TestConversationManager:
    """Tests for ConversationManager."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        mock = MagicMock()
        mock.get_model_info.return_value = MagicMock(
            context_window=128000,
            max_output_tokens=8192,
            capabilities=[],
        )
        mock.generate_structured = AsyncMock(return_value={"result": "test"})
        return mock

    def test_initialization(self, mock_llm):
        """Test ConversationManager initialization."""
        manager = ConversationManager(mock_llm)
        
        assert manager.llm == mock_llm
        assert manager.history is not None
        assert manager.budget is not None

    def test_set_system_prompt(self, mock_llm):
        """Test setting system prompt."""
        manager = ConversationManager(mock_llm)
        
        manager.set_system_prompt("You are a helpful assistant.")
        
        assert manager.history.system_message is not None
        assert manager.history.system_message.content == "You are a helpful assistant."

    def test_add_messages(self, mock_llm):
        """Test adding messages."""
        manager = ConversationManager(mock_llm)
        
        manager.add_user_message("Hello!")
        manager.add_assistant_message("Hi there!")
        
        assert manager.history.message_count == 2

    @pytest.mark.asyncio
    async def test_send_structured(self, mock_llm):
        """Test sending structured request."""
        manager = ConversationManager(mock_llm)
        manager.set_system_prompt("Be helpful.")
        
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        response = await manager.send_structured(
            content="What is 2+2?",
            schema=schema,
        )
        
        assert response == {"result": "test"}
        mock_llm.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_structured_adds_to_history(self, mock_llm):
        """Test that send_structured adds to history by default."""
        manager = ConversationManager(mock_llm)
        
        await manager.send_structured(
            content="Hello",
            schema={},
            add_to_history=True,
        )
        
        # User message + assistant response
        assert manager.history.message_count == 2

    @pytest.mark.asyncio
    async def test_send_structured_no_history(self, mock_llm):
        """Test send_structured without adding to history."""
        manager = ConversationManager(mock_llm)
        
        await manager.send_structured(
            content="Hello",
            schema={},
            add_to_history=False,
        )
        
        assert manager.history.message_count == 0

    def test_get_available_tokens(self, mock_llm):
        """Test available tokens calculation."""
        manager = ConversationManager(mock_llm)
        
        available = manager.get_available_tokens()
        
        assert available > 0

    def test_would_exceed_budget(self, mock_llm):
        """Test budget overflow detection."""
        manager = ConversationManager(mock_llm)
        
        # Short content should not exceed
        exceeds, overflow = manager.would_exceed_budget("Hello")
        assert not exceeds
        
        # Very long content might exceed
        long_content = "x" * 1000000
        exceeds, overflow = manager.would_exceed_budget(long_content)
        assert exceeds

    def test_reset(self, mock_llm):
        """Test conversation reset."""
        manager = ConversationManager(mock_llm)
        manager.set_system_prompt("Be helpful.")
        manager.add_user_message("Hello!")
        
        manager.reset()
        
        assert manager.history.message_count == 0
        # System prompt preserved
        assert manager.history.system_message is not None

    def test_full_reset(self, mock_llm):
        """Test full conversation reset."""
        manager = ConversationManager(mock_llm)
        manager.set_system_prompt("Be helpful.")
        manager.add_user_message("Hello!")
        
        manager.full_reset()
        
        assert manager.history.message_count == 0
        assert manager.history.system_message is None

    def test_get_stats(self, mock_llm):
        """Test statistics retrieval."""
        manager = ConversationManager(mock_llm)
        
        stats = manager.get_stats()
        
        assert "total_requests" in stats
        assert "history_tokens" in stats
        assert "available_tokens" in stats

    def test_format_messages_as_prompt(self, mock_llm):
        """Test message formatting."""
        manager = ConversationManager(mock_llm)
        
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        prompt = manager._format_messages_as_prompt(messages)
        
        assert "User: Hello!" in prompt
        assert "Assistant: Hi there!" in prompt

    @pytest.mark.asyncio
    async def test_send_with_large_content_fits(self, mock_llm):
        """Test send_with_large_content when content fits."""
        manager = ConversationManager(mock_llm)
        
        response = await manager.send_with_large_content(
            content="Short content that fits.",
            instruction="Summarize this.",
            schema={"type": "object"},
        )
        
        assert response == {"result": "test"}
        # Should be called once (single turn)
        assert mock_llm.generate_structured.call_count == 1

    @pytest.mark.asyncio
    async def test_send_with_large_content_chunks(self, mock_llm):
        """Test send_with_large_content when content needs chunking."""
        # Configure mock to have very small context window
        mock_llm.get_model_info.return_value = MagicMock(
            context_window=500,  # Very small
            max_output_tokens=100,
            capabilities=[],
        )
        
        manager = ConversationManager(mock_llm)
        
        # Large content that won't fit
        large_content = "Word. " * 500  # ~1000 tokens
        
        response = await manager.send_with_large_content(
            content=large_content,
            instruction="Summarize this.",
            schema={"type": "object"},
        )
        
        # Should have multiple calls (accumulation + synthesis)
        assert mock_llm.generate_structured.call_count > 1
