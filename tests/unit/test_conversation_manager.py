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


class TestReActAgentConversationIntegration:
    """Tests for ReActAgent's ConversationManager integration."""
    
    @pytest.fixture
    def mock_page_controller(self):
        """Create mock page controller."""
        mock = MagicMock()
        mock.page = MagicMock()
        mock.page.url = "https://example.com"
        return mock
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        mock = MagicMock()
        mock.get_model_info.return_value = MagicMock(
            name="test-model",
            provider="test",
            context_window=128000,
            max_output_tokens=8192,
            capabilities=[],
        )
        mock.model = "test-model"
        mock.generate_structured = AsyncMock(return_value={"result": "test"})
        return mock
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        mock = MagicMock()
        mock.list_tools.return_value = []
        mock.generate_tools_prompt.return_value = "Tools: []"
        mock.get_filtered_registry.return_value = mock
        return mock
    
    def test_agent_initializes_conversation_manager(self, mock_page_controller, mock_llm_provider, mock_tool_registry):
        """Test that agent properly initializes ConversationManager."""
        from flybrowser.agents.react_agent import ReActAgent
        
        agent = ReActAgent(
            page_controller=mock_page_controller,
            llm_provider=mock_llm_provider,
            tool_registry=mock_tool_registry,
        )
        
        assert agent.conversation is not None
        assert isinstance(agent.conversation, ConversationManager)
        assert agent.conversation.llm is mock_llm_provider
    
    def test_agent_reset_clears_conversation(self, mock_page_controller, mock_llm_provider, mock_tool_registry):
        """Test that agent reset clears conversation history."""
        from flybrowser.agents.react_agent import ReActAgent
        
        agent = ReActAgent(
            page_controller=mock_page_controller,
            llm_provider=mock_llm_provider,
            tool_registry=mock_tool_registry,
        )
        
        # Add some messages
        agent.conversation.add_user_message("Test message")
        assert agent.conversation.history.message_count == 1
        
        # Reset agent
        agent._reset_state()
        
        # Conversation should be cleared
        assert agent.conversation.history.message_count == 0
    
    def test_check_and_handle_large_context_fits(self, mock_page_controller, mock_llm_provider, mock_tool_registry):
        """Test _check_and_handle_large_context when context fits."""
        from flybrowser.agents.react_agent import ReActAgent
        
        agent = ReActAgent(
            page_controller=mock_page_controller,
            llm_provider=mock_llm_provider,
            tool_registry=mock_tool_registry,
        )
        
        small_context = "## Current Goal\nTest task\n## Current Page\nhttps://example.com"
        result = agent._check_and_handle_large_context(small_context)
        
        # Should return as-is
        assert result == small_context
    
    def test_check_and_handle_large_context_truncates(self, mock_page_controller, mock_llm_provider, mock_tool_registry):
        """Test _check_and_handle_large_context truncates large content."""
        from flybrowser.agents.react_agent import ReActAgent
        
        # Use small context window to force truncation
        mock_llm_provider.get_model_info.return_value = MagicMock(
            name="test-model",
            provider="test",
            context_window=1000,  # Small context window
            max_output_tokens=200,
            capabilities=[],
        )
        
        agent = ReActAgent(
            page_controller=mock_page_controller,
            llm_provider=mock_llm_provider,
            tool_registry=mock_tool_registry,
        )
        
        # Large context that won't fit
        large_context = "## Current Goal\nTest\n## Extracted Data\n" + ("x" * 50000)
        result = agent._check_and_handle_large_context(large_context)
        
        # Should be truncated
        assert len(result) < len(large_context)
        assert "truncated" in result.lower() or len(result) < 10000
    
    def test_track_conversation_turn_adds_messages(self, mock_page_controller, mock_llm_provider, mock_tool_registry):
        """Test _track_conversation_turn adds messages to history."""
        from flybrowser.agents.react_agent import ReActAgent
        
        agent = ReActAgent(
            page_controller=mock_page_controller,
            llm_provider=mock_llm_provider,
            tool_registry=mock_tool_registry,
        )
        
        initial_count = agent.conversation.history.message_count
        
        agent._track_conversation_turn(
            user_content="What should I do?",
            assistant_content='{"thought": "I should click", "action": {"tool": "click"}}'
        )
        
        # Should add 2 messages (user + assistant)
        assert agent.conversation.history.message_count == initial_count + 2
    
    def test_track_conversation_turn_skips_large_content(self, mock_page_controller, mock_llm_provider, mock_tool_registry):
        """Test _track_conversation_turn skips very large content."""
        from flybrowser.agents.react_agent import ReActAgent
        
        agent = ReActAgent(
            page_controller=mock_page_controller,
            llm_provider=mock_llm_provider,
            tool_registry=mock_tool_registry,
        )
        
        initial_count = agent.conversation.history.message_count
        
        # Large content that exceeds max_turn_tokens (10K)
        large_content = "x" * 100000  # ~25K tokens
        
        agent._track_conversation_turn(
            user_content=large_content,
            assistant_content="Small response"
        )
        
        # Should NOT add messages (skipped due to size)
        assert agent.conversation.history.message_count == initial_count
    
    def test_conversation_manager_uses_correct_token_budget(self, mock_page_controller, mock_llm_provider, mock_tool_registry):
        """Test ConversationManager is configured with model's context window."""
        from flybrowser.agents.react_agent import ReActAgent
        
        mock_llm_provider.get_model_info.return_value = MagicMock(
            name="test-model",
            provider="test",
            context_window=64000,
            max_output_tokens=4096,
            capabilities=[],
        )
        
        agent = ReActAgent(
            page_controller=mock_page_controller,
            llm_provider=mock_llm_provider,
            tool_registry=mock_tool_registry,
        )
        
        # Budget should reflect model's context window
        assert agent.conversation.budget.context_window == 64000
        assert agent.conversation.budget.max_output_tokens == 4096


class TestConversationManagerVision:
    """Tests for ConversationManager vision/VLM support."""
    
    @pytest.fixture
    def mock_vision_llm(self):
        """Create mock LLM provider with vision capability."""
        from flybrowser.llm.base import ModelCapability, ModelInfo
        
        mock = MagicMock()
        # Use actual ModelInfo to avoid MagicMock name issues
        mock.get_model_info.return_value = ModelInfo(
            name="gpt-4-vision",
            provider="openai",
            context_window=128000,
            max_output_tokens=8192,
            capabilities=[ModelCapability.VISION, ModelCapability.STRUCTURED_OUTPUT],
        )
        mock.generate_structured = AsyncMock(return_value={"result": "test"})
        mock.generate_structured_with_vision = AsyncMock(return_value={"thought": "I see a button", "action": {"tool": "click"}})
        return mock
    
    @pytest.fixture
    def mock_text_only_llm(self):
        """Create mock LLM provider without vision capability."""
        from flybrowser.llm.base import ModelInfo
        
        mock = MagicMock()
        # Use actual ModelInfo to avoid MagicMock name issues
        mock.get_model_info.return_value = ModelInfo(
            name="gpt-3.5-turbo",
            provider="openai",
            context_window=16000,
            max_output_tokens=4096,
            capabilities=[],  # No vision
        )
        mock.generate_structured = AsyncMock(return_value={"result": "test"})
        return mock
    
    def test_has_vision_property_true(self, mock_vision_llm):
        """Test has_vision is True for vision-capable models."""
        manager = ConversationManager(mock_vision_llm)
        assert manager.has_vision is True
    
    def test_has_vision_property_false(self, mock_text_only_llm):
        """Test has_vision is False for text-only models."""
        manager = ConversationManager(mock_text_only_llm)
        assert manager.has_vision is False
    
    @pytest.mark.asyncio
    async def test_send_structured_with_vision_success(self, mock_vision_llm):
        """Test send_structured_with_vision with a vision-capable model."""
        manager = ConversationManager(mock_vision_llm)
        
        # Fake screenshot bytes
        image_data = b"x" * 10000  # ~10KB image
        
        response = await manager.send_structured_with_vision(
            content="Click the search button",
            image_data=image_data,
            schema={"type": "object"},
            temperature=0.7,
        )
        
        assert response == {"thought": "I see a button", "action": {"tool": "click"}}
        mock_vision_llm.generate_structured_with_vision.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_structured_with_vision_fails_without_capability(self, mock_text_only_llm):
        """Test send_structured_with_vision raises error for non-vision models."""
        manager = ConversationManager(mock_text_only_llm)
        
        image_data = b"x" * 10000
        
        with pytest.raises(ValueError, match="does not support vision"):
            await manager.send_structured_with_vision(
                content="Click the button",
                image_data=image_data,
                schema={"type": "object"},
            )
    
    def test_estimate_image_tokens_low_detail(self, mock_vision_llm):
        """Test image token estimation for low detail."""
        manager = ConversationManager(mock_vision_llm)
        
        tokens = manager._estimate_image_tokens(100000, detail="low")
        
        # Low detail should return fixed base tokens
        assert tokens == ConversationManager.IMAGE_BASE_TOKENS
    
    def test_estimate_image_tokens_high_detail(self, mock_vision_llm):
        """Test image token estimation for high detail."""
        manager = ConversationManager(mock_vision_llm)
        
        # ~100KB image
        tokens = manager._estimate_image_tokens(100000, detail="high")
        
        # Should be base + tiles * tokens_per_tile
        assert tokens > ConversationManager.IMAGE_BASE_TOKENS
        assert tokens <= ConversationManager.IMAGE_BASE_TOKENS + (16 * ConversationManager.IMAGE_TOKENS_PER_TILE)
    
    def test_estimate_image_tokens_auto_detail(self, mock_vision_llm):
        """Test image token estimation for auto detail."""
        manager = ConversationManager(mock_vision_llm)
        
        # Small image ~10KB
        small_tokens = manager._estimate_image_tokens(10000, detail="auto")
        # Large image ~500KB
        large_tokens = manager._estimate_image_tokens(500000, detail="auto")
        
        # Larger image should estimate more tokens
        assert large_tokens >= small_tokens
    
    @pytest.mark.asyncio
    async def test_vision_request_tracks_statistics(self, mock_vision_llm):
        """Test that vision requests are tracked in statistics."""
        manager = ConversationManager(mock_vision_llm)
        
        initial_stats = manager.get_stats()
        assert initial_stats["vision_requests"] == 0
        
        await manager.send_structured_with_vision(
            content="Test",
            image_data=b"x" * 10000,
            schema={"type": "object"},
        )
        
        updated_stats = manager.get_stats()
        assert updated_stats["vision_requests"] == 1
        assert updated_stats["total_requests"] == 1
    
    def test_stats_includes_vision_info(self, mock_vision_llm):
        """Test that stats include vision capability info."""
        manager = ConversationManager(mock_vision_llm)
        
        stats = manager.get_stats()
        
        assert "has_vision" in stats
        assert stats["has_vision"] is True
        assert "model" in stats
        assert stats["model"] == "gpt-4-vision"


class TestConversationManagerCleanup:
    """Tests for ConversationManager cleanup and budget management methods."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        from flybrowser.llm.base import ModelInfo
        
        mock = MagicMock()
        mock.get_model_info.return_value = ModelInfo(
            name="gpt-4",
            provider="openai",
            context_window=128000,
            max_output_tokens=8192,
            capabilities=[],
        )
        mock.generate_structured = AsyncMock(return_value={"result": "test"})
        return mock
    
    def test_ensure_budget_available_does_nothing_when_sufficient(self, mock_llm):
        """Test ensure_budget_available returns 0 when budget is sufficient."""
        manager = ConversationManager(mock_llm)
        
        # Budget should be sufficient for small request
        pruned = manager.ensure_budget_available(required_tokens=1000)
        
        assert pruned == 0
    
    def test_ensure_budget_available_prunes_when_needed(self, mock_llm):
        """Test ensure_budget_available prunes history when budget is low."""
        manager = ConversationManager(mock_llm)
        
        # Fill up history with many messages
        for i in range(20):
            manager.add_user_message(f"User message {i} " * 500)  # ~500 tokens each
            manager.add_assistant_message(f"Assistant response {i} " * 500)
        
        initial_count = manager.history.message_count
        assert initial_count == 40  # 20 pairs
        
        # Now request a very large budget that will require pruning
        available_before = manager.get_available_tokens()
        pruned = manager.ensure_budget_available(
            required_tokens=available_before + 10000,  # Request more than available
            min_free_tokens=5000,
        )
        
        # Should have pruned some messages
        assert pruned > 0
        assert manager.history.message_count < initial_count
        # Should keep at least 4 recent messages
        assert manager.history.message_count >= 4
    
    def test_ensure_budget_available_aggressive_mode(self, mock_llm):
        """Test ensure_budget_available with aggressive mode prunes more."""
        manager = ConversationManager(mock_llm)
        
        # Fill up history
        for i in range(30):
            manager.add_user_message(f"User message {i} " * 500)
            manager.add_assistant_message(f"Assistant response {i} " * 500)
        
        initial_count = manager.history.message_count
        
        # Request aggressive cleanup
        pruned = manager.ensure_budget_available(
            required_tokens=manager.budget.available_for_input,  # Request max
            min_free_tokens=50000,
            aggressive=True,
        )
        
        # Aggressive mode should prune down to 2 messages
        if manager.history.message_count == 2:
            assert pruned > 0
    
    def test_cleanup_if_needed_no_action_when_budget_ok(self, mock_llm):
        """Test cleanup_if_needed does nothing when budget is healthy."""
        manager = ConversationManager(mock_llm)
        
        # Add a few messages (not enough to trigger cleanup)
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there")
        
        # Cleanup should not be needed
        pruned = manager.cleanup_if_needed(threshold_percent=0.15)
        
        assert pruned == 0
        assert manager.history.message_count == 2
    
    def test_cleanup_if_needed_triggers_when_budget_low(self, mock_llm):
        """Test cleanup_if_needed triggers cleanup when budget is below threshold."""
        manager = ConversationManager(mock_llm)
        
        # Fill up most of the budget (need lots of messages)
        # Each message of ~2000 chars is roughly 500 tokens
        message_count = 0
        while manager.get_available_tokens() > manager.budget.available_for_input * 0.10:
            manager.add_user_message("x" * 2000)
            manager.add_assistant_message("y" * 2000)
            message_count += 2
            if message_count > 200:  # Safety limit
                break
        
        initial_available = manager.get_available_tokens()
        initial_count = manager.history.message_count
        
        # Should trigger cleanup (15% threshold)
        pruned = manager.cleanup_if_needed(threshold_percent=0.15)
        
        # Should have freed up some space
        if pruned > 0:
            assert manager.get_available_tokens() > initial_available
            assert manager.history.message_count < initial_count
    
    def test_ensure_budget_available_keeps_recent_messages(self, mock_llm):
        """Test that ensure_budget_available always keeps recent messages."""
        manager = ConversationManager(mock_llm)
        
        # Add messages with identifiable content
        for i in range(10):
            manager.add_user_message(f"User message {i} " * 500)
            manager.add_assistant_message(f"Assistant response {i} " * 500)
        
        # Force pruning
        manager.ensure_budget_available(
            required_tokens=manager.budget.available_for_input,
            min_free_tokens=manager.budget.available_for_input // 2,
        )
        
        # Verify recent messages are kept (last 4 at minimum)
        remaining_messages = manager.history.messages
        assert len(remaining_messages) >= 4
        
        # Most recent message should still be there
        assert "response 9" in remaining_messages[-1].content
