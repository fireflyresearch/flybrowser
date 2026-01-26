# Copyright 2026 Firefly Software Solutions Inc.
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
Unit tests for the context system.

Tests ContextBuilder, ContextValidator, ActionContext, and FileUploadSpec.
"""

import os
import tempfile
import pytest
from flybrowser.agents.context import (
    ActionContext,
    ContextBuilder,
    ContextValidator,
    ContextType,
    FileUploadSpec,
    create_form_context,
    create_upload_context,
    create_filter_context,
)


class TestActionContext:
    """Test ActionContext data class."""

    def test_empty_context(self):
        """Test creating an empty ActionContext."""
        context = ActionContext()
        assert context.is_empty()
        assert not context.has_type(ContextType.FORM_DATA)
        assert not context.has_type(ContextType.FILES)

    def test_context_with_form_data(self):
        """Test ActionContext with form data."""
        context = ActionContext(form_data={"email": "test@example.com"})
        assert not context.is_empty()
        assert context.has_type(ContextType.FORM_DATA)
        assert not context.has_type(ContextType.FILES)

    def test_context_to_dict(self):
        """Test converting ActionContext to dict."""
        context = ActionContext(
            form_data={"email": "test@example.com"},
            filters={"price_max": 100},
        )
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict["form_data"] == {"email": "test@example.com"}
        assert context_dict["filters"] == {"price_max": 100}
        # to_dict() only includes non-empty fields
        assert "files" not in context_dict

    def test_context_from_dict(self):
        """Test creating ActionContext from dict."""
        data = {
            "form_data": {"email": "test@example.com"},
            "filters": {"price_max": 100},
        }
        context = ActionContext.from_dict(data)
        
        assert context.form_data == {"email": "test@example.com"}
        assert context.filters == {"price_max": 100}
        assert context.files == []

    def test_context_has_type(self):
        """Test checking if context has specific types."""
        context = ActionContext(
            form_data={"email": "test@example.com"},
            filters={"price_max": 100},
        )
        
        assert context.has_type(ContextType.FORM_DATA)
        assert context.has_type(ContextType.FILTERS)
        assert not context.has_type(ContextType.FILES)
        assert not context.has_type(ContextType.PREFERENCES)


class TestFileUploadSpec:
    """Test FileUploadSpec data class."""

    def test_file_upload_spec_creation(self):
        """Test creating a FileUploadSpec."""
        spec = FileUploadSpec(
            field="resume",
            path="/path/to/file.pdf",
            mime_type="application/pdf",
        )
        assert spec.field == "resume"
        assert spec.path == "/path/to/file.pdf"
        assert spec.mime_type == "application/pdf"

    def test_file_upload_spec_to_dict(self):
        """Test converting FileUploadSpec to dict."""
        spec = FileUploadSpec(
            field="resume",
            path="/path/to/file.pdf",
            mime_type="application/pdf",
        )
        spec_dict = spec.to_dict()
        
        assert spec_dict["field"] == "resume"
        assert spec_dict["path"] == "/path/to/file.pdf"
        assert spec_dict["mime_type"] == "application/pdf"

    def test_file_upload_spec_validation_success(self):
        """Test FileUploadSpec validation with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            spec = FileUploadSpec(field="file", path=temp_path, verify_exists=True)
            is_valid, error = spec.validate()
            assert is_valid
            assert error is None
        finally:
            os.unlink(temp_path)

    def test_file_upload_spec_validation_failure(self):
        """Test FileUploadSpec validation with non-existent file."""
        spec = FileUploadSpec(
            field="file",
            path="/nonexistent/file.pdf",
            verify_exists=True,
        )
        is_valid, error = spec.validate()
        assert not is_valid
        assert "File not found" in error


class TestContextBuilder:
    """Test ContextBuilder fluent API."""

    def test_empty_builder(self):
        """Test building empty context."""
        context = ContextBuilder().build()
        assert context.is_empty()

    def test_with_form_data(self):
        """Test building context with form data."""
        context = ContextBuilder()\
            .with_form_data({"email": "test@example.com", "password": "secret"})\
            .build()
        
        assert context.form_data == {"email": "test@example.com", "password": "secret"}

    def test_with_form_field(self):
        """Test adding individual form fields."""
        context = ContextBuilder()\
            .with_form_field("email", "test@example.com")\
            .with_form_field("password", "secret")\
            .build()
        
        assert context.form_data["email"] == "test@example.com"
        assert context.form_data["password"] == "secret"

    def test_with_file(self):
        """Test adding a file upload."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            context = ContextBuilder()\
                .with_file("resume", temp_path, "application/pdf")\
                .build()
            
            assert len(context.files) == 1
            assert context.files[0].field == "resume"
            assert context.files[0].path == temp_path
            assert context.files[0].mime_type == "application/pdf"
        finally:
            os.unlink(temp_path)

    def test_with_multiple_files(self):
        """Test adding multiple files."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f1:
            temp_path1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
            temp_path2 = f2.name
        
        try:
            context = ContextBuilder()\
                .with_file("resume", temp_path1, "application/pdf")\
                .with_file("photo", temp_path2, "image/jpeg")\
                .build()
            
            assert len(context.files) == 2
            assert context.files[0].field == "resume"
            assert context.files[1].field == "photo"
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)

    def test_with_filters(self):
        """Test adding filters."""
        context = ContextBuilder()\
            .with_filters({"price_max": 100, "category": "electronics"})\
            .build()
        
        assert context.filters == {"price_max": 100, "category": "electronics"}

    def test_with_filter(self):
        """Test adding individual filters."""
        context = ContextBuilder()\
            .with_filter("price_max", 100)\
            .with_filter("category", "electronics")\
            .build()
        
        assert context.filters["price_max"] == 100
        assert context.filters["category"] == "electronics"

    def test_with_preferences(self):
        """Test adding preferences."""
        context = ContextBuilder()\
            .with_preferences({"sort_by": "price", "limit": 10})\
            .build()
        
        assert context.preferences == {"sort_by": "price", "limit": 10}

    def test_with_preference(self):
        """Test adding individual preferences."""
        context = ContextBuilder()\
            .with_preference("sort_by", "price")\
            .with_preference("limit", 10)\
            .build()
        
        assert context.preferences["sort_by"] == "price"
        assert context.preferences["limit"] == 10

    def test_with_conditions(self):
        """Test adding conditions."""
        context = ContextBuilder()\
            .with_conditions({"requires_login": False})\
            .build()
        
        assert context.conditions == {"requires_login": False}

    def test_with_condition(self):
        """Test adding individual conditions."""
        context = ContextBuilder()\
            .with_condition("requires_login", False)\
            .with_condition("max_redirects", 3)\
            .build()
        
        assert context.conditions["requires_login"] is False
        assert context.conditions["max_redirects"] == 3

    def test_with_constraints(self):
        """Test adding constraints."""
        context = ContextBuilder()\
            .with_constraints({"timeout_seconds": 30})\
            .build()
        
        assert context.constraints == {"timeout_seconds": 30}

    def test_with_constraint(self):
        """Test adding individual constraints."""
        context = ContextBuilder()\
            .with_constraint("timeout_seconds", 30)\
            .with_constraint("max_retries", 3)\
            .build()
        
        assert context.constraints["timeout_seconds"] == 30
        assert context.constraints["max_retries"] == 3

    def test_with_metadata(self):
        """Test adding metadata."""
        context = ContextBuilder()\
            .with_metadata({"request_id": "abc123"})\
            .build()
        
        assert context.metadata == {"request_id": "abc123"}

    def test_chaining_multiple_types(self):
        """Test chaining multiple context types."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            context = ContextBuilder()\
                .with_form_data({"email": "test@example.com"})\
                .with_file("resume", temp_path)\
                .with_filters({"price_max": 100})\
                .with_preferences({"sort_by": "price"})\
                .with_conditions({"requires_login": False})\
                .with_constraints({"timeout_seconds": 30})\
                .with_metadata({"request_id": "abc123"})\
                .build()
            
            assert not context.is_empty()
            assert context.has_type(ContextType.FORM_DATA)
            assert context.has_type(ContextType.FILES)
            assert context.has_type(ContextType.FILTERS)
            assert context.has_type(ContextType.PREFERENCES)
            assert context.has_type(ContextType.CONDITIONS)
            assert context.has_type(ContextType.CONSTRAINTS)
            assert context.has_type(ContextType.METADATA)
        finally:
            os.unlink(temp_path)

    def test_build_without_validation(self):
        """Test building without validation."""
        context = ContextBuilder()\
            .with_file("file", "/nonexistent/file.pdf", verify_exists=False)\
            .build(validate=False)
        
        assert len(context.files) == 1

    def test_build_with_validation_failure(self):
        """Test building with validation failure."""
        with pytest.raises(ValueError):
            ContextBuilder()\
                .with_file("file", "/nonexistent/file.pdf")\
                .build(validate=True)


class TestContextValidator:
    """Test ContextValidator."""

    def test_validate_empty_context(self):
        """Test validating empty context."""
        context = ActionContext()
        is_valid, errors = ContextValidator.validate(context)
        assert is_valid
        assert len(errors) == 0

    def test_validate_valid_context(self):
        """Test validating valid context."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            context = ActionContext(
                form_data={"email": "test@example.com"},
                files=[FileUploadSpec("resume", temp_path)],
                filters={"price_max": 100},
            )
            is_valid, errors = ContextValidator.validate(context)
            assert is_valid
            assert len(errors) == 0
        finally:
            os.unlink(temp_path)

    def test_validate_invalid_form_data(self):
        """Test validation with invalid form data type."""
        context = ActionContext(form_data="invalid")  # Should be dict
        is_valid, errors = ContextValidator.validate(context)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_file_not_found(self):
        """Test validation with non-existent file."""
        context = ActionContext(
            files=[FileUploadSpec("resume", "/nonexistent/file.pdf")]
        )
        is_valid, errors = ContextValidator.validate(context)
        assert not is_valid
        assert any("File not found" in error for error in errors)

    def test_validate_for_tool(self):
        """Test validating context for specific tool."""
        context = ActionContext(form_data={"email": "test@example.com"})
        
        # Should pass for tool expecting form_data
        is_valid, errors = ContextValidator.validate_for_tool(
            context,
            [ContextType.FORM_DATA],
        )
        assert is_valid
        
        # Should fail for tool expecting files
        is_valid, errors = ContextValidator.validate_for_tool(
            context,
            [ContextType.FILES],
        )
        assert not is_valid


class TestConvenienceFunctions:
    """Test convenience functions for creating contexts."""

    def test_create_form_context(self):
        """Test create_form_context convenience function."""
        context = create_form_context({"email": "test@example.com"})
        assert context.form_data == {"email": "test@example.com"}
        assert context.is_empty() is False

    def test_create_upload_context(self):
        """Test create_upload_context convenience function."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            context = create_upload_context([
                {"field": "resume", "path": temp_path}
            ])
            assert len(context.files) == 1
            assert context.files[0].field == "resume"
        finally:
            os.unlink(temp_path)

    def test_create_filter_context(self):
        """Test create_filter_context convenience function."""
        context = create_filter_context(
            filters={"price_max": 100},
            preferences={"sort_by": "price"},
        )
        assert context.filters == {"price_max": 100}
        assert context.preferences == {"sort_by": "price"}


class TestContextConversion:
    """Test context conversion between ActionContext and dict."""

    def test_dict_to_context_to_dict(self):
        """Test round-trip conversion."""
        original_dict = {
            "form_data": {"email": "test@example.com"},
            "filters": {"price_max": 100},
            "preferences": {"sort_by": "price"},
        }
        
        context = ActionContext.from_dict(original_dict)
        converted_dict = context.to_dict()
        
        assert converted_dict["form_data"] == original_dict["form_data"]
        assert converted_dict["filters"] == original_dict["filters"]
        assert converted_dict["preferences"] == original_dict["preferences"]

    def test_context_to_dict_to_context(self):
        """Test reverse round-trip conversion."""
        original_context = ContextBuilder()\
            .with_form_data({"email": "test@example.com"})\
            .with_filters({"price_max": 100})\
            .build()
        
        context_dict = original_context.to_dict()
        converted_context = ActionContext.from_dict(context_dict)
        
        assert converted_context.form_data == original_context.form_data
        assert converted_context.filters == original_context.filters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
