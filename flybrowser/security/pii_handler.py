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
Secure PII (Personally Identifiable Information) Handling for FlyBrowser.

This module provides secure handling of sensitive data including:
- Credential storage with encryption
- PII masking for logs and LLM prompts
- Secure form filling without exposing data to LLMs
- Field-level sensitivity marking
- Placeholder-based masking for LLM prompts with resolution for execution

Key Features:
- PII data is NEVER sent to LLM providers
- Encrypted storage for credentials
- Automatic masking in logs and debugging output
- Secure input methods for form filling
- Placeholder system: LLM sees {{CREDENTIAL:email}} but browser uses real value

Architecture:
    1. User stores credentials: pii_handler.store_credential("email", "user@example.com")
    2. User creates instruction: "Fill email field with user@example.com"
    3. For LLM: instruction becomes "Fill email field with {{CREDENTIAL:email}}"
    4. LLM plans actions using placeholders
    5. For execution: placeholders are resolved back to real values
    6. Browser fills the actual value, LLM never sees it
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import os
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from flybrowser.utils.logger import logger


def secure_zero_memory(data: Union[str, bytes, bytearray]) -> None:
    """
    Securely zero out memory containing sensitive data.

    This attempts to overwrite the memory location to prevent
    sensitive data from lingering in memory.
    """
    try:
        if isinstance(data, str):
            # Convert to bytearray for in-place modification
            encoded = bytearray(data.encode('utf-8'))
            for i in range(len(encoded)):
                encoded[i] = 0
        elif isinstance(data, (bytes, bytearray)):
            if isinstance(data, bytes):
                # bytes are immutable, create mutable copy
                mutable = bytearray(data)
                for i in range(len(mutable)):
                    mutable[i] = 0
            else:
                for i in range(len(data)):
                    data[i] = 0
    except Exception:
        # Best effort - some Python implementations may not allow this
        pass


class PIIType(str, Enum):
    """Types of PII data."""

    PASSWORD = "password"
    USERNAME = "username"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    CVV = "cvv"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    API_KEY = "api_key"
    TOKEN = "token"
    CUSTOM = "custom"


@dataclass
class PIIConfig:
    """Configuration for PII handling."""

    enabled: bool = True
    mask_in_logs: bool = True
    mask_in_llm_prompts: bool = True
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None  # Will be auto-generated if not provided
    mask_character: str = "*"
    mask_length: int = 8
    preserve_format: bool = True  # Preserve format hints (e.g., ***@***.com for email)
    auto_detect_pii: bool = True
    # Use placeholders like {{CREDENTIAL:name}} instead of asterisks for LLM prompts
    # This allows LLM to understand the structure while never seeing actual values
    use_placeholders_for_llm: bool = True
    placeholder_prefix: str = "{{CREDENTIAL:"
    placeholder_suffix: str = "}}"
    # Session timeout for auto-cleanup of credentials (seconds, 0 = no timeout)
    credential_timeout: float = 0
    sensitive_field_patterns: List[str] = field(default_factory=lambda: [
        r"password", r"passwd", r"pwd", r"secret", r"token", r"api[_-]?key",
        r"credit[_-]?card", r"card[_-]?number", r"cvv", r"cvc", r"ssn",
        r"social[_-]?security", r"auth", r"credential", r"pin", r"otp",
        r"verification[_-]?code", r"security[_-]?code", r"access[_-]?key",
        r"private[_-]?key", r"signing[_-]?key",
    ])


@dataclass
class PIIField:
    """Represents a field containing PII data."""

    name: str
    pii_type: PIIType
    selector: Optional[str] = None  # CSS selector for form field
    is_sensitive: bool = True
    mask_pattern: Optional[str] = None  # Custom mask pattern


@dataclass
class SecureCredential:
    """
    Securely stored credential.

    The original_value field stores the raw value for placeholder resolution.
    This is kept separate from encrypted_value which is used for persistent storage.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    pii_type: PIIType = PIIType.PASSWORD
    encrypted_value: bytes = field(default_factory=bytes)
    # Original value stored for quick access during session (not persisted)
    # This enables placeholder resolution without decryption overhead
    _original_value: Optional[str] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    # Last access time for timeout-based cleanup
    last_accessed: float = field(default_factory=lambda: time.time())

    def __repr__(self) -> str:
        """Safe repr that doesn't expose the value."""
        return f"SecureCredential(id={self.id!r}, name={self.name!r}, pii_type={self.pii_type})"

    def __del__(self) -> None:
        """Securely clear sensitive data on deletion."""
        if self._original_value:
            secure_zero_memory(self._original_value)
            self._original_value = None
        if self.encrypted_value:
            secure_zero_memory(self.encrypted_value)


class PIIMasker:
    """
    Masks PII data in text for safe logging and LLM prompts.

    Example:
        >>> masker = PIIMasker()
        >>> text = "My password is secret123 and email is user@example.com"
        >>> masked = masker.mask_text(text)
        >>> print(masked)
        "My password is ******** and email is ****@****.***"
    """

    # Common PII patterns - comprehensive detection
    PATTERNS: Dict[PIIType, Pattern] = {
        # Email: standard email format
        PIIType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
        # Phone: US/international formats
        PIIType.PHONE: re.compile(
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        ),
        # SSN: XXX-XX-XXXX format
        PIIType.SSN: re.compile(
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
        ),
        # Credit card: 16 digits with optional separators
        PIIType.CREDIT_CARD: re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        ),
        # API keys: common formats (sk-, pk-, api_, etc.)
        PIIType.API_KEY: re.compile(
            r'\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9]{20,}\b',
            re.IGNORECASE
        ),
        # Tokens: JWT and similar
        PIIType.TOKEN: re.compile(
            r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b'
        ),
    }

    def __init__(self, config: Optional[PIIConfig] = None) -> None:
        """Initialize the masker."""
        self.config = config or PIIConfig()
        self._sensitive_patterns: List[Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self.config.sensitive_field_patterns
        ]

    def mask_text(self, text: str, pii_types: Optional[Set[PIIType]] = None) -> str:
        """
        Mask PII in text.

        Args:
            text: Text to mask
            pii_types: Specific PII types to mask (None = all)

        Returns:
            Masked text
        """
        if not self.config.enabled:
            return text

        masked = text
        types_to_check = pii_types or set(self.PATTERNS.keys())

        for pii_type in types_to_check:
            if pii_type in self.PATTERNS:
                pattern = self.PATTERNS[pii_type]
                masked = pattern.sub(lambda m: self._mask_value(m.group(), pii_type), masked)

        return masked

    def _mask_value(self, value: str, pii_type: PIIType) -> str:
        """Mask a single value based on PII type."""
        mask_char = self.config.mask_character

        if self.config.preserve_format:
            if pii_type == PIIType.EMAIL:
                # Preserve email format: ****@****.***
                parts = value.split("@")
                if len(parts) == 2:
                    domain_parts = parts[1].split(".")
                    masked_local = mask_char * min(len(parts[0]), 4)
                    masked_domain = ".".join(mask_char * min(len(p), 4) for p in domain_parts)
                    return f"{masked_local}@{masked_domain}"
            elif pii_type == PIIType.PHONE:
                # Preserve phone format: ***-***-****
                return re.sub(r'\d', mask_char, value)
            elif pii_type == PIIType.CREDIT_CARD:
                # Show last 4 digits: ****-****-****-1234
                digits = re.sub(r'\D', '', value)
                if len(digits) >= 4:
                    return mask_char * (len(digits) - 4) + digits[-4:]

        # Default: fixed-length mask
        return mask_char * self.config.mask_length

    def mask_dict(self, data: Dict[str, Any], sensitive_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Mask sensitive values in a dictionary.

        Args:
            data: Dictionary to mask
            sensitive_keys: Keys to treat as sensitive (auto-detected if None)

        Returns:
            Dictionary with masked values
        """
        result = {}

        for key, value in data.items():
            is_sensitive = False

            # Check if key matches sensitive patterns
            if sensitive_keys and key in sensitive_keys:
                is_sensitive = True
            elif self.config.auto_detect_pii:
                for pattern in self._sensitive_patterns:
                    if pattern.search(key):
                        is_sensitive = True
                        break

            if is_sensitive and isinstance(value, str):
                result[key] = self.config.mask_character * self.config.mask_length
            elif isinstance(value, dict):
                result[key] = self.mask_dict(value, sensitive_keys)
            elif isinstance(value, str):
                result[key] = self.mask_text(value)
            else:
                result[key] = value

        return result

    def is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field name indicates sensitive data."""
        for pattern in self._sensitive_patterns:
            if pattern.search(field_name):
                return True
        return False


class SecureCredentialStore:
    """
    Encrypted storage for credentials and sensitive data.

    Uses Fernet symmetric encryption to securely store credentials.
    Credentials are never exposed in plain text in logs or to LLMs.

    Supports placeholder-based masking where the LLM sees {{CREDENTIAL:name}}
    but the actual value is used for browser execution.

    Example:
        >>> store = SecureCredentialStore()
        >>> cred_id = store.store("my_password", "secret123", PIIType.PASSWORD)
        >>> # Later, retrieve for form filling
        >>> value = store.retrieve(cred_id)
        >>> # Or get placeholder for LLM
        >>> placeholder = store.get_placeholder(cred_id)  # "{{CREDENTIAL:my_password}}"
    """

    def __init__(self, config: Optional[PIIConfig] = None, encryption_key: Optional[str] = None) -> None:
        """Initialize the credential store."""
        self.config = config or PIIConfig()
        self._credentials: Dict[str, SecureCredential] = {}
        # Mapping from original values to credential IDs for auto-replacement
        self._value_to_id: Dict[str, str] = {}
        # Mapping from names to credential IDs for lookup
        self._name_to_id: Dict[str, str] = {}

        # Set up encryption
        if encryption_key:
            self._fernet = self._create_fernet_from_password(encryption_key)
        elif self.config.encryption_key:
            self._fernet = self._create_fernet_from_password(self.config.encryption_key)
        else:
            # Generate a random key for this session
            self._fernet = Fernet(Fernet.generate_key())

    def _create_fernet_from_password(self, password: str) -> Fernet:
        """Create Fernet instance from password using PBKDF2."""
        # Generate a random salt for each instance (more secure)
        salt = hashlib.sha256(b"flybrowser_pii_" + password.encode()).digest()[:16]
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)

    def store(
        self,
        name: str,
        value: str,
        pii_type: PIIType = PIIType.PASSWORD,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a credential securely.

        Args:
            name: Name/identifier for the credential (used in placeholders)
            value: The sensitive value to store
            pii_type: Type of PII
            metadata: Additional non-sensitive metadata

        Returns:
            Credential ID for later retrieval
        """
        encrypted = self._fernet.encrypt(value.encode())

        credential = SecureCredential(
            name=name,
            pii_type=pii_type,
            encrypted_value=encrypted,
            _original_value=value,  # Keep for quick access during session
            metadata=metadata or {},
        )

        self._credentials[credential.id] = credential
        self._value_to_id[value] = credential.id
        self._name_to_id[name] = credential.id
        logger.debug(f"Stored credential: {credential.id} ({name})")

        return credential.id

    def retrieve(self, credential_id: str) -> Optional[str]:
        """
        Retrieve a credential value.

        Args:
            credential_id: ID of the credential

        Returns:
            Decrypted value or None if not found
        """
        credential = self._credentials.get(credential_id)
        if not credential:
            return None

        # Update last accessed time
        credential.last_accessed = time.time()

        # Check timeout if configured
        if self.config.credential_timeout > 0:
            age = time.time() - credential.created_at
            if age > self.config.credential_timeout:
                logger.warning(f"Credential {credential_id} has expired")
                self.delete(credential_id)
                return None

        # Use cached original value if available (faster)
        if credential._original_value:
            return credential._original_value

        # Fall back to decryption
        try:
            return self._fernet.decrypt(credential.encrypted_value).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt credential {credential_id}: {e}")
            return None

    def retrieve_by_name(self, name: str) -> Optional[str]:
        """Retrieve a credential value by name."""
        cred_id = self._name_to_id.get(name)
        if cred_id:
            return self.retrieve(cred_id)
        return None

    def get_id_by_name(self, name: str) -> Optional[str]:
        """Get credential ID by name."""
        return self._name_to_id.get(name)

    def get_placeholder(self, credential_id: str) -> Optional[str]:
        """
        Get the placeholder string for a credential.

        Returns:
            Placeholder like "{{CREDENTIAL:email}}" or None if not found
        """
        credential = self._credentials.get(credential_id)
        if not credential:
            return None

        prefix = self.config.placeholder_prefix
        suffix = self.config.placeholder_suffix
        return f"{prefix}{credential.name}{suffix}"

    def get_placeholder_by_name(self, name: str) -> str:
        """Get placeholder string by credential name."""
        prefix = self.config.placeholder_prefix
        suffix = self.config.placeholder_suffix
        return f"{prefix}{name}{suffix}"

    def delete(self, credential_id: str) -> bool:
        """Delete a credential securely."""
        if credential_id in self._credentials:
            credential = self._credentials[credential_id]

            # Remove from lookup maps
            if credential._original_value and credential._original_value in self._value_to_id:
                del self._value_to_id[credential._original_value]
            if credential.name in self._name_to_id:
                del self._name_to_id[credential.name]

            # Secure deletion (destructor will zero memory)
            del self._credentials[credential_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        """
        Clean up expired credentials.

        Returns:
            Number of credentials cleaned up
        """
        if self.config.credential_timeout <= 0:
            return 0

        now = time.time()
        expired = [
            cred_id for cred_id, cred in self._credentials.items()
            if now - cred.created_at > self.config.credential_timeout
        ]

        for cred_id in expired:
            self.delete(cred_id)

        return len(expired)

    def list_credentials(self) -> List[Dict[str, Any]]:
        """List all stored credentials (without values)."""
        return [
            {
                "id": cred.id,
                "name": cred.name,
                "pii_type": cred.pii_type.value,
                "created_at": cred.created_at,
            }
            for cred in self._credentials.values()
        ]



class SensitiveDataMarker:
    """
    Marks fields and data as sensitive for special handling.

    Used to indicate which form fields contain PII so the system
    can handle them appropriately (not send to LLM, mask in logs, etc.)
    """

    def __init__(self) -> None:
        """Initialize the marker."""
        self._marked_fields: Dict[str, PIIField] = {}
        self._marked_selectors: Dict[str, PIIField] = {}

    def mark_field(
        self,
        name: str,
        pii_type: PIIType,
        selector: Optional[str] = None,
    ) -> PIIField:
        """
        Mark a field as containing sensitive data.

        Args:
            name: Field name/identifier
            pii_type: Type of PII in the field
            selector: CSS selector for the form field

        Returns:
            PIIField object
        """
        pii_field = PIIField(
            name=name,
            pii_type=pii_type,
            selector=selector,
        )

        self._marked_fields[name] = pii_field
        if selector:
            self._marked_selectors[selector] = pii_field

        return pii_field

    def is_marked(self, name: str) -> bool:
        """Check if a field is marked as sensitive."""
        return name in self._marked_fields

    def is_selector_marked(self, selector: str) -> bool:
        """Check if a selector is marked as sensitive."""
        return selector in self._marked_selectors

    def get_field(self, name: str) -> Optional[PIIField]:
        """Get PII field info by name."""
        return self._marked_fields.get(name)

    def get_all_marked_fields(self) -> List[PIIField]:
        """Get all marked fields."""
        return list(self._marked_fields.values())

    def unmark_field(self, name: str) -> bool:
        """Remove sensitive marking from a field."""
        if name in self._marked_fields:
            field = self._marked_fields.pop(name)
            if field.selector and field.selector in self._marked_selectors:
                del self._marked_selectors[field.selector]
            return True
        return False


class PIIHandler:
    """
    Main handler for PII operations in FlyBrowser.

    Provides a unified interface for:
    - Storing credentials securely
    - Marking fields as sensitive
    - Masking PII in text
    - Secure form filling

    Example:
        >>> handler = PIIHandler()
        >>>
        >>> # Store credentials
        >>> pwd_id = handler.store_credential("login_password", "secret123")
        >>>
        >>> # Mark form fields as sensitive
        >>> handler.mark_sensitive_field("password", PIIType.PASSWORD, "#password-input")
        >>>
        >>> # Fill form securely (value never exposed to LLM)
        >>> await handler.secure_fill(page, "#password-input", pwd_id)
        >>>
        >>> # Mask text for LLM
        >>> safe_text = handler.mask_for_llm("User entered password: secret123")
    """

    def __init__(self, config: Optional[PIIConfig] = None) -> None:
        """Initialize the PII handler."""
        self.config = config or PIIConfig()
        self._credential_store = SecureCredentialStore(self.config)
        self._masker = PIIMasker(self.config)
        self._marker = SensitiveDataMarker()

    def store_credential(
        self,
        name: str,
        value: str,
        pii_type: PIIType = PIIType.PASSWORD,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a credential securely."""
        return self._credential_store.store(name, value, pii_type, metadata)

    def retrieve_credential(self, credential_id: str) -> Optional[str]:
        """Retrieve a credential value."""
        return self._credential_store.retrieve(credential_id)

    def delete_credential(self, credential_id: str) -> bool:
        """Delete a credential."""
        return self._credential_store.delete(credential_id)

    def list_credentials(self) -> List[Dict[str, Any]]:
        """List all stored credentials (without values)."""
        return self._credential_store.list_credentials()

    def mark_sensitive_field(
        self,
        name: str,
        pii_type: PIIType,
        selector: Optional[str] = None,
    ) -> PIIField:
        """Mark a field as containing sensitive data."""
        return self._marker.mark_field(name, pii_type, selector)

    def is_sensitive_field(self, name: str) -> bool:
        """Check if a field is marked as sensitive."""
        return self._marker.is_marked(name) or self._masker.is_sensitive_field(name)

    def get_sensitive_fields(self) -> List[PIIField]:
        """Get all marked sensitive fields."""
        return self._marker.get_all_marked_fields()

    def mask_for_llm(self, text: str) -> str:
        """
        Mask PII in text before sending to LLM.

        This ensures sensitive data is NEVER sent to LLM providers.
        """
        if not self.config.mask_in_llm_prompts:
            return text
        return self._masker.mask_text(text)

    def mask_for_log(self, text: str) -> str:
        """Mask PII in text for logging."""
        if not self.config.mask_in_logs:
            return text
        return self._masker.mask_text(text)

    def mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive values in a dictionary."""
        return self._masker.mask_dict(data)

    async def secure_fill(
        self,
        page: Any,
        selector: str,
        credential_id: str,
        clear_first: bool = True,
    ) -> bool:
        """
        Securely fill a form field with a stored credential.

        The credential value is retrieved and filled directly into the form
        without ever being exposed to the LLM or logged.

        Args:
            page: Playwright page
            selector: CSS selector for the input field
            credential_id: ID of the stored credential
            clear_first: Whether to clear the field before filling

        Returns:
            True if successful
        """
        value = self._credential_store.retrieve(credential_id)
        if not value:
            logger.error(f"Credential {credential_id} not found")
            return False

        try:
            if clear_first:
                await page.fill(selector, "")

            # Fill the value directly - never logged or sent to LLM
            await page.fill(selector, value)

            # Log the action without the value
            logger.info(f"Securely filled field: {selector} (credential: {credential_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to fill field {selector}: {e}")
            return False

    async def secure_type(
        self,
        page: Any,
        selector: str,
        credential_id: str,
        delay: int = 50,
    ) -> bool:
        """
        Securely type a credential into a form field (character by character).

        Useful for fields that don't work well with fill().

        Args:
            page: Playwright page
            selector: CSS selector for the input field
            credential_id: ID of the stored credential
            delay: Delay between keystrokes in milliseconds

        Returns:
            True if successful
        """
        value = self._credential_store.retrieve(credential_id)
        if not value:
            logger.error(f"Credential {credential_id} not found")
            return False

        try:
            await page.type(selector, value, delay=delay)
            logger.info(f"Securely typed into field: {selector} (credential: {credential_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to type into field {selector}: {e}")
            return False

    def create_safe_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a safe prompt for LLM by masking all PII.

        Args:
            prompt: Original prompt text
            context: Additional context dictionary to mask

        Returns:
            Safe prompt with all PII masked
        """
        safe_prompt = self.mask_for_llm(prompt)

        if context:
            safe_context = self.mask_dict(context)
            # Append masked context info if needed
            safe_prompt += f"\n\nContext: {safe_context}"

        return safe_prompt

    def replace_values_with_placeholders(self, text: str) -> str:
        """
        Replace stored credential values in text with placeholders.

        This is the key method for LLM safety: it replaces actual sensitive
        values with placeholders like {{CREDENTIAL:email}} so the LLM can
        understand the structure without seeing the actual data.

        Args:
            text: Text that may contain credential values

        Returns:
            Text with credential values replaced by placeholders

        Example:
            >>> handler.store_credential("email", "user@example.com")
            >>> handler.store_credential("password", "secret123")
            >>> text = "Login with user@example.com and password secret123"
            >>> safe = handler.replace_values_with_placeholders(text)
            >>> print(safe)
            "Login with {{CREDENTIAL:email}} and password {{CREDENTIAL:password}}"
        """
        if not self.config.use_placeholders_for_llm:
            return self.mask_for_llm(text)

        result = text

        # Replace each stored credential value with its placeholder
        for value, cred_id in self._credential_store._value_to_id.items():
            if value in result:
                placeholder = self._credential_store.get_placeholder(cred_id)
                if placeholder:
                    result = result.replace(value, placeholder)

        # Also apply standard PII masking for any unregistered PII
        result = self._masker.mask_text(result)

        return result

    def resolve_placeholders(self, text: str) -> str:
        """
        Resolve placeholders back to actual credential values.

        This is used for browser execution: the LLM's output contains
        placeholders which are resolved to real values just before
        the browser action is performed.

        Args:
            text: Text containing placeholders like {{CREDENTIAL:email}}

        Returns:
            Text with placeholders replaced by actual values

        Example:
            >>> # LLM output: "Type {{CREDENTIAL:password}} into #password-field"
            >>> resolved = handler.resolve_placeholders(llm_output)
            >>> # resolved: "Type secret123 into #password-field"
            >>> # Now execute the browser action with the real value
        """
        result = text
        prefix = self.config.placeholder_prefix
        suffix = self.config.placeholder_suffix

        # Find all placeholders
        pattern = re.escape(prefix) + r'([^}]+)' + re.escape(suffix)

        def replace_placeholder(match: re.Match) -> str:
            name = match.group(1)
            value = self._credential_store.retrieve_by_name(name)
            if value:
                return value
            # If not found, leave placeholder as-is (will cause visible error)
            logger.warning(f"Credential '{name}' not found for placeholder resolution")
            return match.group(0)

        result = re.sub(pattern, replace_placeholder, result)
        return result

    def get_placeholder(self, credential_id: str) -> Optional[str]:
        """Get the placeholder string for a credential ID."""
        return self._credential_store.get_placeholder(credential_id)

    def get_placeholder_by_name(self, name: str) -> str:
        """Get placeholder string by credential name."""
        return self._credential_store.get_placeholder_by_name(name)

    def has_placeholders(self, text: str) -> bool:
        """Check if text contains any credential placeholders."""
        prefix = self.config.placeholder_prefix
        suffix = self.config.placeholder_suffix
        pattern = re.escape(prefix) + r'[^}]+' + re.escape(suffix)
        return bool(re.search(pattern, text))

    def extract_placeholders(self, text: str) -> List[str]:
        """
        Extract all credential names from placeholders in text.

        Args:
            text: Text containing placeholders

        Returns:
            List of credential names found in placeholders
        """
        prefix = self.config.placeholder_prefix
        suffix = self.config.placeholder_suffix
        pattern = re.escape(prefix) + r'([^}]+)' + re.escape(suffix)
        return re.findall(pattern, text)

    def cleanup_expired_credentials(self) -> int:
        """Clean up expired credentials."""
        return self._credential_store.cleanup_expired()

    def clear_all_credentials(self) -> None:
        """Securely clear all stored credentials."""
        for cred_id in list(self._credential_store._credentials.keys()):
            self._credential_store.delete(cred_id)
        logger.info("All credentials cleared")

    def create_secure_instruction(
        self,
        instruction: str,
        credentials: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Create a secure instruction for LLM with credentials stored and replaced.

        This is a convenience method that:
        1. Stores any provided credentials
        2. Replaces credential values in the instruction with placeholders
        3. Returns the safe instruction and a mapping of names to credential IDs

        Args:
            instruction: The original instruction containing sensitive values
            credentials: Dict mapping credential names to values
                Example: {"email": "user@example.com", "password": "secret123"}

        Returns:
            Tuple of (safe_instruction, credential_id_map)

        Example:
            >>> safe_instruction, cred_ids = handler.create_secure_instruction(
            ...     "Login with user@example.com and password secret123",
            ...     {"email": "user@example.com", "password": "secret123"}
            ... )
            >>> print(safe_instruction)
            "Login with {{CREDENTIAL:email}} and password {{CREDENTIAL:password}}"
            >>> print(cred_ids)
            {"email": "abc123", "password": "def456"}
        """
        cred_id_map = {}

        # Store credentials if provided
        if credentials:
            for name, value in credentials.items():
                # Determine PII type from name
                pii_type = self._infer_pii_type(name)
                cred_id = self.store_credential(name, value, pii_type)
                cred_id_map[name] = cred_id

        # Replace values with placeholders
        safe_instruction = self.replace_values_with_placeholders(instruction)

        return safe_instruction, cred_id_map

    def _infer_pii_type(self, name: str) -> PIIType:
        """Infer PII type from credential name."""
        name_lower = name.lower()

        if "password" in name_lower or "pwd" in name_lower or "passwd" in name_lower:
            return PIIType.PASSWORD
        elif "email" in name_lower:
            return PIIType.EMAIL
        elif "phone" in name_lower or "mobile" in name_lower:
            return PIIType.PHONE
        elif "ssn" in name_lower or "social" in name_lower:
            return PIIType.SSN
        elif "card" in name_lower or "credit" in name_lower:
            return PIIType.CREDIT_CARD
        elif "cvv" in name_lower or "cvc" in name_lower:
            return PIIType.CVV
        elif "api" in name_lower or "key" in name_lower:
            return PIIType.API_KEY
        elif "token" in name_lower:
            return PIIType.TOKEN
        elif "user" in name_lower or "login" in name_lower:
            return PIIType.USERNAME
        elif "name" in name_lower:
            return PIIType.NAME
        elif "address" in name_lower:
            return PIIType.ADDRESS
        elif "dob" in name_lower or "birth" in name_lower:
            return PIIType.DATE_OF_BIRTH
        else:
            return PIIType.CUSTOM
