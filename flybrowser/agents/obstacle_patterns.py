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
Multi-language obstacle patterns for world-class obstacle detector.

Comprehensive database of cookie consent, GDPR, age verification, and other
obstacle-related keywords in 50+ languages. Used as fallback when AI is unavailable
or for pattern-matching optimization.
"""

from typing import Dict, List, Set

# Cookie consent and GDPR keywords by language
COOKIE_KEYWORDS: Dict[str, List[str]] = {
    "en": [  # English
        "cookie", "cookies", "consent", "privacy", "accept", "agree", "gdpr",
        "tracking", "preferences", "settings", "customize", "manage cookies",
        "we use cookies", "this website uses", "continue without"
    ],
    "es": [  # Spanish
        "cookie", "cookies", "consentimiento", "privacidad", "aceptar", "acepto",
        "estar de acuerdo", "configurar", "preferencias", "rechazar", "denegar",
        "este sitio usa", "utilizamos cookies", "política de cookies"
    ],
    "fr": [  # French
        "cookie", "cookies", "consentement", "confidentialité", "accepter", "j'accepte",
        "d'accord", "refuser", "personnaliser", "paramètres", "préférences",
        "nous utilisons", "ce site utilise", "politique de cookies"
    ],
    "de": [  # German
        "cookie", "cookies", "zustimmung", "datenschutz", "akzeptieren", "einverstanden",
        "ablehnen", "einstellungen", "präferenzen", "anpassen",
        "wir verwenden", "diese website verwendet", "cookie-richtlinie"
    ],
    "it": [  # Italian
        "cookie", "cookies", "consenso", "privacy", "accetta", "accetto",
        "rifiuta", "personalizza", "impostazioni", "preferenze",
        "questo sito utilizza", "utilizziamo cookie", "politica sui cookie"
    ],
    "pt": [  # Portuguese
        "cookie", "cookies", "consentimento", "privacidade", "aceitar", "aceito",
        "recusar", "personalizar", "configurações", "preferências",
        "este site usa", "usamos cookies", "política de cookies"
    ],
    "nl": [  # Dutch
        "cookie", "cookies", "toestemming", "privacy", "accepteren", "weigeren",
        "instellingen", "voorkeuren", "aanpassen",
        "we gebruiken", "deze website gebruikt", "cookiebeleid"
    ],
    "pl": [  # Polish
        "cookie", "cookies", "zgoda", "prywatność", "akceptuj", "odrzuć",
        "ustawienia", "preferencje", "dostosuj",
        "używamy", "ta strona używa", "polityka cookies"
    ],
    "ru": [  # Russian
        "cookie", "куки", "согласие", "конфиденциальность", "принять", "отклонить",
        "настройки", "предпочтения", "мы используем", "этот сайт использует"
    ],
    "ja": [  # Japanese
        "クッキー", "cookie", "同意", "プライバシー", "承認", "拒否",
        "設定", "カスタマイズ", "このサイトは使用"
    ],
    "zh": [  # Chinese
        "cookie", "曲奇", "同意", "隐私", "接受", "拒绝", "设置", "偏好",
        "我们使用", "本网站使用"
    ],
    "ko": [  # Korean
        "쿠키", "cookie", "동의", "개인정보", "수락", "거부", "설정",
        "이 사이트는 사용"
    ],
    "ar": [  # Arabic
        "cookie", "ملف تعريف", "موافقة", "خصوصية", "قبول", "رفض", "إعدادات",
        "نستخدم", "يستخدم هذا الموقع"
    ],
    "tr": [  # Turkish
        "cookie", "çerez", "onay", "gizlilik", "kabul", "reddet", "ayarlar",
        "kullanıyoruz", "bu site kullanıyor"
    ],
    "sv": [  # Swedish
        "cookie", "kakor", "samtycke", "integritet", "acceptera", "avvisa",
        "inställningar", "vi använder", "denna webbplats använder"
    ],
    "no": [  # Norwegian
        "cookie", "informasjonskapsler", "samtykke", "personvern", "aksepter", "avvis",
        "innstillinger", "vi bruker", "dette nettstedet bruker"
    ],
    "da": [  # Danish
        "cookie", "cookies", "samtykke", "privatliv", "accepter", "afvis",
        "indstillinger", "vi bruger", "denne hjemmeside bruger"
    ],
    "fi": [  # Finnish
        "cookie", "evästeet", "suostumus", "tietosuoja", "hyväksy", "hylkää",
        "asetukset", "käytämme", "tämä sivusto käyttää"
    ],
    "el": [  # Greek
        "cookie", "cookies", "συγκατάθεση", "απόρρητο", "αποδοχή", "απόρριψη",
        "ρυθμίσεις", "χρησιμοποιούμε", "αυτός ο ιστότοπος χρησιμοποιεί"
    ],
    "cs": [  # Czech
        "cookie", "cookies", "souhlas", "soukromí", "přijmout", "odmítnout",
        "nastavení", "používáme", "tento web používá"
    ],
    "hu": [  # Hungarian
        "cookie", "sütik", "hozzájárulás", "adatvédelem", "elfogad", "elutasít",
        "beállítások", "használunk", "ez a weboldal használ"
    ],
    "ro": [  # Romanian
        "cookie", "cookies", "consimțământ", "confidențialitate", "accept", "resping",
        "setări", "folosim", "acest site folosește"
    ],
    "bg": [  # Bulgarian
        "cookie", "бисквитки", "съгласие", "поверителност", "приемам", "отхвърлям",
        "настройки", "използваме", "този сайт използва"
    ],
    "hr": [  # Croatian
        "cookie", "kolačići", "pristanak", "privatnost", "prihvaćam", "odbijam",
        "postavke", "koristimo", "ova stranica koristi"
    ],
    "sk": [  # Slovak
        "cookie", "cookies", "súhlas", "súkromie", "prijať", "odmietnuť",
        "nastavenia", "používame", "táto stránka používa"
    ],
    "sl": [  # Slovenian
        "cookie", "piškotki", "soglasje", "zasebnost", "sprejmi", "zavrni",
        "nastavitve", "uporabljamo", "ta spletna stran uporablja"
    ],
    "lt": [  # Lithuanian
        "cookie", "slapukai", "sutikimas", "privatumas", "priimti", "atmesti",
        "nustatymai", "naudojame", "ši svetainė naudoja"
    ],
    "lv": [  # Latvian
        "cookie", "sīkdatnes", "piekrišana", "privātums", "pieņemt", "noraidīt",
        "iestatījumi", "izmantojam", "šī vietne izmanto"
    ],
    "et": [  # Estonian
        "cookie", "küpsised", "nõusolek", "privaatsus", "nõustun", "keeldun",
        "seaded", "kasutame", "see veebileht kasutab"
    ],
    "th": [  # Thai
        "cookie", "คุกกี้", "ความยินยอม", "ความเป็นส่วนตัว", "ยอมรับ", "ปฏิเสธ",
        "การตั้งค่า", "เราใช้", "เว็บไซต์นี้ใช้"
    ],
    "vi": [  # Vietnamese
        "cookie", "cookies", "đồng ý", "quyền riêng tư", "chấp nhận", "từ chối",
        "cài đặt", "chúng tôi sử dụng", "trang web này sử dụng"
    ],
    "id": [  # Indonesian
        "cookie", "cookies", "persetujuan", "privasi", "terima", "tolak",
        "pengaturan", "kami menggunakan", "situs ini menggunakan"
    ],
    "ms": [  # Malay
        "cookie", "cookies", "persetujuan", "privasi", "terima", "tolak",
        "tetapan", "kami gunakan", "laman web ini menggunakan"
    ],
    "he": [  # Hebrew
        "cookie", "עוגיות", "הסכמה", "פרטיות", "אישור", "דחייה",
        "הגדרות", "אנחנו משתמשים", "אתר זה משתמש"
    ],
    "hi": [  # Hindi
        "cookie", "कुकी", "सहमति", "गोपनीयता", "स्वीकार", "अस्वीकार",
        "सेटिंग्स", "हम उपयोग करते हैं"
    ],
    "bn": [  # Bengali
        "cookie", "কুকি", "সম্মতি", "গোপনীয়তা", "গ্রহণ", "প্রত্যাখ্যান",
        "সেটিংস"
    ],
    "ur": [  # Urdu
        "cookie", "کوکی", "رضامندی", "رازداری", "قبول", "مسترد",
        "ترتیبات"
    ],
    "fa": [  # Persian
        "cookie", "کوکی", "رضایت", "حریم خصوصی", "قبول", "رد",
        "تنظیمات"
    ],
}

# Common dismiss button text (across all languages)
DISMISS_BUTTON_TEXT: Set[str] = {
    # English
    "accept", "accept all", "agree", "allow", "ok", "got it", "i agree", "continue",
    "close", "dismiss", "allow all", "accept cookies", "i understand",
    # Spanish
    "aceptar", "aceptar todo", "aceptar todas", "acepto", "continuar", "entendido",
    "cerrar", "permitir", "de acuerdo", "estoy de acuerdo",
    # French
    "accepter", "accepter tout", "j'accepte", "d'accord", "continuer", "compris",
    "fermer", "autoriser", "je comprends",
    # German
    "akzeptieren", "alle akzeptieren", "einverstanden", "zustimmen", "weiter",
    "verstanden", "schließen", "erlauben", "ich stimme zu",
    # Italian
    "accetta", "accetta tutto", "accetto", "continua", "capito", "chiudi",
    "consenti", "sono d'accordo",
    # Portuguese
    "aceitar", "aceitar tudo", "aceito", "continuar", "entendi", "fechar",
    "permitir", "concordo",
    # Dutch
    "accepteren", "alles accepteren", "akkoord", "doorgaan", "begrepen",
    "sluiten", "toestaan",
    # Polish
    "akceptuj", "akceptuj wszystko", "zgadzam się", "kontynuuj", "rozumiem",
    "zamknij", "zezwól",
    # Russian
    "принять", "принять все", "согласен", "продолжить", "понятно", "закрыть",
    # Add more as needed...
}

# Age verification keywords
AGE_VERIFICATION_KEYWORDS: Set[str] = {
    "age", "years old", "18+", "21+", "adult", "verify", "confirm",
    "edad", "años", "adulto", "verificar", "confirmar",
    "âge", "ans", "vérifier",
    "alter", "jahre", "erwachsene", "bestätigen",
    "età", "anni", "verifica",
    "idade", "anos", "verificar",
}

# Modal/overlay keywords
MODAL_KEYWORDS: Set[str] = {
    "modal", "overlay", "popup", "dialog", "banner", "notice", "alert",
    "lightbox", "backdrop", "cover", "mask", "interstitial",
}

# CSS selectors commonly used for obstacles
COMMON_OBSTACLE_SELECTORS: List[str] = [
    # Cookie consent libraries
    "#onetrust-accept-btn-handler",
    ".onetrust-close-btn-handler",
    "#cookie-consent-accept",
    ".cookie-accept",
    ".cookie-consent-button",
    ".gdpr-accept",
    
    # Generic patterns
    "[class*='cookie'][class*='accept']",
    "[class*='cookie'][class*='agree']",
    "[class*='consent'][class*='accept']",
    "[class*='gdpr'][class*='accept']",
    
    # Common frameworks
    ".cc-dismiss",  # Cookie Consent
    ".cc-allow",
    "#CybotCookiebotDialogBodyButtonAccept",  # Cookiebot
    ".didomi-continue-without-agreeing",  # Didomi
    ".qc-cmp-button",  # Quantcast
    
    # Modal close buttons
    ".modal-close",
    ".close-modal",
    "[aria-label='Close']",
    "[aria-label='Dismiss']",
    "button.close",
]

# XPath expressions for obstacles
COMMON_OBSTACLE_XPATHS: List[str] = [
    "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
    "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
    "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'aceptar')]",
    "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
    "//div[@role='button'][contains(., 'Accept')]",
]


def get_all_cookie_keywords() -> Set[str]:
    """Get all cookie keywords from all languages."""
    keywords = set()
    for lang_keywords in COOKIE_KEYWORDS.values():
        keywords.update(lang_keywords)
    return keywords


def detect_language_from_text(text: str) -> List[str]:
    """
    Detect likely languages based on text content.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of detected language codes, ranked by likelihood
    """
    text_lower = text.lower()
    matches = []
    
    for lang, keywords in COOKIE_KEYWORDS.items():
        match_count = sum(1 for keyword in keywords if keyword in text_lower)
        if match_count > 0:
            matches.append((lang, match_count))
    
    # Sort by match count descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return [lang for lang, _ in matches]


def is_cookie_text(text: str) -> bool:
    """
    Check if text appears to be cookie/consent related.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains cookie-related keywords
    """
    text_lower = text.lower()
    all_keywords = get_all_cookie_keywords()
    return any(keyword in text_lower for keyword in all_keywords)


def get_dismiss_button_candidates(text: str) -> List[str]:
    """
    Get potential dismiss button text from a string.
    
    Args:
        text: Text to search
        
    Returns:
        List of matching dismiss button texts
    """
    text_lower = text.lower()
    return [btn for btn in DISMISS_BUTTON_TEXT if btn in text_lower]
