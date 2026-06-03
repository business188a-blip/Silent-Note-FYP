"""
urdu_helper.py
Fixes Urdu text rendering for PDF export.
"""

def fix_urdu(text):
    """Reshape and fix direction of Urdu text for PDF rendering."""
    if not text or not text.strip():
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        config = arabic_reshaper.ArabicReshaper(configuration={
            'delete_harakat':                    False,
            'support_zwj':                       True,
            'RIAL SIGN':                         True,
            'support_ligatures':                 True,
            'Reh_Zain_Heh_Ligature':            True,
            'support_combined_characters':       True,
        })

        reshaped = config.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def is_urdu(text):
    """Check if text contains Urdu characters."""
    if not text:
        return False
    for ch in text:
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            return True
    return False