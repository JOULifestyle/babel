from __future__ import annotations

import copy
import logging
import re
import statistics
import unicodedata
from functools import cache

import pymupdf
import regex
from rtree import index

from babeldoc.const import WATERMARK_VERSION
from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import PdfCharacter
from babeldoc.format.pdf.document_il import PdfCurve
from babeldoc.format.pdf.document_il import PdfForm
from babeldoc.format.pdf.document_il import PdfFormula
from babeldoc.format.pdf.document_il import PdfParagraphComposition
from babeldoc.format.pdf.document_il import PdfStyle
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.formular_helper import update_formula_data
from babeldoc.format.pdf.document_il.utils.layout_helper import box_to_tuple
from babeldoc.format.pdf.document_il.utils.layout_helper import get_char_unicode_string
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode

logger = logging.getLogger(__name__)

LINE_BREAK_REGEX = regex.compile(
    r"^["
    r"a-z"
    r"A-Z"
    r"0-9"
    r"\u00C0-\u00FF"  # Latin-1 Supplement
    r"\u0100-\u017F"  # Latin Extended A
    r"\u0180-\u024F"  # Latin Extended B
    r"\u1E00-\u1EFF"  # Latin Extended Additional
    r"\u2C60-\u2C7F"  # Latin Extended C
    r"\uA720-\uA7FF"  # Latin Extended D
    r"\uAB30-\uAB6F"  # Latin Extended E
    r"\u0250-\u02A0"  # IPA Extensions
    r"\u0400-\u04FF"  # Cyrillic
    r"\u0300-\u036F"  # Combining Diacritical Marks
    r"\u0500-\u052F"  # Cyrillic Supplement
    r"\u0370-\u03FF"  # Greek and Coptic
    r"\u2DE0-\u2DFF"  # Cyrillic Extended-A
    r"\uA650-\uA69F"  # Cyrillic Extended-B
    r"\u1200-\u137F"  # Ethiopic
    r"\u1380-\u139F"  # Ethiopic Supplement
    r"\u2D80-\u2DDF"  # Ethiopic Extended
    r"\uAB00-\uAB2F"  # Ethiopic Extended-A
    r"\U0001E7E0-\U0001E7FF"  # Ethiopic Extended-B
    r"\u0E80-\u0EFF"  # Lao
    r"\u0D00-\u0D7F"  # Malayalam
    r"\u0A80-\u0AFF"  # Gujarati
    r"\u0E00-\u0E7F"  # Thai
    r"\u1000-\u109F"  # Myanmar
    r"\uAA60-\uAA7F"  # Myanmar Extended-A
    r"\uA9E0-\uA9FF"  # Myanmar Extended-B
    r"\U000116D0-\U000116FF"  # Myanmar Extended-C
    r"\u0B80-\u0BFF"  # Tamil
    r"\u0C00-\u0C7F"  # Telugu
    r"\u0B00-\u0B7F"  # Oriya
    r"\u0530-\u058F"  # Armenian
    r"\u10A0-\u10FF"  # Georgian
    r"\u1C90-\u1CBF"  # Georgian Extended
    r"\u2D00-\u2D2F"  # Georgian Supplement
    r"\u1780-\u17FF"  # Khmer
    r"\u19E0-\u19FF"  # Khmer Symbols
    r"\U00010B00-\U00010B3F"  # Avestan
    r"\u1D00-\u1D7F"  # Phonetic Extensions
    r"\u1400-\u167F"  # Unified Canadian Aboriginal Syllabics
    r"\u0B00-\u0B7F"  # Oriya
    r"\u0780-\u07BF"  # Thaana
    r"\U0001E900-\U0001E95F"  # Adlam
    r"\u1C80-\u1C8F"  # Cyrillic Extended-C
    r"\U0001E030-\U0001E08F"  # Cyrillic Extended-D
    r"\uA000-\uA48F"  # Yi Syllables
    r"\uA490-\uA4CF"  # Yi Radicals
    r"'"
    r"-"  # Hyphen
    r"·"  # Middle Dot (U+00B7) For Català
    r"ʻ"  # Spacing Modifier Letters U+02BB
    r"]+$"
)


# ============================================================================
# LAYOUT-AWARE FORMULA RELOCATION
# These functions preserve the structural relationships (baselines, vertical
# levels) of formula characters during relocation, preventing distortion.
# ============================================================================

def group_chars_by_vertical_level(chars: list, tolerance: float = 2.0) -> dict:
    """
    Group formula characters by their vertical position (baseline level).
    
    Characters at the same vertical level (e.g., main line, subscript, superscript)
    are grouped together to preserve their relative alignment during relocation.
    
    Args:
        chars: List of PdfCharacter objects
        tolerance: Vertical tolerance for grouping (pixels)
    
    Returns:
        Dict mapping level_id to list of (char, original_y_center) tuples
    """
    if not chars:
        return {}
    
    levels = {}
    level_id = 0
    
    # Sort chars by y position for easier grouping
    sorted_chars = sorted(chars, key=lambda c: c.visual_bbox.box.y if c.visual_bbox else c.box.y)
    
    for char in sorted_chars:
        y_center = (char.visual_bbox.box.y + char.visual_bbox.box.y2) / 2 if char.visual_bbox else (char.box.y + char.box.y2) / 2
        
        # Find if this char belongs to an existing level
        assigned = False
        for lid, level_chars in levels.items():
            if level_chars:
                existing_y = level_chars[0][1]
                if abs(y_center - existing_y) < tolerance:
                    levels[lid].append((char, y_center))
                    assigned = True
                    break
        
        if not assigned:
            levels[level_id] = [(char, y_center)]
            level_id += 1
    
    return levels


def calculate_level_baseline(level_chars: list) -> float:
    """
    Calculate the baseline y-position for a group of characters at the same level.
    
    Args:
        level_chars: List of (char, y_center) tuples
    
    Returns:
        Baseline y-position for this level
    """
    if not level_chars:
        return 0.0
    
    # Use the average y-center as the baseline
    y_centers = [yc for _, yc in level_chars]
    return sum(y_centers) / len(y_centers)


def calculate_level_spacing(levels: dict) -> dict:
    """
    Calculate the spacing between different vertical levels.
    
    Args:
        levels: Dict from group_chars_by_vertical_level
    
    Returns:
        Dict mapping (level_id, level_id) pairs to their vertical spacing
    """
    baselines = {lid: calculate_level_baseline(chars) for lid, chars in levels.items()}
    spacing = {}
    
    sorted_lids = sorted(baselines.keys(), key=lambda lid: baselines[lid])
    
    for i in range(len(sorted_lids) - 1):
        lid1 = sorted_lids[i]
        lid2 = sorted_lids[i + 1]
        spacing[(lid1, lid2)] = baselines[lid2] - baselines[lid1]
    
    return spacing


def get_font_size_for_level(level_chars: list) -> float:
    """
    Get the predominant font size for a level (for scaling calculations).
    
    Args:
        level_chars: List of (char, y_center) tuples
    
    Returns:
        Most common font size in this level
    """
    if not level_chars:
        return 12.0  # Default
    
    font_sizes = [char.pdf_style.font_size for char, _ in level_chars if char.pdf_style and char.pdf_style.font_size]
    if not font_sizes:
        return 12.0
    
    # Return most common font size
    from collections import Counter
    counter = Counter(font_sizes)
    return counter.most_common(1)[0][0]


class TypesettingUnit:
    def __str__(self):
        return self.try_get_unicode() or ""

    def __init__(
        self,
        char: PdfCharacter | None = None,
        formular: PdfFormula | None = None,
        unicode: str | None = None,
        font: pymupdf.Font | None = None,
        original_font: il_version_1.PdfFont | None = None,
        font_size: float | None = None,
        style: PdfStyle | None = None,
        xobj_id: int | None = None,
        debug_info: bool = False,
    ):
        assert (char is not None) + (formular is not None) + (
            unicode is not None
        ) == 1, "Only one of chars and formular can be not None"
        self.char = char
        self.formular = formular
        self.unicode = unicode
        self.x = None
        self.y = None
        self.scale = None
        self.debug_info = debug_info

        # Cache variables
        self.box_cache: Box | None = None
        self.can_break_line_cache: bool | None = None
        self.is_cjk_char_cache: bool | None = None
        self.mixed_character_blacklist_cache: bool | None = None
        self.is_space_cache: bool | None = None
        self.is_hung_punctuation_cache: bool | None = None
        self.is_cannot_appear_in_line_end_punctuation_cache: bool | None = None
        self.can_passthrough_cache: bool | None = None
        self.width_cache: float | None = None
        self.height_cache: float | None = None

        self.font_size: float | None = None

        if unicode:
            assert font_size, "Font size must be provided when unicode is provided"
            assert style, "Style must be provided when unicode is provided"
            assert len(unicode) == 1, "Unicode must be a single character"
            assert xobj_id is not None, (
                "Xobj id must be provided when unicode is provided"
            )

            self.font = font
            if font is not None and hasattr(font, "font_id"):
                self.font_id = font.font_id
            else:
                self.font_id = "base"
            if original_font:
                self.original_font = original_font
            else:
                self.original_font = None

            self.font_size = font_size
            self.style = style
            self.xobj_id = xobj_id

    def try_resue_cache(self, old_tu: TypesettingUnit):
        if old_tu.is_cjk_char_cache is not None:
            self.is_cjk_char_cache = old_tu.is_cjk_char_cache

        if old_tu.can_break_line_cache is not None:
            self.can_break_line_cache = old_tu.can_break_line_cache

        if old_tu.is_space_cache is not None:
            self.is_space_cache = old_tu.is_space_cache

        if old_tu.is_hung_punctuation_cache is not None:
            self.is_hung_punctuation_cache = old_tu.is_hung_punctuation_cache

        if old_tu.is_cannot_appear_in_line_end_punctuation_cache is not None:
            self.is_cannot_appear_in_line_end_punctuation_cache = (
                old_tu.is_cannot_appear_in_line_end_punctuation_cache
            )

        if old_tu.can_passthrough_cache is not None:
            self.can_passthrough_cache = old_tu.can_passthrough_cache

        if old_tu.mixed_character_blacklist_cache is not None:
            self.mixed_character_blacklist_cache = (
                old_tu.mixed_character_blacklist_cache
            )

    def try_get_unicode(self) -> str | None:
        if self.char:
            return self.char.char_unicode
        elif self.formular:
            return None
        elif self.unicode:
            return self.unicode

    @property
    def mixed_character_blacklist(self):
        if self.mixed_character_blacklist_cache is None:
            self.mixed_character_blacklist_cache = self.calc_mixed_character_blacklist()

        return self.mixed_character_blacklist_cache

    def calc_mixed_character_blacklist(self):
        unicode = self.try_get_unicode()
        if unicode:
            return unicode in [
                "。",
                "，",
                "：",
                "？",
                "！",
            ]
        return False

    @property
    def can_break_line(self):
        if self.can_break_line_cache is None:
            self.can_break_line_cache = self.calc_can_break_line()

        return self.can_break_line_cache

    def calc_can_break_line(self):
        unicode = self.try_get_unicode()
        if not unicode:
            return True
        if LINE_BREAK_REGEX.match(unicode):
            return False
        return True

    @property
    def is_cjk_char(self):
        if self.is_cjk_char_cache is None:
            self.is_cjk_char_cache = self.calc_is_cjk_char()

        return self.is_cjk_char_cache

    def calc_is_cjk_char(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()
        if not unicode:
            return False
        if "(cid" in unicode:
            return False
        if len(unicode) > 1:
            return False
        assert len(unicode) == 1, "Unicode must be a single character"
        if unicode in [
            "（",
            "）",
            "【",
            "】",
            "《",
            "》",
            "〔",
            "〕",
            "〈",
            "〉",
            "〖",
            "〗",
            "「",
            "」",
            "『",
            "』",
            "、",
            "。",
            "：",
            "？",
            "！",
            "，",
        ]:
            return True
        if unicode:
            if re.match(
                r"^["
                r"\u3000-\u303f"  # CJK Symbols and Punctuation
                r"\u3040-\u309f"  # Hiragana
                r"\u30a0-\u30ff"  # Katakana
                r"\u3100-\u312f"  # Bopomofo
                r"\uac00-\ud7af"  # Hangul Syllables
                r"\u1100-\u11ff"  # Hangul Jamo
                r"\u3130-\u318f"  # Hangul Compatibility Jamo
                r"\ua960-\ua97f"  # Hangul Jamo Extended-A
                r"\ud7b0-\ud7ff"  # Hangul Jamo Extended-B
                r"\u3190-\u319f"  # Kanbun
                r"\u3200-\u32ff"  # Enclosed CJK Letters and Months
                r"\u3300-\u33ff"  # CJK Compatibility
                r"\ufe30-\ufe4f"  # CJK Compatibility Forms
                r"\u4e00-\u9fff"  # CJK Unified Ideographs
                r"\u2e80-\u2eff"  # CJK Radicals Supplement
                r"\u31c0-\u31ef"  # CJK Strokes
                r"\u2f00-\u2fdf"  # Kangxi Radicals
                r"\ufe10-\ufe1f"  # Vertical Forms
                r"]+$",
                unicode,
            ):
                return True
            try:
                unicodedata_name = unicodedata.name(unicode)
                return (
                    "CJK UNIFIED IDEOGRAPH" in unicodedata_name
                    or "FULLWIDTH" in unicodedata_name
                )
            except ValueError:
                return False
        return False

    @property
    def is_space(self):
        if self.is_space_cache is None:
            self.is_space_cache = self.calc_is_space()

        return self.is_space_cache

    def calc_is_space(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()
        return unicode == " "

    @property
    def is_hung_punctuation(self):
        if self.is_hung_punctuation_cache is None:
            self.is_hung_punctuation_cache = self.calc_is_hung_punctuation()

        return self.is_hung_punctuation_cache

    def calc_is_hung_punctuation(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()

        if unicode:
            return unicode in [
                # 英文标点
                ",",
                ".",
                ":",
                ";",
                "?",
                "!",
                # 中文点号
                "，",  # 逗号
                "。",  # 句号
                "．",  # 全角句号
                "、",  # 顿号
                "：",  # 冒号
                "；",  # 分号
                "！",  # 叹号
                "‼",  # 双叹号
                "？",  # 问号
                "⁇",  # 双问号
                # 结束引号
                "”",  # 右双引号
                "’",  # 右单引号
                "」",  # 右直角单引号
                "』",  # 右直角双引号
                # 结束括号
                ")",  # 右圆括号
                "]",  # 右方括号
                "}",  # 右花括号
                "）",  # 右圆括号
                "〕",  # 右龟甲括号
                "〉",  # 右单书名号
                "】",  # 右黑色方头括号
                "〗",  # 右空白方头括号
                "］",  # 全角右方括号
                "｝",  # 全角右花括号
                # 结束双书名号
                "》",  # 右双书名号
                # 连接号
                "～",  # 全角波浪号
                "-",  # 连字符减号
                "–",  # 短破折号 (EN DASH)
                "—",  # 长破折号 (EM DASH)
                # 间隔号
                "·",  # 中间点
                "・",  # 片假名中间点
                "‧",  # 连字点
                # 分隔号
                "/",  # 斜杠
                "／",  # 全角斜杠
                "⁄",  # 分数斜杠
            ]
        return False

    @property
    def is_cannot_appear_in_line_end_punctuation(self):
        if self.is_cannot_appear_in_line_end_punctuation_cache is None:
            self.is_cannot_appear_in_line_end_punctuation_cache = (
                self.calc_is_cannot_appear_in_line_end_punctuation()
            )

        return self.is_cannot_appear_in_line_end_punctuation_cache

    def calc_is_cannot_appear_in_line_end_punctuation(self):
        if self.formular:
            return False
        unicode = self.try_get_unicode()
        if not unicode:
            return False
        return unicode in [
            # 开始引号
            "“",  # 左双引号
            "‘",  # 左单引号
            "「",  # 左直角单引号
            "『",  # 左直角双引号
            # 开始括号
            "(",  # 左圆括号
            "[",  # 左方括号
            "{",  # 左花括号
            "（",  # 左圆括号
            "〔",  # 左龟甲括号
            "〈",  # 左单书名号
            "《",  # 左双书名号
            # 开始单双书名号
            "〖",  # 左空白方头括号
            "〘",  # 左黑色方头括号
            "〚",  # 左单书名号
        ]

    def passthrough(
        self,
    ) -> tuple[list[PdfCharacter], list[PdfCurve], list[PdfForm]]:
        if self.char:
            return [self.char], [], []
        elif self.formular:
            return (
                self.formular.pdf_character,
                self.formular.pdf_curve,
                self.formular.pdf_form,
            )
        elif self.unicode:
            logger.error(f"Cannot passthrough unicode. TypesettingUnit: {self}. ")
            logger.error(f"Cannot passthrough unicode. TypesettingUnit: {self}. ")
            return [], [], []

    @property
    def can_passthrough(self):
        if self.can_passthrough_cache is None:
            self.can_passthrough_cache = self.calc_can_passthrough()

        return self.can_passthrough_cache

    def calc_can_passthrough(self):
        return self.unicode is None

    def calculate_box(self):
        if self.char:
            box = copy.deepcopy(self.char.box)
            if self.char.visual_bbox and self.char.visual_bbox.box:
                box.y = self.char.visual_bbox.box.y
                box.y2 = self.char.visual_bbox.box.y2
                # return self.char.visual_bbox.box

            return box
        elif self.formular:
            return self.formular.box
            # if self.formular.x_offset <= 0.5:
            #     return self.formular.box
            # formular_box = copy.copy(self.formular.box)
            # formular_box.x2 += self.formular.x_advance
            # return formular_box
        elif self.unicode:
            char_width = self.font.char_lengths(self.unicode, self.font_size)[0]
            if self.x is None or self.y is None or self.scale is None:
                return Box(0, 0, char_width, self.font_size)
            return Box(self.x, self.y, self.x + char_width, self.y + self.font_size)

    @property
    def box(self):
        if not self.box_cache:
            self.box_cache = self.calculate_box()

        return self.box_cache

    @property
    def width(self):
        if self.width_cache is None:
            self.width_cache = self.calc_width()

        return self.width_cache

    def calc_width(self):
        box = self.box
        return box.x2 - box.x

    @property
    def height(self):
        if self.height_cache is None:
            self.height_cache = self.calc_height()

        return self.height_cache

    def calc_height(self):
        box = self.box
        return box.y2 - box.y

    def relocate(
        self,
        x: float,
        y: float,
        scale: float,
    ) -> TypesettingUnit:
        """重定位并缩放排版单元

        Args:
            x: 新的 x 坐标
            y: 新的 y 坐标
            scale: 缩放因子

        Returns:
            新的排版单元
        """
        if self.char:
            # 创建新的字符对象
            new_char = PdfCharacter(
                pdf_character_id=self.char.pdf_character_id,
                char_unicode=self.char.char_unicode,
                box=Box(
                    x=x,
                    y=y,
                    x2=x + self.width * scale,
                    y2=y + self.height * scale,
                ),
                pdf_style=PdfStyle(
                    font_id=self.char.pdf_style.font_id,
                    font_size=self.char.pdf_style.font_size * scale,
                    graphic_state=self.char.pdf_style.graphic_state,
                ),
                scale=scale,
                vertical=self.char.vertical,
                advance=self.char.advance * scale if self.char.advance else None,
                debug_info=self.debug_info,
                xobj_id=self.char.xobj_id,
            )
            new_tu = TypesettingUnit(char=new_char)
            new_tu.try_resue_cache(self)
            return new_tu

        elif self.formular:
            # Layout-aware relocation preserving formula structure
            # Group chars by vertical level (main line, subscript, superscript)
            # to preserve relative alignment
            levels = group_chars_by_vertical_level(self.formular.pdf_character)
            
            new_chars = []
            min_x = self.formular.box.x
            min_y = self.formular.box.y

            for level_id, level_items in levels.items():
                # Calculate baseline for this level
                level_chars = [item[0] for item in level_items]
                original_level_baseline = calculate_level_baseline(level_items) # Average Y-center
                
                # Calculate relative Y position of this level within formula
                rel_level_y = original_level_baseline - min_y
                
                for char in level_chars:
                    # Calculate relative X position
                    rel_x = char.box.x - min_x
                    visual_rel_x = char.visual_bbox.box.x - min_x
                    
                    # For Y, use the LEVEL's baseline to align characters (snap to grid effect)
                    # This fixes "jittery" characters in formulas
                    
                    # Calculate target Y center for this level in new coordinates
                    new_level_baseline_y = y + (rel_level_y + self.formular.y_offset) * scale
                    
                    # Calculate character dimensions
                    char_height = char.box.y2 - char.box.y
                    char_width = char.box.x2 - char.box.x
                    
                    new_height = char_height * scale
                    new_width = char_width * scale
                    
                    # Center the character vertically on the new level baseline
                    new_y = new_level_baseline_y - (new_height / 2)
                    new_y2 = new_level_baseline_y + (new_height / 2)
                    
                    # Calculate X positions
                    new_x_pos = x + (rel_x + self.formular.x_offset) * scale
                    new_x2_pos = new_x_pos + new_width
                    
                    # Create new character
                    new_char = PdfCharacter(
                        pdf_character_id=char.pdf_character_id,
                        char_unicode=char.char_unicode,
                        box=Box(
                            x=new_x_pos,
                            y=new_y,
                            x2=new_x2_pos,
                            y2=new_y2,
                        ),
                        visual_bbox=il_version_1.VisualBbox(
                            box=Box(
                                x=x + (visual_rel_x + self.formular.x_offset) * scale,
                                y=new_y, # Align visual box too
                                x2=x + (visual_rel_x + (char.visual_bbox.box.x2 - char.visual_bbox.box.x) + self.formular.x_offset) * scale,
                                y2=new_y2,
                            ),
                        ),
                        pdf_style=PdfStyle(
                            font_id=char.pdf_style.font_id,
                            font_size=char.pdf_style.font_size * scale,
                            graphic_state=char.pdf_style.graphic_state,
                        ),
                        scale=scale,
                        vertical=char.vertical,
                        advance=char.advance * scale if char.advance else None,
                        xobj_id=char.xobj_id,
                    )
                    new_chars.append(new_char)

            # Calculate bounding box from new_chars
            if new_chars:
                min_x = min(char.box.x for char in new_chars)
                min_y = min(char.box.y for char in new_chars)
                max_x = max(char.box.x2 for char in new_chars)
                max_y = max(char.box.y2 for char in new_chars)
            else:
                min_x, min_y, max_x, max_y = 0, 0, 0, 0

            new_formula = PdfFormula(
                box=Box(
                    x=min_x,
                    y=min_y,
                    x2=max_x,
                    y2=max_y,
                ),
                pdf_character=new_chars,
                x_offset=self.formular.x_offset * scale,
                y_offset=self.formular.y_offset * scale,
                x_advance=self.formular.x_advance * scale,
            )

            # Handle contained curves
            new_curves = []
            for curve in self.formular.pdf_curve:
                new_curve = self._transform_curve_for_relocation(
                    curve,
                    self.formular.box.x,
                    self.formular.box.y,
                    x,
                    y,
                    scale,
                )
                new_curves.append(new_curve)
            new_formula.pdf_curve = new_curves

            # Handle contained forms
            new_forms = []
            for form in self.formular.pdf_form:
                new_form = self._transform_form_for_relocation(
                    form, self.formular.box.x, self.formular.box.y, x, y, scale
                )
                new_forms.append(new_form)
            new_formula.pdf_form = new_forms

            update_formula_data(new_formula)

            new_tu = TypesettingUnit(formular=new_formula)
            new_tu.can_passthrough_cache = True
            new_tu.try_resue_cache(self)
            return new_tu


        elif self.unicode:
            # 对于 Unicode 字符，我们存储新的位置信息
            new_unit = TypesettingUnit(
                unicode=self.unicode,
                font=self.font,
                original_font=self.original_font,
                font_size=self.font_size * scale,
                style=self.style,
                xobj_id=self.xobj_id,
                debug_info=self.debug_info,
            )
            new_unit.x = x
            new_unit.y = y
            new_unit.scale = scale
            new_unit.try_resue_cache(self)
            return new_unit

    def _transform_curve_for_relocation(
        self,
        curve,
        original_formula_x: float,
        original_formula_y: float,
        new_x: float,
        new_y: float,
        scale: float,
    ):
        """Transform a curve for formula relocation."""
        import copy

        new_curve = copy.deepcopy(curve)

        if new_curve.box:
            # Calculate relative position to formula's original position (same as chars)
            rel_x = new_curve.box.x - original_formula_x
            rel_y = new_curve.box.y - original_formula_y

            # Apply same transformation as characters
            new_curve.box = Box(
                x=new_x + (rel_x + self.formular.x_offset) * scale,
                y=new_y + (rel_y + self.formular.y_offset) * scale,
                x2=new_x
                + (
                    rel_x
                    + (new_curve.box.x2 - new_curve.box.x)
                    + self.formular.x_offset
                )
                * scale,
                y2=new_y
                + (
                    rel_y
                    + (new_curve.box.y2 - new_curve.box.y)
                    + self.formular.y_offset
                )
                * scale,
            )

        # Set relocation transform instead of modifying original CTM
        translation_x = (
            new_x + self.formular.x_offset * scale - original_formula_x * scale
        )
        translation_y = (
            new_y + self.formular.y_offset * scale - original_formula_y * scale
        )

        # Create relocation transformation matrix
        from babeldoc.format.pdf.document_il.utils.matrix_helper import (
            create_translation_and_scale_matrix,
        )

        relocation_matrix = create_translation_and_scale_matrix(
            translation_x, translation_y, scale
        )
        new_curve.relocation_transform = list(relocation_matrix)

        return new_curve

    def _transform_form_for_relocation(
        self,
        form,
        original_formula_x: float,
        original_formula_y: float,
        new_x: float,
        new_y: float,
        scale: float,
    ):
        """Transform a form for formula relocation."""
        import copy

        new_form = copy.deepcopy(form)

        if new_form.box:
            # Calculate relative position to formula's original position (same as chars)
            rel_x = new_form.box.x - original_formula_x
            rel_y = new_form.box.y - original_formula_y

            # Apply same transformation as characters
            new_form.box = Box(
                x=new_x + (rel_x + self.formular.x_offset) * scale,
                y=new_y + (rel_y + self.formular.y_offset) * scale,
                x2=new_x
                + (rel_x + (new_form.box.x2 - new_form.box.x) + self.formular.x_offset)
                * scale,
                y2=new_y
                + (rel_y + (new_form.box.y2 - new_form.box.y) + self.formular.y_offset)
                * scale,
            )

        # Set relocation transform instead of modifying original matrices
        translation_x = (
            new_x + self.formular.x_offset * scale - original_formula_x * scale
        )
        translation_y = (
            new_y + self.formular.y_offset * scale - original_formula_y * scale
        )

        # Create relocation transformation matrix
        from babeldoc.format.pdf.document_il.utils.matrix_helper import (
            create_translation_and_scale_matrix,
        )

        relocation_matrix = create_translation_and_scale_matrix(
            translation_x, translation_y, scale
        )
        new_form.relocation_transform = list(relocation_matrix)

        return new_form

    def render(
        self,
    ) -> tuple[list[PdfCharacter], list[PdfCurve], list[PdfForm]]:
        """渲染排版单元为 PdfCharacter 列表

        Returns:
            PdfCharacter 列表
        """
        if self.can_passthrough:
            return self.passthrough()
        elif self.unicode:
            assert self.x is not None, (
                "x position must be set, should be set by `relocate`"
            )
            assert self.y is not None, (
                "y position must be set, should be set by `relocate`"
            )
            assert self.scale is not None, (
                "scale must be set, should be set by `relocate`"
            )
            x = self.x
            y = self.y
            # if self.original_font and self.font and hasattr(self.original_font, "descent") and hasattr(self.font, "descent_fontmap"):
            #     original_descent = self.original_font.descent
            #     new_descent = self.font.descent_fontmap
            #     y -= (original_descent - new_descent) * self.font_size / 1000

            # 计算字符宽度
            char_width = self.width

            new_char = PdfCharacter(
                pdf_character_id=self.font.has_glyph(ord(self.unicode)),
                char_unicode=self.unicode,
                box=Box(
                    x=x,  # 使用存储的位置
                    y=y,
                    x2=x + char_width,
                    y2=y + self.font_size,
                ),
                pdf_style=PdfStyle(
                    font_id=self.font_id,
                    font_size=self.font_size,
                    graphic_state=self.style.graphic_state,
                ),
                scale=self.scale,
                vertical=False,
                advance=char_width,
                xobj_id=self.xobj_id,
                debug_info=self.debug_info,
            )
            return [new_char], [], []
        elif self.formular:
            return self.passthrough()
        else:
            logger.error(f"Unknown typesetting unit. TypesettingUnit: {self}. ")
            return [], [], []


class Typesetting:
    stage_name = "Typesetting"

    def __init__(self, translation_config: TranslationConfig):
        self.font_mapper = FontMapper(translation_config)
        self.translation_config = translation_config
        self.lang_code = self.translation_config.lang_out.upper()
        self.is_cjk = (
            # Why zh-CN/zh-HK/zh-TW here but not zh-Hans and so on?
            # See https://funstory-ai.github.io/BabelDOC/supported_languages/
            ("ZH" in self.lang_code)  # C
            or ("JA" in self.lang_code)
            or ("JP" in self.lang_code)  # J
            or ("KR" in self.lang_code)  # K
            or ("CN" in self.lang_code)
            or ("HK" in self.lang_code)
            or ("TW" in self.lang_code)
        )

    def preprocess_document(self, document: il_version_1.Document, pbar):
        """预处理文档，获取每个段落的最优缩放因子，不执行实际排版"""
        all_scales: list[float] = []
        all_paragraphs: list[il_version_1.PdfParagraph] = []

        for page in document.page:
            pbar.advance()
            # 准备字体信息（复制自 render_page 的逻辑）
            fonts: dict[
                str | int,
                il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
            ] = {f.font_id: f for f in page.pdf_font if f.font_id}
            page_fonts = {f.font_id: f for f in page.pdf_font if f.font_id}
            for k, v in self.font_mapper.fontid2font.items():
                fonts[k] = v
            for xobj in page.pdf_xobject:
                if xobj.xobj_id is not None:
                    fonts[xobj.xobj_id] = page_fonts.copy()
                    for font in xobj.pdf_font:
                        if (
                            xobj.xobj_id in fonts
                            and isinstance(fonts[xobj.xobj_id], dict)
                            and font.font_id
                        ):
                            fonts[xobj.xobj_id][font.font_id] = font

            # 处理每个段落
            for paragraph in page.pdf_paragraph:
                all_paragraphs.append(paragraph)
                unit_count = 0
                try:
                    typesetting_units = self.create_typesetting_units(paragraph, fonts)
                    unit_count = len(typesetting_units)
                    for unit in typesetting_units:
                        if unit.formular:
                            unit_count += len(unit.formular.pdf_character) - 1

                    # 如果所有单元都可以直接传递，则 scale = 1.0
                    if all(unit.can_passthrough for unit in typesetting_units):
                        paragraph.optimal_scale = 1.0
                    else:
                        # 获取最优缩放因子
                        optimal_scale = self._get_optimal_scale(
                            paragraph, page, typesetting_units
                        )
                        paragraph.optimal_scale = optimal_scale
                except Exception as e:
                    # 如果预处理出错，默认使用 1.0 缩放因子
                    logger.warning(f"预处理段落时出错：{e}")
                    paragraph.optimal_scale = 1.0

                if paragraph.optimal_scale is not None:
                    all_scales.extend([paragraph.optimal_scale] * unit_count)

        # 获取缩放因子的众数
        if all_scales:
            try:
                modes = statistics.multimode(all_scales)
                mode_scale = min(modes)
            except statistics.StatisticsError:
                logger.warning(
                    "Could not find a mode for paragraph scales. Falling back to median."
                )
                mode_scale = statistics.median(all_scales)
            # 将所有大于众数的值修改为众数
            for paragraph in all_paragraphs:
                if (
                    paragraph.optimal_scale is not None
                    and paragraph.optimal_scale > mode_scale
                ):
                    paragraph.optimal_scale = mode_scale
        else:
            logger.error(
                "document_scales is empty, there seems no paragraph in this PDF"
            )

    def _find_optimal_scale_and_layout(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
        initial_scale: float = 1.0,
        use_english_line_break: bool = True,
        apply_layout: bool = False,
    ) -> tuple[float, list[TypesettingUnit] | None]:
        """查找最优缩放因子并可选择性地执行布局

        Args:
            paragraph: 段落对象
            page: 页面对象
            typesetting_units: 排版单元列表
            initial_scale: 初始缩放因子
            use_english_line_break: 是否使用英文换行规则
            apply_layout: 是否应用布局到 paragraph（True 时执行实际排版）

        Returns:
            tuple[float, list[TypesettingUnit] | None]: (最终缩放因子，排版后的单元列表或 None)
        """
        if not paragraph.box:
            return initial_scale, None

        box = paragraph.box
        scale = initial_scale
        line_skip = 1.50 if self.is_cjk else 1.4
        min_scale = 0.1
        expand_space_flag = 0
        final_typeset_units = None

        while scale >= min_scale:
            try:
                # 尝试布局排版单元
                typeset_units, all_units_fit = self._layout_typesetting_units(
                    typesetting_units,
                    box,
                    scale,
                    line_skip,
                    paragraph,
                    use_english_line_break,
                )

                # 如果所有单元都放得下
                if all_units_fit:
                    if apply_layout:
                        # 实际应用排版结果
                        paragraph.scale = scale
                        paragraph.pdf_paragraph_composition = []
                        for unit in typeset_units:
                            chars, curves, forms = unit.render()
                            for char in chars:
                                paragraph.pdf_paragraph_composition.append(
                                    PdfParagraphComposition(pdf_character=char),
                                )
                            for curve in curves:
                                page.pdf_curve.append(curve)
                            for form in forms:
                                page.pdf_form.append(form)
                        final_typeset_units = typeset_units
                    return scale, final_typeset_units
            except Exception:
                # 如果布局检查出错，继续尝试下一个缩放因子
                pass

            # 添加与原 retypeset 一致的逻辑检查
            if not hasattr(paragraph, "debug_id") or not paragraph.debug_id:
                return scale, final_typeset_units

            # 减小缩放因子
            if scale > 0.6:
                scale -= 0.05
            else:
                scale -= 0.1

            if scale < 0.7:
                # For structured paragraphs that should preserve line structure, don't expand the box
                # and don't scale down further - allow horizontal overflow
                if getattr(paragraph, 'preserve_line_structure', False):
                    logger.info(f"Not expanding box or scaling for structured paragraph {getattr(paragraph, 'debug_id', 'unknown')} - allowing overflow")
                    # Set scale back to 0.7 and apply layout to allow overflow
                    scale = 0.7
                    typeset_units, all_units_fit = self._layout_typesetting_units(
                        typesetting_units,
                        box,
                        scale,
                        line_skip,
                        paragraph,
                        use_english_line_break,
                    )
                    if apply_layout:
                        paragraph.scale = scale
                        paragraph.pdf_paragraph_composition = []
                        for unit in typeset_units:
                            chars, curves, forms = unit.render()
                            for char in chars:
                                paragraph.pdf_paragraph_composition.append(
                                    PdfParagraphComposition(pdf_character=char),
                                )
                            for curve in curves:
                                page.pdf_curve.append(curve)
                            for form in forms:
                                page.pdf_form.append(form)
                    return scale, typeset_units if apply_layout else None
                    continue

                space_expanded = False  # 标记是否成功扩展了空间

                if expand_space_flag == 0:
                    # 尝试向下扩展
                    try:
                        min_y = self.get_max_bottom_space(box, page) + 2
                        if min_y < box.y:
                            expanded_box = Box(x=box.x, y=min_y, x2=box.x2, y2=box.y2)
                            box = expanded_box
                            if apply_layout:
                                # 更新段落的边界框
                                paragraph.box = expanded_box
                            space_expanded = True
                    except Exception:
                        pass
                    expand_space_flag = 1

                    # 只有成功扩展空间时才 continue，否则继续减小 scale
                    if space_expanded:
                        continue

                elif expand_space_flag == 1:
                    # 尝试向右扩展
                    try:
                        max_x = self.get_max_right_space(box, page) - 5
                        if max_x > box.x2:
                            expanded_box = Box(x=box.x, y=box.y, x2=max_x, y2=box.y2)
                            box = expanded_box
                            if apply_layout:
                                # 更新段落的边界框
                                paragraph.box = expanded_box
                            space_expanded = True
                    except Exception:
                        pass
                    expand_space_flag = 2

                    # 只有成功扩展空间时才 continue，否则继续减小 scale
                    if space_expanded:
                        continue

                # 只有在扩展尝试阶段 (expand_space_flag < 2) 且扩展失败时才重置 scale
                # 当 expand_space_flag >= 2 时，说明已经尝试过所有扩展，应该继续正常的 scale 减小
                if expand_space_flag < 2:
                    # 如果无法扩展空间，重置 scale 并继续循环
                    scale = 1.0

        # 如果仍然放不下，尝试去除英文换行限制
        if use_english_line_break:
            return self._find_optimal_scale_and_layout(
                paragraph,
                page,
                typesetting_units,
                initial_scale,
                use_english_line_break=False,
                apply_layout=apply_layout,
            )

        # 最后返回最小缩放因子
        return min_scale, final_typeset_units

    def _get_optimal_scale(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
        use_english_line_break: bool = True,
    ) -> float:
        """获取段落的最优缩放因子，不执行实际排版"""
        scale, _ = self._find_optimal_scale_and_layout(
            paragraph,
            page,
            typesetting_units,
            1.0,
            use_english_line_break,
            apply_layout=False,
        )
        return scale

    def retypeset_with_precomputed_scale(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        typesetting_units: list[TypesettingUnit],
        precomputed_scale: float,
        use_english_line_break: bool = True,
    ):
        """使用预计算的缩放因子进行排版"""
        if not paragraph.box:
            return

        # For structured paragraphs, disable English line breaking and use uniform scaling
        is_structured = getattr(paragraph, 'preserve_line_structure', False)
        effective_use_english_line_break = use_english_line_break and not is_structured

        # For structured paragraphs, use fixed uniform scaling to ensure consistent appearance
        effective_precomputed_scale = precomputed_scale
        if is_structured:
            effective_precomputed_scale = 0.6  # Fixed uniform scaling for all structured content

        # 使用通用方法进行排版，传入预计算的缩放因子作为初始值
        self._find_optimal_scale_and_layout(
            paragraph,
            page,
            typesetting_units,
            effective_precomputed_scale,
            effective_use_english_line_break,
            apply_layout=True,
        )

    def typesetting_document(self, document: il_version_1.Document):
        # 原有的排版逻辑
        if self.translation_config.progress_monitor:
            with self.translation_config.progress_monitor.stage_start(
                self.stage_name,
                len(document.page) * 2,
            ) as pbar:
                # 预处理：获取所有段落的最优缩放因子
                self.preprocess_document(document, pbar)

                for page in document.page:
                    self.translation_config.raise_if_cancelled()
                    self.render_page(page)
                    pbar.advance()
        else:
            for page in document.page:
                self.translation_config.raise_if_cancelled()
                self.render_page(page)

    def render_page(self, page: il_version_1.Page):
        fonts: dict[
            str | int,
            il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
        ] = {f.font_id: f for f in page.pdf_font if f.font_id}
        page_fonts = {f.font_id: f for f in page.pdf_font if f.font_id}
        for k, v in self.font_mapper.fontid2font.items():
            fonts[k] = v
        for xobj in page.pdf_xobject:
            if xobj.xobj_id is not None:
                fonts[xobj.xobj_id] = page_fonts.copy()
                for font in xobj.pdf_font:
                    if font.font_id:
                        fonts[xobj.xobj_id][font.font_id] = font
        if (
            page.page_number == 0
            and self.translation_config.watermark_output_mode
            == WatermarkOutputMode.Watermarked
        ):
            self.add_watermark(page)
        try:
            para_index = index.Index()
            para_map = {}
            #
            valid_paras = [
                p
                for p in page.pdf_paragraph
                if p.box
                and all(c is not None for c in [p.box.x, p.box.y, p.box.x2, p.box.y2])
            ]

            for i, para in enumerate(valid_paras):
                para_map[i] = para
                para_index.insert(i, box_to_tuple(para.box))

            for i, p_upper in para_map.items():
                if not (p_upper.box and p_upper.box.y is not None):
                    continue

                # Calculate paragraph height and set required gap accordingly
                para_height = p_upper.box.y2 - p_upper.box.y
                required_gap = 0.5 if para_height < 36 else 3

                check_area = il_version_1.Box(
                    x=p_upper.box.x,
                    y=p_upper.box.y - required_gap,
                    x2=p_upper.box.x2,
                    y2=p_upper.box.y,
                )

                candidate_ids = list(para_index.intersection(box_to_tuple(check_area)))

                conflicting_paras = []
                for para_id in candidate_ids:
                    if para_id == i:
                        continue
                    p_lower = para_map[para_id]
                    if not (
                        p_lower.box
                        and p_upper.box
                        and p_lower.box.x2 < p_upper.box.x
                        or p_lower.box.x > p_upper.box.x2
                    ):
                        conflicting_paras.append(p_lower)

                if conflicting_paras:
                    max_y2 = max(
                        p.box.y2
                        for p in conflicting_paras
                        if p.box and p.box.y2 is not None
                    )

                    new_y = max_y2 + required_gap
                    if p_upper.box and new_y < p_upper.box.y2:
                        p_upper.box.y = new_y
        except Exception as e:
            logger.warning(
                f"Failed to adjust paragraph positions on page {page.page_number}: {e}"
            )
        # 开始实际的渲染过程
        for paragraph in page.pdf_paragraph:
            self.render_paragraph(paragraph, page, fonts)

    def add_watermark(self, page: il_version_1.Page):
        page_width = page.cropbox.box.x2 - page.cropbox.box.x
        page_height = page.cropbox.box.y2 - page.cropbox.box.y
        # Define text parts
        prefix = "This document is translated by lunartech.ai's open-source PDF translation library Babel ("
        link1 = "https://lunartech.ai"
        middle = "). This repository is currently under active construction, welcome to star and follow. Link to github "
        link2 = "https://github.com/LunarTechAI/babel"
        
        full_text = prefix + link1 + middle + link2
        
        if self.translation_config.debug:
            full_text += "\n 当前为 DEBUG 模式，将显示更多辅助信息。请注意，部分框的位置对应原文，但在译文中可能不正确。"

        # Try to find a bold font
        bold_font_id = "base"
        # Attempt to find a bold version of the base font or a fallback bold font
        # We know from assets.py that "SourceHanSansCN-Bold.ttf" is available and is likely the bold version of base (SourceHanSansCN-Regular.ttf)
        # However, to be safe and generic, we can try to ask font_mapper
        
        # Simple heuristic: try to find a font with "Bold" in its ID from the loaded fonts
        for fid in self.font_mapper.fontid2font:
            if "Bold" in fid and "SourceHanSans" in fid:
                bold_font_id = fid
                break
        
        if bold_font_id == "base":
             # Fallback to any bold font if specific one not found, or keep base
             for fid in self.font_mapper.fontid2font:
                if "Bold" in fid:
                    bold_font_id = fid
                    break

        style_normal = il_version_1.PdfStyle(
            font_id="base",
            font_size=6,
            graphic_state=il_version_1.GraphicState(),
        )
        
        style_bold = il_version_1.PdfStyle(
            font_id=bold_font_id,
            font_size=6,
            graphic_state=il_version_1.GraphicState(),
        )

        composition = []
        
        # Helper to add text
        def add_text(text, style):
            composition.append(
                il_version_1.PdfParagraphComposition(
                    pdf_same_style_unicode_characters=il_version_1.PdfSameStyleUnicodeCharacters(
                        unicode=text,
                        pdf_style=style,
                    ),
                )
            )

        add_text(prefix, style_normal)
        add_text(link1, style_bold)
        add_text(middle, style_normal)
        add_text(link2, style_bold)
        
        if self.translation_config.debug:
             add_text("\n 当前为 DEBUG 模式，将显示更多辅助信息。请注意，部分框的位置对应原文，但在译文中可能不正确。", style_normal)

        page.pdf_paragraph.append(
            il_version_1.PdfParagraph(
                first_line_indent=False,
                box=il_version_1.Box(
                    x=page.cropbox.box.x + page_width * 0.05,
                    y=page.cropbox.box.y,
                    x2=page.cropbox.box.x2,
                    y2=page.cropbox.box.y2 - page_height * 0.05,
                ),
                vertical=False,
                pdf_style=style_normal, # Default style for paragraph
                pdf_paragraph_composition=composition,
                unicode=full_text,
                xobj_id=-1,
            )
        )



    def render_paragraph(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        fonts: dict[
            str | int,
            il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
        ],
    ):
        # Check for preserve_line_structure flag for structured content
        preserve_flag = getattr(paragraph, 'preserve_line_structure', False)

        # Also check if the paragraph text starts with list markers (post-translation detection)
        is_list_item = False
        if paragraph.unicode:
            import re
            # Expanded regex for post-translation list detection
            if re.match(r'^\s*(\d+\.|\d+\)|\d+\-|[a-zA-Z]\.|[a-zA-Z]\)|[a-zA-Z]\-|\•|\-|\*|[ivxlcdm]+\.|[IVXLCDM]+\.|\(\d+\)|\([a-zA-Z]\))\s', paragraph.unicode.strip()):
                is_list_item = True
                preserve_flag = True

        print(f"DEBUG: Rendering paragraph {getattr(paragraph, 'debug_id', 'unknown')} at y={paragraph.box.y:.1f}, preserve_line_structure={preserve_flag}, is_list_item={is_list_item}")
        if preserve_flag:
            print(f"DEBUG: Using structured rendering for {getattr(paragraph, 'debug_id', 'unknown')}")
            self._render_structured_paragraph(paragraph, page, fonts)
            return
            logger.info(f"Rendering structured paragraph {getattr(paragraph, 'debug_id', 'unknown')} at y={paragraph.box.y:.1f}")
            self._render_structured_paragraph(paragraph, page, fonts)
            return
        
        typesetting_units = self.create_typesetting_units(paragraph, fonts)
        # 如果所有单元都可以直接传递，则直接传递
        if all(unit.can_passthrough for unit in typesetting_units):
            paragraph.scale = 1.0
            paragraph.pdf_paragraph_composition = self.create_passthrough_composition(
                typesetting_units,
            )
        else:
            # 使用预计算的缩放因子进行重排版
            precomputed_scale = (
                paragraph.optimal_scale if paragraph.optimal_scale is not None else 1.0
            )

            # 如果有单元无法直接传递，则进行重排版
            paragraph.pdf_paragraph_composition = []
            self.retypeset_with_precomputed_scale(
                paragraph, page, typesetting_units, precomputed_scale
            )

            # 重排版后，重新设置段落各字符的 render order
            self._update_paragraph_render_order(paragraph)


    def _render_structured_paragraph(
        self,
        paragraph: il_version_1.PdfParagraph,
        page: il_version_1.Page,
        fonts: dict[
            str | int,
            il_version_1.PdfFont | dict[str, il_version_1.PdfFont],
        ],
    ):
        """
        Render an atomic structured paragraph (a single line from an exploded section).
        
        This ensures that even if the translated text is sightly longer/shorter,
        it stays on the same vertical line.
        """
        if not paragraph.pdf_paragraph_composition:
            return
            
        # RESTORE FORMULA LOGIC:
        # If this paragraph has a backup original composition and looks like a formula,
        # restore it to preserve exact rendering (fonts, positions, symbols).
        if hasattr(paragraph, 'original_composition') and paragraph.original_composition:
            # Check if it looks like a formula
            orig_text = ""
            for comp in paragraph.original_composition:
                if comp.pdf_line:
                    orig_text += get_char_unicode_string(comp.pdf_line.pdf_character)
                elif comp.pdf_formula:
                    orig_text += get_char_unicode_string(comp.pdf_formula.pdf_character)
            
            # Improved heuristic: check if it contains a PdfFormula composition directly,
            # or if the text contains math indicators.
            has_formula_comp = any(comp.pdf_formula is not None for comp in paragraph.original_composition)
            is_formula = has_formula_comp or "=" in orig_text or (len(orig_text) < 15 and any(c.isdigit() for c in orig_text))
            
            if is_formula:
                paragraph.pdf_paragraph_composition = paragraph.original_composition
                paragraph.scale = 1.0
                self._update_paragraph_render_order(paragraph)
                return

        # For exploded paragraphs, we usually have a single composition which is a PdfLine
        # Use a high scale to avoid reflow unless absolutely necessary
        typesetting_units = self.create_typesetting_units(paragraph, fonts)
        
        # If the line is short enough, we can just pass it through at original position
        # but with new translated characters.
        # We'll use initial_scale=1.0 and allow horizontal fit but NO vertical flow outside box.
        paragraph.pdf_paragraph_composition = []
        
        # We use a very strict box that matches the original line height
        self.retypeset_with_precomputed_scale(
            paragraph, page, typesetting_units, 1.0
        )
        
        self._update_paragraph_render_order(paragraph)
        return

    def _create_typesetting_unit_from_char(
        self,
        char: il_version_1.PdfCharacter,
        fonts: dict,
    ) -> TypesettingUnit | None:
        """Create a TypesettingUnit from a single character."""
        if not char or not char.char_unicode:
            return None
        
        # Get font for this character
        font_id = char.font_id
        xobj_id = getattr(char, 'xobj_id', None)
        
        original_font = None
        if xobj_id is not None and xobj_id in fonts:
            xobj_fonts = fonts[xobj_id]
            if isinstance(xobj_fonts, dict) and font_id in xobj_fonts:
                original_font = xobj_fonts[font_id]
        elif font_id in fonts:
            original_font = fonts[font_id]
        
        return TypesettingUnit(
            char=char,
            unicode=char.char_unicode,
            font=self.font_mapper.base_font,
            original_font=original_font,
            font_size=char.font_size or 10.0,
            style=char.pdf_style,
            xobj_id=xobj_id,
            debug_info=getattr(char, 'debug_info', False),
        )

    def _get_width_before_next_break_point(
        self, typesetting_units: list[TypesettingUnit], scale: float
    ) -> float:
        if not typesetting_units:
            return 0
        if typesetting_units[0].can_break_line:
            return 0

        total_width = 0
        for unit in typesetting_units:
            if unit.can_break_line:
                return total_width * scale
            total_width += unit.width
        return total_width * scale

    def _layout_typesetting_units(
        self,
        typesetting_units: list[TypesettingUnit],
        box: Box,
        scale: float,
        line_skip: float,
        paragraph: il_version_1.PdfParagraph,
        use_english_line_break: bool = True,
    ) -> tuple[list[TypesettingUnit], bool]:
        """布局排版单元 (Refactored with Line Buffering)。

        Args:
            typesetting_units: 要布局的排版单元列表
            box: 布局边界框
            scale: 缩放因子

        Returns:
            tuple[list[TypesettingUnit], bool]: (已布局的排版单元列表，是否所有单元都放得下)
        """
        if not typesetting_units:
            return [], True

        # Constants
        FORMULA_PADDING = 3.0 * scale
        
        # Initialize
        current_y_top = box.y2  # Start from top of box
        
        # Buffers
        line_units = []
        current_line_width = 0.0
        if paragraph.first_line_indent:
             # Space width estimation
             font_sizes = [u.font_size for u in typesetting_units if u.font_size] or [10.0]
             avg_fs = sum(font_sizes)/len(font_sizes) if font_sizes else 10.0
             indent_width = (avg_fs * scale * 0.5) * 4
             current_line_width += indent_width
        
        typeset_units = []
        all_units_fit = True
        
        # Calculate space width for estimation
        base_font_size = 10.0
        if typesetting_units:
             sizes = [u.font_size for u in typesetting_units if u.font_size]
             if sizes:
                 base_font_size = sizes[0] # Approximation
        
        space_width = (
            self.font_mapper.base_font.char_lengths(" ", base_font_size * scale)[0]
        )

        
        # 1. Main Layout Loop (Buffering)
        idx = 0
        while idx < len(typesetting_units):
            unit = typesetting_units[idx]
            
            # Calculate unit dimensions
            unit_width = unit.width * scale
            unit_height = unit.height * scale
            
            # Add padding for formulas
            extra_width = 0.0
            if unit.formular:
                extra_width = FORMULA_PADDING * 2
            
            total_unit_width = unit_width + extra_width
            
            # For structured paragraphs, don't break lines - allow horizontal overflow
            is_structured = getattr(paragraph, 'preserve_line_structure', False)

            if not is_structured:
                # Check for English line break lookahead
                width_lookahead = 0.0
                if use_english_line_break:
                    width_lookahead = self._get_width_before_next_break_point(
                        typesetting_units[idx:], scale
                    )

                # Check mixed char spacing (simplified for lookahead)
                # strictly, we should check previous unit in buffer, but simplified here

                # 2. Determine Line Break
                # If adding this unit (plus lookahead) exceeds box width...
                # OR if logic enforces break
                should_break = False

                if not unit.is_hung_punctuation and (
                    (current_line_width + total_unit_width > (box.x2 - box.x))
                    or (use_english_line_break and current_line_width + total_unit_width + width_lookahead > (box.x2 - box.x))
                    or (unit.is_cannot_appear_in_line_end_punctuation and current_line_width + total_unit_width * 2 > (box.x2 - box.x))
                ):
                     should_break = True
            else:
                # For structured paragraphs, allow horizontal extension but constrain to page width
                # Get page width (assuming standard page width, could be made dynamic)
                page_width = 595.0  # Standard A4 width in points
                max_width = page_width - box.x  # From current position to page edge

                should_break = False

                if not unit.is_hung_punctuation and (
                    current_line_width + total_unit_width > max_width
                ):
                     should_break = True
            
            if should_break:
                # 3. Process the Buffered Line
                if not line_units and not paragraph.first_line_indent:
                     # Determine if single unit is too wide to fit at all? 
                     # For now, if line is empty, we force at least one unit unless it's huge
                     pass
                
                if not line_units:
                     # Force at least one unit to prevent infinite loop if a word is too long
                     line_units.append(unit)
                     current_line_width += total_unit_width
                     idx += 1
                
                # FLUSH LINE
                processed_units, next_y_top = self._flush_line(
                    line_units, box, current_y_top, scale, line_skip, FORMULA_PADDING, paragraph.first_line_indent and len(typeset_units)==0
                )
                
                typeset_units.extend(processed_units)
                
                # Check vertical overflow
                # Logic: next_y_top is the TOP of the NEXT line.
                # If the BOTTOM of the CURRENT line was below box.y, we have an issue.
                # But _flush_line calculates positions.
                # Let's verify processed_units positions.
                if processed_units:
                    lowest_y = min(u.box.y for u in processed_units)
                    if lowest_y < box.y:
                        all_units_fit = False
                
                # Reset for next line
                current_y_top = next_y_top
                line_units = []
                current_line_width = 0.0
                
                # Note: If we forced the unit into the line, idx was incremented.
                # If we broke BEFORE the unit, idx is valid for next line.
            else:
                # Add to buffer
                line_units.append(unit)
                current_line_width += total_unit_width
                
                # Add mixed-char spacing approximation
                if len(line_units) > 1:
                     last = line_units[-2]
                     curr = line_units[-1]
                     if last.is_cjk_char ^ curr.is_cjk_char and not last.is_space and not curr.is_space: # Simplified
                          current_line_width += space_width * 0.5
                
                idx += 1

        # 4. Flush Remaining Units (Last Line)
        if line_units:
             processed_units, next_y_top = self._flush_line(
                    line_units, box, current_y_top, scale, line_skip, FORMULA_PADDING, paragraph.first_line_indent and len(typeset_units)==0
                )
             typeset_units.extend(processed_units)
             if processed_units:
                lowest_y = min(u.box.y for u in processed_units)
                if lowest_y < box.y:
                    all_units_fit = False

        return typeset_units, all_units_fit

    def _flush_line(self, line_units, box, y_top, scale, line_skip, formula_padding, is_first_line):
        """Helper to position units in a single line and calculate next Y position."""
        if not line_units:
            return [], y_top

        # Calculate line stats
        heights = [u.height * scale for u in line_units if not u.is_space]
        if not heights:
             heights = [10.0 * scale] # Fallback
        
        max_height = max(heights)
        try:
            mode_height = statistics.mode(heights)
        except:
            mode_height = sum(heights)/len(heights)
            
        # Determine Baseline Y for this line
        # We align text to the bottom-left.
        # But symbols might go lower (descent).
        # Standard approach: y_bottom = y_top - max_height.
        # But we want consistent line spacing.
        
        # Let's say y_top is the ascender line of the previous line (or box top).
        # We want to place the current line such that its CONTENT fits below y_top.
        # current_y_bottom = y_top - max_height
        
        # HOWEVER, the surrounding logic expects strict line spacing.
        # Gap = max(mode_height * line_skip, max_height * 1.05)
        # We need to subtract this gap from the PREVIOUS BASELINE to get CURRENT BASELINE?
        # Or subtract height from y_top?
        
        # Let's use the line_skip logic to determine the separation from the PREVIOUS line.
        # But for the FIRST line, we just drop by max_height.
        
        if is_first_line:
             current_y_bottom = y_top - max_height
        else:
             # Calculate gap based on this line's content (to accommodate tall formulas) including the previous line?
             # No, standard leading is based on current font size.
             spacing = max(mode_height * line_skip, max_height * 1.05)
             # Wait, y_top passed here is the TOP of the *available space*? 
             # No, in the loop: current_y_top = next_y_top.
             # Ideally y_top is the Y-coordinate of the BASELINE of the previous line?
             # The original code maintained `current_y` as the baseline position.
             
             # If `y_top` is the previous baseline:
             current_y_bottom = y_top - spacing
        
        # Place units
        current_x = box.x
        
        # First line indent
        font_sizes = [u.font_size for u in line_units if u.font_size] or [10.0]
        base_fs = font_sizes[0] if font_sizes else 10.0
        space_w = self.font_mapper.base_font.char_lengths(" ", base_fs * scale)[0]
        
        if is_first_line:
             indent = (base_fs * scale * 0.5) * 4
             current_x += indent

        relocated_units = []
        last_unit = None
        
        for unit in line_units:
            # Formula Padding
            if unit.formular:
                current_x += formula_padding

            # Handle CJK spacing
            if (last_unit and last_unit.is_cjk_char ^ unit.is_cjk_char
                and not last_unit.mixed_character_blacklist
                and not unit.mixed_character_blacklist
                and not unit.is_space and not last_unit.is_space
                and last_unit.try_get_unicode() not in ["。", "！", "？", "；", "：", "，"]):
                 current_x += space_w * 0.5
            
            # Relocate
            # Note: unit.relocate(x, y, s) uses y as the BOTTOM-LEFT corner.
            # So passing current_y_bottom works.
            new_unit = unit.relocate(current_x, current_y_bottom, scale)
            relocated_units.append(new_unit)
            
            last_unit = new_unit
            current_x = new_unit.box.x2
            
            if unit.formular:
                current_x += formula_padding

        return relocated_units, current_y_bottom

    def create_typesetting_units(
        self,
        paragraph: il_version_1.PdfParagraph,
        fonts: dict[str, il_version_1.PdfFont],
    ) -> list[TypesettingUnit]:
        if not paragraph.pdf_paragraph_composition:
            return []
        result = []

        @cache
        def get_font(font_id: str, xobj_id: int | None):
            if xobj_id in fonts:
                font = fonts[xobj_id][font_id]
            else:
                font = fonts[font_id]
            return font

        for composition in paragraph.pdf_paragraph_composition:
            if composition is None:
                continue
            if composition.pdf_line:
                result.extend(
                    [
                        TypesettingUnit(char=char)
                        for char in composition.pdf_line.pdf_character
                    ],
                )
            elif composition.pdf_character:
                result.append(
                    TypesettingUnit(
                        char=composition.pdf_character,
                        debug_info=paragraph.debug_info,
                    ),
                )
            elif composition.pdf_same_style_characters:
                result.extend(
                    [
                        TypesettingUnit(char=char)
                        for char in composition.pdf_same_style_characters.pdf_character
                    ],
                )
            elif composition.pdf_same_style_unicode_characters:
                style = composition.pdf_same_style_unicode_characters.pdf_style
                if style is None:
                    logger.warning(
                        f"Style is None. "
                        f"Composition: {composition}. "
                        f"Paragraph: {paragraph}. ",
                    )
                    continue
                font_id = style.font_id
                if font_id is None:
                    logger.warning(
                        f"Font ID is None. "
                        f"Composition: {composition}. "
                        f"Paragraph: {paragraph}. ",
                    )
                    continue
                font = get_font(font_id, paragraph.xobj_id)
                if composition.pdf_same_style_unicode_characters.unicode:
                    result.extend(
                        [
                            TypesettingUnit(
                                unicode=char_unicode,
                                font=self.font_mapper.map(
                                    font,
                                    char_unicode,
                                ),
                                original_font=font,
                                font_size=style.font_size,
                                style=style,
                                xobj_id=paragraph.xobj_id,
                                debug_info=composition.pdf_same_style_unicode_characters.debug_info
                                or False,
                            )
                            for char_unicode in composition.pdf_same_style_unicode_characters.unicode
                            if char_unicode not in ("\n",)
                        ],
                    )
            elif composition.pdf_formula:
                result.extend([TypesettingUnit(formular=composition.pdf_formula)])
            else:
                logger.error(
                    f"Unknown composition type. "
                    f"Composition: {composition}. "
                    f"Paragraph: {paragraph}. ",
                )
                continue
        result = list(
            filter(
                lambda x: x.unicode is None or x.font is not None,
                result,
            ),
        )

        if any(x.width < 0 for x in result):
            logger.warning("有排版单元宽度小于 0，请检查字体映射是否正确。")
        return result

    def create_passthrough_composition(
        self,
        typesetting_units: list[TypesettingUnit],
    ) -> list[PdfParagraphComposition]:
        """从排版单元创建直接传递的段落组合。

        Args:
            typesetting_units: 排版单元列表

        Returns:
            段落组合列表
        """
        composition = []
        for unit in typesetting_units:
            if unit.formular:
                # 对于公式单元，直接创建包含完整公式的组合
                composition.append(PdfParagraphComposition(pdf_formula=unit.formular))
            else:
                # 对于字符单元，使用原有逻辑
                chars, curves, forms = unit.passthrough()
                composition.extend(
                    [PdfParagraphComposition(pdf_character=char) for char in chars],
                )
        return composition

    def get_max_right_space(self, current_box: Box, page) -> float:
        """获取段落右侧最大可用空间

        Args:
            current_box: 当前段落的边界框
            page: 当前页面

        Returns:
            可以扩展到的最大 x 坐标
        """
        # 获取页面的裁剪框作为初始最大限制
        max_x = page.cropbox.box.x2 * 0.9

        # 检查所有可能的阻挡元素
        for para in page.pdf_paragraph:
            if para.box == current_box or para.box is None:  # 跳过当前段落
                continue
            # 只考虑在当前段落右侧且有垂直重叠的元素
            if para.box.x > current_box.x and not (
                para.box.y >= current_box.y2 or para.box.y2 <= current_box.y
            ):
                max_x = min(max_x, para.box.x)
        for char in page.pdf_character:
            if char.box.x > current_box.x and not (
                char.box.y >= current_box.y2 or char.box.y2 <= current_box.y
            ):
                max_x = min(max_x, char.box.x)
        # 检查图形
        for figure in page.pdf_figure:
            if figure.box.x > current_box.x and not (
                figure.box.y >= current_box.y2 or figure.box.y2 <= current_box.y
            ):
                max_x = min(max_x, figure.box.x)

        return max_x

    def get_max_bottom_space(self, current_box: Box, page: il_version_1.Page) -> float:
        """获取段落下方最大可用空间

        Args:
            current_box: 当前段落的边界框
            page: 当前页面

        Returns:
            可以扩展到的最小 y 坐标
        """
        # 获取页面的裁剪框作为初始最小限制
        min_y = page.cropbox.box.y * 1.1

        # 检查所有可能的阻挡元素
        for para in page.pdf_paragraph:
            if para.box == current_box or para.box is None:  # 跳过当前段落
                continue
            # 只考虑在当前段落下方且有水平重叠的元素
            if para.box.y2 < current_box.y and not (
                para.box.x >= current_box.x2 or para.box.x2 <= current_box.x
            ):
                min_y = max(min_y, para.box.y2)
        for char in page.pdf_character:
            if char.box.y2 < current_box.y and not (
                char.box.x >= current_box.x2 or char.box.x2 <= current_box.x
            ):
                min_y = max(min_y, char.box.y2)
        # 检查图形
        for figure in page.pdf_figure:
            if figure.box.y2 < current_box.y and not (
                figure.box.x >= current_box.x2 or figure.box.x2 <= current_box.x
            ):
                min_y = max(min_y, figure.box.y2)

        return min_y

    def _update_paragraph_render_order(self, paragraph: il_version_1.PdfParagraph):
        """
        重新设置段落各字符的 render order
        主 render order 等于 paragraph 的 renderorder，sub render order 从 1 开始自增
        """
        if not hasattr(paragraph, "render_order") or paragraph.render_order is None:
            return

        main_render_order = paragraph.render_order
        sub_render_order = 1

        # 遍历段落的所有组成部分
        for composition in paragraph.pdf_paragraph_composition:
            # 检查单个字符
            if composition.pdf_character:
                char = composition.pdf_character
                char.render_order = main_render_order
                char.sub_render_order = sub_render_order
                sub_render_order += 1
