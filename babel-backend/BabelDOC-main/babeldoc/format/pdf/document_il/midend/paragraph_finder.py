import logging
import random
import re

import numpy as np

from babeldoc.babeldoc_exception.BabelDOCException import ExtractTextError
from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import Document
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PdfCharacter
from babeldoc.format.pdf.document_il import PdfLine
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il import PdfParagraphComposition
from babeldoc.format.pdf.document_il import PdfRectangle
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.formular_helper import (
    collect_page_formula_font_ids,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    HEIGHT_NOT_USFUL_CHAR_IN_CHAR,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import SPACE_REGEX
from babeldoc.format.pdf.document_il.utils.layout_helper import Layout
from babeldoc.format.pdf.document_il.utils.layout_helper import add_space_dummy_chars
from babeldoc.format.pdf.document_il.utils.layout_helper import build_layout_index
from babeldoc.format.pdf.document_il.utils.layout_helper import calculate_iou_for_boxes
from babeldoc.format.pdf.document_il.utils.layout_helper import get_char_unicode_string
from babeldoc.format.pdf.document_il.utils.layout_helper import get_character_layout
from babeldoc.format.pdf.document_il.utils.layout_helper import is_bullet_point
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    is_character_in_formula_layout,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import is_text_layout
from babeldoc.format.pdf.document_il.utils.paragraph_helper import is_cid_paragraph
from babeldoc.format.pdf.document_il.utils.style_helper import INDIGO
from babeldoc.format.pdf.document_il.utils.style_helper import WHITE
from babeldoc.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)

# Base58 alphabet (Bitcoin style, without numbers 0, O, I, l)
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def generate_base58_id(length: int = 5) -> str:
    """Generate a random base58 ID of specified length."""
    return "".join(random.choice(BASE58_ALPHABET) for _ in range(length))


class ParagraphFinder:
    stage_name = "Parse Paragraphs"

    # 定义项目符号的正则表达式模式

    def __init__(self, translation_config: TranslationConfig):
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)

    def _preprocess_formula_layouts(self, page: Page):
        """
        Identifies 'formula' layouts that do not significantly overlap with any text layouts
        and re-labels them as 'isolate_formula'.
        """
        # Use a simplified Layout object for is_text_layout check
        text_layouts = [
            layout
            for layout in page.page_layout
            if is_text_layout(Layout(layout.id, layout.class_name))
        ]
        formula_layouts = [
            layout for layout in page.page_layout if layout.class_name == "formula"
        ]

        if not text_layouts or not formula_layouts:
            return

        for formula_layout in formula_layouts:
            is_isolated = True
            for text_layout in text_layouts:
                iou = calculate_iou_for_boxes(formula_layout.box, text_layout.box)
                if iou >= 0.5:
                    is_isolated = False
                    break

            if is_isolated:
                formula_layout.class_name = "isolate_formula"

    def add_text_fill_background(self, page: Page):
        layout_map = {layout.id: layout for layout in page.page_layout}
        for paragraph in page.pdf_paragraph:
            layout_id = paragraph.layout_id
            if layout_id is None:
                continue
            layout = layout_map[layout_id]
            if paragraph.box is None:
                continue
            x1, y1, x2, y2 = (
                paragraph.box.x,
                paragraph.box.y,
                paragraph.box.x2,
                paragraph.box.y2,
            )
            layout_box = layout.box
            if layout_box.x < x1:
                x1 = layout_box.x
            if layout_box.y < y1:
                y1 = layout_box.y
            if layout_box.x2 > x2:
                x2 = layout_box.x2
            if layout_box.y2 > y2:
                y2 = layout_box.y2
            assert x2 > x1 and y2 > y1
            page.pdf_rectangle.append(
                PdfRectangle(
                    box=Box(x1, y1, x2, y2),
                    fill_background=True,
                    graphic_state=WHITE,
                    debug_info=False,
                    xobj_id=paragraph.xobj_id,
                )
            )

    def update_paragraph_data(self, paragraph: PdfParagraph, update_unicode=False):
        if not paragraph.pdf_paragraph_composition:
            return

        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_line:
                chars.extend(composition.pdf_line.pdf_character)
            elif composition.pdf_formula:
                chars.extend(composition.pdf_formula.pdf_character)
            elif composition.pdf_character:
                chars.append(composition.pdf_character)
            elif composition.pdf_same_style_unicode_characters:
                continue
            else:
                logger.error(
                    "Unexpected composition type"
                    " in PdfParagraphComposition. "
                    "This type only appears in the IL "
                    "after the translation is completed.",
                )
                continue

        if update_unicode and chars:
            paragraph.unicode = get_char_unicode_string(chars)
        if not chars:
            return
        # 更新边界框
        min_x = min(char.visual_bbox.box.x for char in chars)
        min_y = min(char.visual_bbox.box.y for char in chars)
        max_x = max(char.visual_bbox.box.x2 for char in chars)
        max_y = max(char.visual_bbox.box.y2 for char in chars)
        paragraph.box = Box(min_x, min_y, max_x, max_y)
        paragraph.vertical = chars[0].vertical
        paragraph.xobj_id = chars[0].xobj_id

        paragraph.first_line_indent = False
        if (
            paragraph.pdf_paragraph_composition
            and paragraph.pdf_paragraph_composition[0].pdf_line
            and paragraph.pdf_paragraph_composition[0]
            .pdf_line.pdf_character[0]
            .visual_bbox.box.x
            - paragraph.box.x
            > 1
        ):
            paragraph.first_line_indent = True

    def update_line_data(self, line: PdfLine):
        min_x = min(char.visual_bbox.box.x for char in line.pdf_character)
        min_y = min(char.visual_bbox.box.y for char in line.pdf_character)
        max_x = max(char.visual_bbox.box.x2 for char in line.pdf_character)
        max_y = max(char.visual_bbox.box.y2 for char in line.pdf_character)
        line.box = Box(min_x, min_y, max_x, max_y)

    def should_preserve_line_structure(self, paragraph: PdfParagraph) -> bool:
        """
        Detect if a paragraph has structured content that should preserve line boundaries.

        This identifies nomenclatures, indices, glossaries, and similar content where
        each line is a semantically independent unit (e.g., "c = absolute velocity").
        Also detects multi-line centered titles that should be treated as single units.
        Additionally detects footer content that should preserve fragment structure.

        Heuristics:
        1. Multiple short lines (average line length < 50 chars)
        2. Lines don't end with sentence-ending punctuation
        3. Lines have regular vertical spacing (list-like)
        4. Lines start with symbols/single characters (index-like)
        5. Lines are horizontally centered (centered titles)
        6. Footer layout labels (footer, page_footer, etc.)

        Returns:
            True if the paragraph should preserve line structure
        """
        if not paragraph.pdf_paragraph_composition:
            return False

        # Check for footer layout labels - these should preserve structure
        footer_labels = {"footer", "page_footer", "page_footer_hybrid"}
        if paragraph.layout_label in footer_labels:
            logger.info(f"Detected footer paragraph {getattr(paragraph, 'debug_id', 'unknown')} with layout {paragraph.layout_label} - preserving line structure")
            return True

        # Extract lines from composition
        lines = []
        for comp in paragraph.pdf_paragraph_composition:
            if comp.pdf_line and comp.pdf_line.pdf_character:
                line_text = get_char_unicode_string(comp.pdf_line.pdf_character)
                line_box = comp.pdf_line.box
                if line_text.strip():
                    lines.append((line_text, line_box))

        logger.info(f"Checking paragraph {getattr(paragraph, 'debug_id', 'unknown')} with {len(lines)} lines")
        if len(lines) >= 1:
            for i, (text, box) in enumerate(lines[:3]):  # Log first 3 lines
                logger.info(f"  Line {i}: '{text[:50]}...' at y={box.y:.1f}")

        # Check for single-line list items
        if len(lines) == 1:
            text, _ = lines[0]
            stripped = text.strip()
            # Expanded regex to handle more list formats including Roman numerals
            match = re.match(r'^\s*(\d+\.|\d+\)|\d+\-|[a-zA-Z]\.|[a-zA-Z]\)|[a-zA-Z]\-|\•|\-|\*|[ivxlcdm]+\.|[IVXLCDM]+\.|\(\d+\)|\([a-zA-Z]\))\s', stripped)
            if match:
                logger.warning(f"FOUND SINGLE-LINE LIST ITEM: '{stripped[:50]}...' - match: '{match.group(0)}' - preserving line structure")
                return True
            else:
                logger.info(f"NO MATCH for single-line: '{stripped[:30]}...'")

        # Need at least 2 lines to detect structure (lowered threshold)
        if len(lines) < 2:
            return False

        # Heuristic 0: Check for centered multi-line text (titles)
        if self._are_lines_centered(lines):
            logger.warning(f"DETECTED CENTERED MULTI-LINE TEXT - preserving as single unit")
            return True

        # Heuristic 1: Average line length - be more lenient for potential lists
        avg_line_length = sum(len(text) for text, _ in lines) / len(lines)

        # Heuristic 5: Detect numbered or bulleted lists FIRST (before length check)
        list_markers = 0
        for text, _ in lines:
            stripped = text.strip()
            # Expanded regex for multi-line detection
            if re.match(r'^\s*(\d+\.|\d+\)|\d+\-|[a-zA-Z]\.|[a-zA-Z]\)|[a-zA-Z]\-|\•|\-|\*|[ivxlcdm]+\.|[IVXLCDM]+\.|\(\d+\)|\([a-zA-Z]\))\s', stripped):
                list_markers += 1
                logger.info(f"Found list marker in multi-line paragraph: '{stripped[:30]}...'")

        # If we have multiple list markers, this is definitely a list regardless of length
        if list_markers >= 2:
            logger.warning(f"DETECTED MULTI-LINE STRUCTURED PARAGRAPH with {list_markers} list markers - will explode")
            return True

        # For single list markers or no markers, apply length restrictions
        if avg_line_length > 80:  # Increased threshold for list detection
            return False

        # Heuristic 2: Most lines don't end with sentence punctuation
        sentence_endings = sum(
            1 for text, _ in lines
            if text.rstrip() and text.rstrip()[-1] in '.!?;:'
        )
        if sentence_endings > len(lines) * 0.5:
            return False

        # Heuristic 3: Check for regular vertical spacing (list-like)
        if len(lines) >= 3:
            y_positions = [box.y for _, box in lines if box]
            if len(y_positions) >= 3:
                y_gaps = [
                    abs(y_positions[i+1] - y_positions[i])
                    for i in range(len(y_positions) - 1)
                ]
                if y_gaps:
                    avg_gap = sum(y_gaps) / len(y_gaps)
                    if avg_gap > 0:
                        # Check if gaps are uniform (std dev < 20% of mean)
                        variance = sum((g - avg_gap) ** 2 for g in y_gaps) / len(y_gaps)
                        std_dev = variance ** 0.5
                        if std_dev < avg_gap * 0.3:
                            # Uniform spacing detected - likely a list/index
                            return True

        # Heuristic 4: Lines start with short tokens (symbols, letters, numbers)
        short_starts = sum(
            1 for text, _ in lines
            if text.strip() and len(text.strip().split()[0]) <= 4
        )
        if short_starts > len(lines) * 0.6 and avg_line_length < 40:
            return True

        # Check for list-like structure with single marker
        elif list_markers >= 1 and len(lines) >= 2:
            # Check for list-like structure even with fewer markers
            non_empty_lines = [text for text, _ in lines if text.strip()]
            if len(non_empty_lines) >= 2:
                # Check if lines have consistent short starts (potential list items)
                short_starts = sum(1 for text in non_empty_lines if len(text.strip().split()[0]) <= 5)
                if short_starts >= len(non_empty_lines) * 0.6:
                    logger.warning(f"DETECTED LIST-LIKE STRUCTURE with {list_markers} markers and {short_starts}/{len(non_empty_lines)} short starts - will explode")
                    return True
        return False

    def _are_lines_centered(self, lines: list[tuple[str, Box]]) -> bool:
        """
        Check if multiple lines are horizontally centered.

        For centered text, the horizontal centers of all lines should be similar.
        """
        if len(lines) < 2:
            return False

        # Calculate horizontal centers for each line
        centers = []
        for text, box in lines:
            if box:
                center_x = (box.x + box.x2) / 2
                centers.append(center_x)

        if len(centers) < 2:
            return False

        # Check if all centers are within a tolerance of each other
        # Use the median as reference and check variance
        centers_sorted = sorted(centers)
        median_center = centers_sorted[len(centers) // 2]

        # Calculate the maximum deviation from median
        max_deviation = max(abs(c - median_center) for c in centers)

        # For centered text, allow small tolerance (e.g., 5 units)
        tolerance = 5.0

        is_centered = max_deviation <= tolerance

        if is_centered:
            logger.info(f"Lines are centered: centers={centers}, median={median_center:.1f}, max_deviation={max_deviation:.1f}")

        return is_centered


    def add_debug_info(self, page: Page):
        if not self.translation_config.debug:
            return
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    page.pdf_rectangle.append(
                        PdfRectangle(
                            box=line.box,
                            fill_background=False,
                            graphic_state=INDIGO,
                            debug_info=True,
                            line_width=0.2,
                        )
                    )

    def process(self, document):
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            len(document.page),
        ) as pbar:
            if not document.page:
                return
            for page in document.page:
                self.translation_config.raise_if_cancelled()
                self.process_page(page)
                pbar.advance()

            total_paragraph_count = 0
            for page in document.page:
                total_paragraph_count += len(page.pdf_paragraph)
            if total_paragraph_count == 0:
                raise ExtractTextError("The document contains no paragraphs.")

            if self.check_cid_paragraph(document):
                raise ExtractTextError("The document contains too many CID paragraphs.")

    def check_cid_paragraph(self, doc: Document):
        cid_para_count = 0
        para_total = 0
        for page in doc.page:
            para_total += len(page.pdf_paragraph)
            for para in page.pdf_paragraph:
                if is_cid_paragraph(para):
                    cid_para_count += 1
        return cid_para_count / para_total > 0.8

    def bbox_overlap(self, bbox1: Box, bbox2: Box) -> bool:
        return (
            bbox1.x < bbox2.x2
            and bbox1.x2 > bbox2.x
            and bbox1.y < bbox2.y2
            and bbox1.y2 > bbox2.y
        )

    def process_page(self, page: Page):
        layout_index, layout_map = build_layout_index(page)
        # 预处理公式布局的标签
        self._preprocess_formula_layouts(page)

        # 第一步：根据 layout 创建 paragraphs
        # 在这一步中，page.pdf_character 中的字符会被移除
        paragraphs = self._group_characters_into_paragraphs(
            page, layout_index, layout_map
        )
        page.pdf_paragraph = paragraphs

        page_level_formula_font_ids, xobj_specific_formula_font_ids = (
            collect_page_formula_font_ids(
                page, self.translation_config.formular_font_pattern
            )
        )

        # for para in paragraphs:
        #     if not para.debug_id:
        #         continue
        #     new_line = PdfLine(
        #         pdf_character=[x.pdf_character for x in para.pdf_paragraph_composition]
        #     )
        #     self.update_line_data(new_line)
        #     para.pdf_paragraph_composition = [
        #         PdfParagraphComposition(pdf_line=new_line)
        #     ]

        # 第二步：将段落内的字符拆分为行
        for paragraph in paragraphs:
            if (
                paragraph.xobj_id
                and paragraph.xobj_id in xobj_specific_formula_font_ids
            ):
                current_formula_font_ids = xobj_specific_formula_font_ids[
                    paragraph.xobj_id
                ]
            else:
                current_formula_font_ids = page_level_formula_font_ids
            self._split_paragraph_into_lines(paragraph, current_formula_font_ids)

        # 第三步：处理段落中的空格
        for paragraph in paragraphs:
            add_space_dummy_chars(paragraph)
            self.process_paragraph_spacing(paragraph)
            self.update_paragraph_data(paragraph)

        # 第四步：计算所有行宽度的中位数
        median_width = self.calculate_median_line_width(paragraphs)

        # 第五步：处理独立段落
        self.process_independent_paragraphs(paragraphs, median_width)

        # 新增后处理：合并带行号交替的正文段落（a 正文、b 行号、c 正文 -> 合并 a 与 c，保留 b）
        if getattr(self.translation_config, "merge_alternating_line_numbers", True):
            self.merge_alternating_line_number_paragraphs(paragraphs)

        for paragraph in paragraphs:
            self.update_paragraph_data(paragraph, update_unicode=True)

        if self.translation_config.ocr_workaround:
            self.add_text_fill_background(page)
            # since this is ocr file,
            # image characters are not needed
            page.pdf_character = []

        # Debug: check preserve_line_structure flags
        for para in page.pdf_paragraph:
            flag = getattr(para, 'preserve_line_structure', None)
            if flag:
                print(f"DEBUG: Paragraph {getattr(para, 'debug_id', 'unknown')} has preserve_line_structure={flag}")

        self.fix_overlapping_paragraphs(page)

        # 第六步：对每一行的字符进行排序
        # self._sort_characters_in_lines(page)

        self.add_debug_info(page)

        # 新阶段：设置段落的 renderorder 为所有组成部分中 renderorder 最小的
        self._set_paragraph_render_order(page)

        # Detect structured paragraphs and set preserve_line_structure flag
        self._detect_structured_paragraphs(page)

    def _set_paragraph_render_order(self, page: Page):
        """
        设置段落的 renderorder 为段落所有组成部分中 renderorder 最小的值
        """
        for paragraph in page.pdf_paragraph:
            min_render_order = 9999999999999999

            # 遍历段落的所有组成部分
            for composition in paragraph.pdf_paragraph_composition:
                # 检查 PdfLine 中的字符
                if composition.pdf_line:
                    for char in composition.pdf_line.pdf_character:
                        if (
                            hasattr(char, "render_order")
                            and char.render_order is not None
                        ):
                            min_render_order = min(min_render_order, char.render_order)

                # 检查单个字符
                elif composition.pdf_character:
                    char = composition.pdf_character
                    if hasattr(char, "render_order") and char.render_order is not None:
                        min_render_order = min(min_render_order, char.render_order)

                # 检查公式中的字符
                elif composition.pdf_formula:
                    for char in composition.pdf_formula.pdf_character:
                        if (
                            hasattr(char, "render_order")
                            and char.render_order is not None
                        ):
                            min_render_order = min(min_render_order, char.render_order)

            # 如果找到了有效的 renderorder，设置段落的 renderorder
            if min_render_order != 9999999999999999:
                paragraph.render_order = min_render_order

    def _detect_structured_paragraphs(self, page: Page):
        """
        Detect paragraphs with structured content and explode them into single-line paragraphs.
        
        By turning each line of a nomenclature/index into its own independent paragraph,
         we ensure that the system naturally preserves the vertical layout and 
        translates each entry as a distinct unit.
        """
        new_paragraphs = []
        exploded_count = 0
        
        for paragraph in page.pdf_paragraph:
            if self.should_preserve_line_structure(paragraph):
                # Check if this is a multi-line paragraph that needs exploding
                lines = []
                for comp in paragraph.pdf_paragraph_composition:
                    if comp.pdf_line and comp.pdf_line.pdf_character:
                        line_text = get_char_unicode_string(comp.pdf_line.pdf_character)
                        if line_text.strip():
                            lines.append((line_text, comp.pdf_line.box))

                if len(lines) > 1:
                    # Explode this multi-line paragraph!
                    logger.info(f"Exploding structured paragraph {paragraph.debug_id} into line-paragraphs.")

                    # Debug: log the lines being exploded
                    for comp in paragraph.pdf_paragraph_composition:
                        if comp.pdf_line:
                            line_text = get_char_unicode_string(comp.pdf_line.pdf_character)
                            logger.info(f"  Exploding line: '{line_text[:50]}...' at y={comp.pdf_line.box.y:.1f}")

                    # Each composition (line) becomes its own paragraph
                    for composition in paragraph.pdf_paragraph_composition:
                        if composition.pdf_line:
                            # Create a new atomic paragraph for this line
                            line_para = PdfParagraph(
                                box=composition.pdf_line.box,
                                pdf_style=paragraph.pdf_style,
                                pdf_paragraph_composition=[composition],
                                xobj_id=paragraph.xobj_id,
                                unicode=get_char_unicode_string(composition.pdf_line.pdf_character),
                                layout_label=paragraph.layout_label,
                                layout_id=paragraph.layout_id,
                                render_order=paragraph.render_order,
                                debug_id=f"{paragraph.debug_id}_L{composition.pdf_line.box.y:.1f}"
                            )
                            # Mark it to skip reflow during typesetting
                            line_para.preserve_line_structure = True
                            # Backup original composition for restoration if needed (e.g. formulas)
                            line_para.original_composition = [composition]
                            new_paragraphs.append(line_para)
                        else:
                            # Keep non-line compositions as-is
                            new_paragraphs.append(PdfParagraph(
                                box=paragraph.box, # Fallback to parent box
                                pdf_style=paragraph.pdf_style,
                                pdf_paragraph_composition=[composition],
                                xobj_id=paragraph.xobj_id,
                                unicode=paragraph.unicode,
                                layout_label=paragraph.layout_label,
                                render_order=paragraph.render_order,
                                debug_id=f"{paragraph.debug_id}_C"
                            ))
                    exploded_count += 1
                else:
                    # Single-line paragraph that should preserve structure
                    logger.info(f"Marking single-line structured paragraph {paragraph.debug_id} for preservation")
                    paragraph.preserve_line_structure = True
                    print(f"DEBUG: Set preserve_line_structure=True for {paragraph.debug_id}")
                    new_paragraphs.append(paragraph)
            else:
                new_paragraphs.append(paragraph)
        
        if exploded_count > 0:
            page.pdf_paragraph = new_paragraphs
            logger.info(f"Exploded {exploded_count} structured paragraphs on page {page.page_number}.")

    def is_isolated_formula(self, char: PdfCharacter):
        return char.char_unicode in (
            "(cid:122)",
            "(cid:123)",
            "(cid:124)",
            "(cid:125)",
        )

    def _paragraph_text_ascii(self, p: PdfParagraph) -> str:
        parts: list[str] = []
        for comp in p.pdf_paragraph_composition or []:
            if comp.pdf_line:
                for ch in comp.pdf_line.pdf_character or []:
                    if ch.char_unicode is not None:
                        parts.append(ch.char_unicode)
            elif comp.pdf_character and comp.pdf_character.char_unicode is not None:
                parts.append(comp.pdf_character.char_unicode)
        return "".join(parts)

    def _is_ascii_digit_or_space_paragraph(self, p: PdfParagraph) -> bool:
        text = self._paragraph_text_ascii(p)
        if not text:
            return True
        has_digit = False
        for c in text:
            if c.isdigit() and ord(c) < 128:
                has_digit = True
                continue
            if c.isspace():
                continue
            return False
        return True if has_digit or text.strip() == "" else False

    @staticmethod
    def _same_layout_and_xobj(a: PdfParagraph, c: PdfParagraph) -> bool:
        return (
            a.layout_id is not None
            and c.layout_id is not None
            and a.layout_id == c.layout_id
            and a.xobj_id is not None
            and c.xobj_id is not None
            and a.xobj_id == c.xobj_id
        )

    def merge_alternating_line_number_paragraphs(self, paragraphs: list[PdfParagraph]):
        # a 代表正文
        # l 代表行号
        if not paragraphs or len(paragraphs) < 3:
            return
        i = 0
        while i < len(paragraphs) - 2:
            a = paragraphs[i]
            # 吞掉一个或多个连续的行号段 l
            j = i + 1
            saw_l = False
            while j < len(paragraphs) and self._is_ascii_digit_or_space_paragraph(
                paragraphs[j]
            ):
                saw_l = True
                j += 1
            # 现在 j 指向候选的 c
            if saw_l and j < len(paragraphs):
                c = paragraphs[j]
                if self._same_layout_and_xobj(a, c):
                    a.pdf_paragraph_composition.extend(c.pdf_paragraph_composition)
                    self.update_paragraph_data(a)
                    del paragraphs[j]
                    # 不移动 i，继续尝试把更多正文接到 a，实现 a l+ a l+ a ... 链式合并
                    continue
            i += 1

    def _group_characters_into_paragraphs(
        self, page: Page, layout_index, layout_map
    ) -> list[PdfParagraph]:
        paragraphs: list[PdfParagraph] = []
        if page.pdf_paragraph:
            paragraphs.extend(page.pdf_paragraph)
            page.pdf_paragraph = []

        char_areas = [
            (char.visual_bbox.box.x2 - char.visual_bbox.box.x)
            * (char.visual_bbox.box.y2 - char.visual_bbox.box.y)
            for char in page.pdf_character
        ]
        median_char_area = 0.0
        if char_areas:
            char_areas.sort()
            mid = len(char_areas) // 2
            median_char_area = (
                char_areas[mid]
                if len(char_areas) % 2 == 1
                else (char_areas[mid - 1] + char_areas[mid]) / 2
            )

        current_paragraph: PdfParagraph | None = None
        current_layout: Layout | None = None
        skip_chars = []

        for char in page.pdf_character:
            char_layout = get_character_layout(char, layout_index, layout_map)
            # Check if character is in any formula layout and set formula_layout_id
            char.formula_layout_id = is_character_in_formula_layout(
                char, page, layout_index, layout_map
            )

            if not is_text_layout(char_layout):
                skip_chars.append(char)
                continue

            char_box = char.visual_bbox.box
            # char_pdf_box = char.box
            # if calculate_iou_for_boxes(char_box, char_pdf_box) < 0.2:
            #     char_box = char_pdf_box
            char_area = (char_box.x2 - char_box.x) * (char_box.y2 - char_box.y)
            is_small_char = char_area < median_char_area * 0.05

            # Get the last character added to the current paragraph for newline detection
            last_added_char = None
            if current_paragraph and current_paragraph.pdf_paragraph_composition:
                last_comp = current_paragraph.pdf_paragraph_composition[-1]
                if last_comp.pdf_character:
                    last_added_char = last_comp.pdf_character
                elif last_comp.pdf_line and last_comp.pdf_line.pdf_character:
                    last_added_char = last_comp.pdf_line.pdf_character[-1]

            is_newline = False
            if last_added_char:
                is_newline = Layout.is_newline(last_added_char, char)

            # Check for list indicators: Bullet points or common list starts like "(", "[", or "-"
            # We only treat them as list starts if they appear at the beginning of a line.
            is_list_start_char = (
                is_bullet_point(char)
                or char.char_unicode in ["(", "[", "-", "–", "—"]
            )

            is_new_paragraph = False
            if current_paragraph is None:
                is_new_paragraph = True
            elif (
                not (
                    is_small_char
                    and current_paragraph.pdf_paragraph_composition
                    and char_layout.id == current_layout.id
                )
                and char.char_unicode not in HEIGHT_NOT_USFUL_CHAR_IN_CHAR
            ):
                if (
                    (
                        char_layout.id != current_layout.id
                        and not SPACE_REGEX.match(char.char_unicode)
                    )
                    or (  # not same xobject
                        current_paragraph.pdf_paragraph_composition
                        and current_paragraph.pdf_paragraph_composition[
                            -1
                        ].pdf_character.xobj_id
                        != char.xobj_id
                    )
                    or (
                        # Break paragraph if it looks like a list item at the start of a line
                        is_list_start_char
                        and (is_newline or not current_paragraph.pdf_paragraph_composition)
                    )
                ):
                    is_new_paragraph = True

            if is_new_paragraph:
                current_layout = char_layout
                current_paragraph = PdfParagraph(
                    pdf_paragraph_composition=[],
                    layout_id=current_layout.id,
                    debug_id=generate_base58_id(),
                    layout_label=current_layout.name,
                )
                paragraphs.append(current_paragraph)

            current_paragraph.pdf_paragraph_composition.append(
                PdfParagraphComposition(pdf_character=char)
            )

        page.pdf_character = skip_chars
        for para in paragraphs:
            self.update_paragraph_data(para)
        return paragraphs

    def _merge_overlapping_clusters(
        self, lines: dict[int, list[PdfCharacter]], char_height_average: float
    ) -> dict[int, list[PdfCharacter]]:
        """
        Merge clusters that have significant y-axis overlap.
        If y_intersection / min_height > 0.5 or the distance between y-midlines is less than char_height_average, merge the two clusters.
        """
        if len(lines) <= 1:
            return lines

        # Calculate y-axis ranges for each cluster
        cluster_ranges = {}
        cluster_midlines = {}
        for label, chars in lines.items():
            y_values = [char.visual_bbox.box.y for char in chars] + [
                char.visual_bbox.box.y2 for char in chars
            ]
            y_min, y_max = min(y_values), max(y_values)
            cluster_ranges[label] = (y_min, y_max)
            cluster_midlines[label] = (y_min + y_max) / 2

        # Keep merging until no more merges are possible
        changed = True
        while changed:
            changed = False
            labels_to_check = list(lines.keys())

            for i in range(len(labels_to_check)):
                if not changed:  # Only continue if no merge happened in this iteration
                    for j in range(i + 1, len(labels_to_check)):
                        label1, label2 = labels_to_check[i], labels_to_check[j]

                        # Skip if either label has been merged away
                        if label1 not in lines or label2 not in lines:
                            continue

                        y1_min, y1_max = cluster_ranges[label1]
                        y2_min, y2_max = cluster_ranges[label2]

                        # Calculate intersection
                        intersection_start = max(y1_min, y2_min)
                        intersection_end = min(y1_max, y2_max)

                        # Calculate midline distance
                        midline_distance = abs(
                            cluster_midlines[label1] - cluster_midlines[label2]
                        )

                        should_merge = False
                        if (
                            intersection_end > intersection_start
                        ):  # There is intersection
                            intersection_height = intersection_end - intersection_start
                            height1 = y1_max - y1_min
                            height2 = y2_max - y2_min
                            min_height = min(height1, height2)

                            # Check if intersection ratio exceeds threshold
                            if (
                                min_height > 0
                                and intersection_height / min_height > 0.3
                            ):
                                should_merge = True

                        # Check if midline distance is less than char_height_average
                        if midline_distance < char_height_average:
                            should_merge = True

                        if should_merge:
                            # Merge label2 into label1
                            lines[label1].extend(lines[label2])
                            del lines[label2]

                            # Update cluster range and midline for the merged cluster
                            new_y_min = min(y1_min, y2_min)
                            new_y_max = max(y1_max, y2_max)
                            cluster_ranges[label1] = (new_y_min, new_y_max)
                            cluster_midlines[label1] = (new_y_min + new_y_max) / 2
                            del cluster_ranges[label2]
                            del cluster_midlines[label2]

                            changed = True
                            break

        return lines

    def _get_effective_y_bounds(self, char: PdfCharacter) -> tuple[float, float]:
        """
        Determines the effective vertical boundaries (y1, y2) for a character.

        It prioritizes the visual bounding box if its Intersection over Union (IoU)
        with the PDF bounding box is high (>= 0.5), otherwise, it falls back to the
        PDF bounding box. This helps use more accurate layout information when available.
        """
        visual_box = char.visual_bbox.box
        return visual_box.y, visual_box.y2
        pdf_box = char.box
        if calculate_iou_for_boxes(visual_box, pdf_box) >= 0.5:
            return visual_box.y, visual_box.y2
        return pdf_box.y, pdf_box.y2

    @staticmethod
    def _compute_collision_counts_histogram(
        y1_arr: np.ndarray,
        y2_arr: np.ndarray,
        para_y_min: float,
        para_y_max: float,
        step: float,
    ) -> np.ndarray:
        """Compute overlap counts at each scan line using a difference-array histogram.

        Args:
            y1_arr: 1-D array with lower y bounds of characters (inclusive).
            y2_arr: 1-D array with upper y bounds of characters (exclusive).
            para_y_min: Minimum y of the paragraph.
            para_y_max: Maximum y of the paragraph.
            step: Scan step size.

        Returns:
            1-D NumPy int32 array where index i corresponds to y = para_y_max - i × step.
        """
        # Number of scan positions
        m = int(np.ceil((para_y_max - para_y_min) / step))
        if m <= 0:
            return np.array([], dtype=np.int32)

        # Map character bounds to discrete indices (top inclusive, bottom exclusive)
        starts = np.floor((para_y_max - y2_arr) / step).astype(np.int32)
        ends = np.floor((para_y_max - y1_arr) / step).astype(np.int32) + 1
        # Clip ends to the valid range [0, m]
        np.clip(ends, 0, m, out=ends)

        hist = np.zeros(m + 1, dtype=np.int32)
        np.add.at(hist, starts, 1)
        np.add.at(hist, ends, -1)

        return np.cumsum(hist[:-1])

    def _split_footer_by_horizontal_gaps(self, chars: list[PdfCharacter]) -> list[list[PdfCharacter]]:
        """
        Split footer characters into lines based on horizontal gaps.
        Large gaps indicate separate text blocks that should be on different lines.
        """
        if not chars:
            return [chars]

        # Sort characters by x position
        sorted_chars = sorted(chars, key=lambda c: c.visual_bbox.box.x)

        # Calculate gaps between consecutive characters
        lines = []
        current_line = [sorted_chars[0]]

        for i in range(1, len(sorted_chars)):
            prev_char = sorted_chars[i-1]
            curr_char = sorted_chars[i]

            # Calculate the gap between the end of previous char and start of current char
            gap = curr_char.visual_bbox.box.x - prev_char.visual_bbox.box.x2

            # If gap is significant, start new line
            # Use a very small threshold to detect spacing
            gap_threshold = 1.0  # Fixed small threshold for any spacing

            logger.debug(f"Footer gap between '{prev_char.char_unicode}' and '{curr_char.char_unicode}': {gap:.1f}, threshold: {gap_threshold:.1f}")

            if gap > gap_threshold:
                # Large gap detected, start new line
                logger.debug(f"Large gap detected, starting new line at character {i}")
                lines.append(current_line)
                current_line = [curr_char]
            else:
                current_line.append(curr_char)

        # Add the last line
        if current_line:
            lines.append(current_line)

        # If no gaps were detected, try text-based splitting as fallback
        if len(lines) == 1:
            text_based_lines = self._split_footer_by_text_patterns(chars)
            if len(text_based_lines) > 1:
                logger.debug(f"Using text-based splitting for footer, found {len(text_based_lines)} lines")
                return text_based_lines

        return lines

    def _split_footer_by_text_patterns(self, chars: list[PdfCharacter]) -> list[list[PdfCharacter]]:
        """
        Split footer characters into lines based on text patterns.
        Look for patterns that indicate line breaks in footers.
        """
        if not chars:
            return [chars]

        # Get the full text
        full_text = get_char_unicode_string(chars)
        logger.debug(f"Footer text for pattern splitting: '{full_text}'")

        # Patterns that indicate potential line breaks in footers
        # Split before email addresses, phone numbers, addresses, etc.
        split_patterns = [
            r'(?=\bAdresse\b)',  # "Adresse" (Address in French)
            r'(?=\bE-mail\b)',   # "E-mail"
            r'(?=\bemail\b)',    # "email"
            r'(?=\bTel\b)',      # "Tel" (Telephone)
            r'(?=\bTél\b)',      # "Tél" (Telephone in French)
            r'(?=\bfax\b)',      # "fax"
            r'(?=\+\d)',         # Phone numbers starting with +
            r'(?=\(\d)',         # Phone numbers in parentheses
        ]

        split_positions = []
        for pattern in split_patterns:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                split_positions.append(match.start())

        if not split_positions:
            logger.debug("No text patterns found for splitting footer")
            return [chars]

        # Sort and deduplicate split positions
        split_positions = sorted(set(split_positions))

        # Convert text positions to character indices
        char_splits = []
        text_pos = 0
        for i, char in enumerate(chars):
            char_len = len(char.char_unicode)
            if any(pos >= text_pos and pos < text_pos + char_len for pos in split_positions):
                char_splits.append(i)
            text_pos += char_len

        if not char_splits:
            return [chars]

        # Split characters at the identified positions
        lines = []
        start = 0
        for split_pos in char_splits:
            if split_pos > start:
                lines.append(chars[start:split_pos])
                start = split_pos

        # Add remaining characters
        if start < len(chars):
            lines.append(chars[start:])

        logger.debug(f"Text-based splitting resulted in {len(lines)} lines")
        return lines

    def _split_paragraph_into_lines(
        self, paragraph: PdfParagraph, formula_font_ids: set[str]
    ):
        """
        Splits a paragraph into lines using a "line-threading" method.

        This method works by scanning vertically across the paragraph's bounding
        box and counting how many characters intersect with a horizontal line
        at each y-coordinate. The regions with a low number of intersections
        (less than 2) are identified as gaps between lines. The characters
        are then partitioned into lines based on these identified gaps.
        """
        if not paragraph.pdf_paragraph_composition:
            return

        # 1. Extract all characters and other compositions from the paragraph.
        all_chars: list[PdfCharacter] = []
        other_compositions: list[PdfParagraphComposition] = []
        for comp in paragraph.pdf_paragraph_composition:
            if comp.pdf_character:
                all_chars.append(comp.pdf_character)
            else:
                other_compositions.append(comp)

        if not all_chars:
            return

        # 2. Determine effective y-bounds for each character and the paragraph's total vertical range.
        char_y_bounds = [
            {"char": char, "y1": y1, "y2": y2}
            for char in all_chars
            for y1, y2 in [self._get_effective_y_bounds(char)]
        ]

        if not char_y_bounds:
            paragraph.pdf_paragraph_composition = other_compositions
            self.update_paragraph_data(paragraph)
            return

        para_y_min = min(b["y1"] for b in char_y_bounds)
        para_y_max = max(b["y2"] for b in char_y_bounds)

        # If the paragraph is vertically flat, treat it as a single line.
        # But for footer paragraphs, try to split based on horizontal gaps even if vertically flat
        footer_labels = {"footer", "page_footer", "page_footer_hybrid"}
        is_footer = paragraph.layout_label in footer_labels

        if (para_y_max - para_y_min) < 5 and not is_footer:  # Using a small threshold, but allow footer splitting
            # all_chars.sort(key=lambda c: c.visual_bbox.box.x)
            single_line_composition = self.create_line(all_chars)
            paragraph.pdf_paragraph_composition = [
                single_line_composition
            ] + other_compositions
            self.update_paragraph_data(paragraph)
            return
        elif is_footer and (para_y_max - para_y_min) < 5:
            # For footer paragraphs that are vertically flat, split based on horizontal gaps
            logger.info(f"Attempting to split footer paragraph {getattr(paragraph, 'debug_id', 'unknown')} with {len(all_chars)} characters")
            lines = self._split_footer_by_horizontal_gaps(all_chars)
            logger.info(f"Footer splitting resulted in {len(lines)} lines")
            if len(lines) > 1:
                paragraph.pdf_paragraph_composition = [
                    self.create_line(line_chars) for line_chars in lines
                ] + other_compositions
                self.update_paragraph_data(paragraph)
                logger.info(f"Successfully split footer into {len(lines)} lines")
                return
            else:
                # Fall back to single line if no gaps detected
                logger.info("No gaps detected in footer, keeping as single line")
                single_line_composition = self.create_line(all_chars)
                paragraph.pdf_paragraph_composition = [
                    single_line_composition
                ] + other_compositions
                self.update_paragraph_data(paragraph)
                return

        # 3. Perform "threading" scan to create a collision histogram.
        # Scan from top (max y) to bottom (min y) with a step of 0.5.
        scan_y_min = para_y_min
        scan_y_max = para_y_max
        step = 0.25

        y_coordinates = np.arange(scan_y_max, scan_y_min, -step)

        # Compute collision counts using NumPy histogram (O(m + n))
        y1_arr = np.array([b["y1"] for b in char_y_bounds], dtype=np.float32)
        y2_arr = np.array([b["y2"] for b in char_y_bounds], dtype=np.float32)
        collision_counts = self._compute_collision_counts_histogram(
            y1_arr,
            y2_arr,
            scan_y_min,
            scan_y_max,
            step,
        )

        # 4. Find gaps (regions with low collision count) from the histogram.
        gaps = []
        in_gap = False
        for i, count in enumerate(collision_counts):
            if count < 1 and not in_gap:
                in_gap = True
                gap_start_index = i
            elif count >= 1 and in_gap:
                in_gap = False
                gaps.append((gap_start_index, i - 1))
        if in_gap:
            gaps.append((gap_start_index, len(collision_counts) - 1))

        # If no significant gaps are found, treat it as a single line.
        if not gaps:
            # all_chars.sort(key=lambda c: c.visual_bbox.box.x)
            single_line_composition = self.create_line(all_chars)
            paragraph.pdf_paragraph_composition = [
                single_line_composition
            ] + other_compositions
            self.update_paragraph_data(paragraph)
            return

        # 5. Assign characters to lines based on the identified gaps.
        # Calculate separator y-coordinates from the midpoints of the gaps.
        separator_y_coords = sorted(
            [y_coordinates[start_idx] for start_idx, end_idx in gaps],
            reverse=True,
        )

        lines: list[list[PdfCharacter]] = [
            [] for _ in range(len(separator_y_coords) + 1)
        ]

        for b in char_y_bounds:
            char_y_center = (b["y1"] + b["y2"]) / 2
            line_idx = 0
            # Find which line bucket the character belongs to.
            for sep_y in separator_y_coords:
                if char_y_center > sep_y:
                    break
                line_idx += 1
            lines[line_idx].append(b["char"])

        # 6. Rebuild the paragraph's composition list from the new lines.
        new_line_compositions = []
        for line_chars in lines:
            if line_chars:
                # Sort characters within each line by x-coordinate (left-to-right).
                # line_chars.sort(key=lambda c: c.visual_bbox.box.x)
                new_line_compositions.append(self.create_line(line_chars))

        # The lines are already sorted vertically due to the scanning process.
        paragraph.pdf_paragraph_composition = new_line_compositions + other_compositions
        self.update_paragraph_data(paragraph)

    def process_paragraph_spacing(self, paragraph: PdfParagraph):
        if not paragraph.pdf_paragraph_composition:
            return

        # 处理行级别的空格
        processed_lines = []
        for composition in paragraph.pdf_paragraph_composition:
            if not composition.pdf_line:
                processed_lines.append(composition)
                continue

            line = composition.pdf_line
            if not "".join(
                x.char_unicode for x in line.pdf_character
            ).strip():  # 跳过完全空白的行
                continue

            # 处理行内字符的尾随空格
            processed_chars = []
            for char in line.pdf_character:
                if not char.char_unicode.isspace():
                    processed_chars = processed_chars + [char]
                elif processed_chars:  # 只有在有非空格字符后才考虑保留空格
                    processed_chars.append(char)

            # 移除尾随空格
            while processed_chars and processed_chars[-1].char_unicode.isspace():
                processed_chars.pop()

            if processed_chars:  # 如果行内还有字符
                line = self.create_line(processed_chars)
                processed_lines.append(line)

        paragraph.pdf_paragraph_composition = processed_lines
        self.update_paragraph_data(paragraph)

    def create_line(self, chars: list[PdfCharacter]) -> PdfParagraphComposition:
        assert chars

        line = PdfLine(pdf_character=chars)
        self.update_line_data(line)
        return PdfParagraphComposition(pdf_line=line)

    def calculate_median_line_width(self, paragraphs: list[PdfParagraph]) -> float:
        # 收集所有行的宽度
        line_widths = []
        for paragraph in paragraphs:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    line_widths.append(line.box.x2 - line.box.x)

        if not line_widths:
            return 0.0

        # 计算中位数
        line_widths.sort()
        mid = len(line_widths) // 2
        if len(line_widths) % 2 == 0:
            return (line_widths[mid - 1] + line_widths[mid]) / 2
        return line_widths[mid]

    def process_independent_paragraphs(
        self,
        paragraphs: list[PdfParagraph],
        median_width: float,
    ):
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]
            if len(paragraph.pdf_paragraph_composition) <= 1:  # 跳过只有一行的段落
                i += 1
                continue

            j = 1
            while j < len(paragraph.pdf_paragraph_composition):
                prev_composition = paragraph.pdf_paragraph_composition[j - 1]
                if not prev_composition.pdf_line:
                    j += 1
                    continue

                prev_line = prev_composition.pdf_line
                prev_width = prev_line.box.x2 - prev_line.box.x
                prev_text = "".join([c.char_unicode for c in prev_line.pdf_character])

                # 检查是否包含连续的点（至少 20 个）
                # 如果有至少连续 20 个点，则代表这是目录条目
                if re.search(r"\.{20,}", prev_text):
                    # 创建新的段落
                    new_paragraph = PdfParagraph(
                        box=Box(0, 0, 0, 0),  # 临时边界框
                        pdf_paragraph_composition=(
                            paragraph.pdf_paragraph_composition[j:]
                        ),
                        unicode="",
                        debug_id=generate_base58_id(),
                        layout_label=paragraph.layout_label,
                        layout_id=paragraph.layout_id,
                    )
                    # 更新原段落
                    paragraph.pdf_paragraph_composition = (
                        paragraph.pdf_paragraph_composition[:j]
                    )

                    # 更新两个段落的数据
                    self.update_paragraph_data(paragraph)
                    self.update_paragraph_data(new_paragraph)

                    # 在原段落后插入新段落
                    paragraphs.insert(i + 1, new_paragraph)
                    break

                # 如果前一行宽度小于中位数的一半，将当前行及后续行分割成新段落
                if (
                    self.translation_config.split_short_lines
                    and prev_width
                    < median_width * self.translation_config.short_line_split_factor
                ) or (
                    paragraph.pdf_paragraph_composition
                    and (current_line := paragraph.pdf_paragraph_composition[j])
                    and (line := current_line.pdf_line)
                    and (chars := line.pdf_character)
                    and (char := chars[0])
                    and is_bullet_point(char)
                ):
                    # 创建新的段落
                    new_paragraph = PdfParagraph(
                        box=Box(0, 0, 0, 0),  # 临时边界框
                        pdf_paragraph_composition=(
                            paragraph.pdf_paragraph_composition[j:]
                        ),
                        unicode="",
                        debug_id=generate_base58_id(),
                        layout_label=paragraph.layout_label,
                        layout_id=paragraph.layout_id,
                    )
                    # 更新原段落
                    paragraph.pdf_paragraph_composition = (
                        paragraph.pdf_paragraph_composition[:j]
                    )

                    # 更新两个段落的数据
                    self.update_paragraph_data(paragraph)
                    self.update_paragraph_data(new_paragraph)

                    # 在原段落后插入新段落
                    paragraphs.insert(i + 1, new_paragraph)
                    break
                j += 1
            i += 1

    @staticmethod
    def is_bbox_contain_in_vertical(bbox1: Box, bbox2: Box) -> bool:
        """Check if one bounding box is completely contained within the other."""
        # Check if bbox1 is contained in bbox2
        bbox1_in_bbox2 = bbox1.y >= bbox2.y and bbox1.y2 <= bbox2.y2
        # Check if bbox2 is contained in bbox1
        bbox2_in_bbox1 = bbox2.y >= bbox1.y and bbox2.y2 <= bbox1.y2
        return bbox1_in_bbox2 or bbox2_in_bbox1

    def fix_overlapping_paragraphs(self, page: Page):
        """
        Adjusts the bounding boxes of paragraphs on a page to resolve vertical overlaps.

        Iteratively checks pairs of paragraphs and adjusts their vertical boundaries
        (y and y2) if they overlap, aiming to place the boundary at the midpoint
        of the vertical overlap.
        """
        paragraphs = page.pdf_paragraph
        if not paragraphs or len(paragraphs) < 2:
            return

        max_iterations = len(paragraphs) * len(paragraphs)  # Safety break
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            overlap_found_in_pass = False

            for i in range(len(paragraphs)):
                for j in range(i + 1, len(paragraphs)):
                    para1 = paragraphs[i]
                    para2 = paragraphs[j]

                    if para1.box is None or para2.box is None:
                        continue

                    if para1.xobj_id != para2.xobj_id:
                        continue

                    # Check for overlap using the existing method
                    if self.bbox_overlap(para1.box, para2.box):
                        if self.is_bbox_contain_in_vertical(para1.box, para2.box):
                            continue
                        # Calculate vertical overlap details
                        overlap_y_start = max(para1.box.y, para2.box.y)
                        overlap_y_end = min(para1.box.y2, para2.box.y2)
                        overlap_height = overlap_y_end - overlap_y_start

                        # Calculate horizontal overlap details
                        overlap_x_start = max(para1.box.x, para2.box.x)
                        overlap_x_end = min(para1.box.x2, para2.box.x2)
                        overlap_width = overlap_x_end - overlap_x_start

                        # Ensure there's a real 2D overlap, focusing on vertical adjustment
                        if overlap_height > 1e-6 and overlap_width > 1e-6:
                            overlap_found_in_pass = True

                            # Determine which paragraph is visually higher
                            if para1.box.y2 > para2.box.y and para1.box.y < para2.box.y:
                                lower_para = para1
                                higher_para = para2
                            # Handle cases where y values are identical (or very close)
                            # Prefer the one with smaller y2 as the higher one, or break tie arbitrarily
                            elif para1.box.y2 < para2.box.y2:
                                lower_para = para1
                                higher_para = para2
                            else:
                                lower_para = para2
                                higher_para = para1

                            # Calculate the midpoint of the vertical overlap
                            mid_y = overlap_y_start + overlap_height / 2

                            # Adjust boxes, ensuring they remain valid (y2 > y)
                            if mid_y > higher_para.box.y and mid_y < lower_para.box.y2:
                                higher_para.box.y = mid_y + 1
                                lower_para.box.y2 = mid_y - 1
                            else:
                                # This might happen if one box is fully contained vertically
                                # within another, or due to floating point issues.
                                # Log a warning and skip adjustment for this pair in this iteration.
                                # A more complex strategy might be needed for full containment.
                                logger.warning(
                                    "Could not resolve overlap between paragraphs"
                                    f" {higher_para.debug_id} and {lower_para.debug_id}"
                                    " using simple midpoint strategy."
                                    f" Midpoint: {mid_y},"
                                    f" Higher Box: {higher_para.box},"
                                    f" Lower Box: {lower_para.box}"
                                )

            # If no overlaps were found and adjusted in this pass, we're done.
            if not overlap_found_in_pass:
                break

        if iterations == max_iterations:
            logger.warning(
                f"Maximum iterations ({max_iterations}) reached in"
                f" fix_overlapping_paragraphs for page {page.page_number}."
                " Some overlaps might remain."
            )

    def _sort_characters_in_lines(self, page: Page):
        """Sort characters in each line from left to right, top to bottom."""
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    line.pdf_character.sort(key=self._get_char_sort_key)

    def _get_char_sort_key(self, char: PdfCharacter):
        """Get sort key for character positioning (top to bottom, left to right)."""
        visual_box = char.visual_bbox.box
        pdf_box = char.box

        # Use visual box if IoU with bbox is >= 0.1, otherwise use bbox
        if calculate_iou_for_boxes(visual_box, pdf_box) >= 0.1:
            box = visual_box
        else:
            box = pdf_box

        # Sort by y coordinate first (top to bottom), then x coordinate (left to right)
        # Note: In PDF coordinate system, y increases upward, so we negate y for top-to-bottom sorting
        return (box.x, -box.y)
