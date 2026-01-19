import base64
import functools
import logging
import math
import re
import unicodedata
from io import BytesIO
from itertools import islice
from typing import Literal

import freetype
import pymupdf
import tiktoken

import babeldoc.pdfminer.pdfinterp
from babeldoc.format.pdf.babelpdf.base14 import get_base14_bbox
from babeldoc.format.pdf.babelpdf.cidfont import get_cidfont_bbox
from babeldoc.format.pdf.babelpdf.encoding import WinAnsiEncoding
from babeldoc.format.pdf.babelpdf.encoding import get_type1_encoding
from babeldoc.format.pdf.babelpdf.utils import guarded_bbox
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils import zstd_helper
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.matrix_helper import decompose_ctm
from babeldoc.format.pdf.document_il.utils.style_helper import BLACK
from babeldoc.format.pdf.document_il.utils.style_helper import YELLOW
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.pdfminer.layout import LTChar
from babeldoc.pdfminer.layout import LTFigure
from babeldoc.pdfminer.pdffont import PDFCIDFont
from babeldoc.pdfminer.pdffont import PDFFont

# from babeldoc.pdfminer.pdfpage import PDFPage as PDFMinerPDFPage
# from babeldoc.pdfminer.pdftypes import PDFObjRef as PDFMinerPDFObjRef
# from babeldoc.pdfminer.pdftypes import resolve1 as pdftypes_resolve1
from babeldoc.pdfminer.psparser import PSLiteral
from babeldoc.pdfminer.utils import apply_matrix_pt
from babeldoc.pdfminer.utils import get_bound
from babeldoc.pdfminer.utils import mult_matrix


def invert_matrix(
    ctm: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """
    Calculate the inverse of a 2D transformation matrix.
    Matrix format: (a, b, c, d, e, f) representing:
    [a c e]
    [b d f]
    [0 0 1]
    """
    a, b, c, d, e, f = ctm

    # Calculate determinant
    det = a * d - b * c

    if abs(det) < 1e-10:
        # Matrix is singular, return identity matrix
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    # Calculate inverse matrix elements
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    inv_e = (c * f - d * e) / det
    inv_f = (b * e - a * f) / det

    return (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f)


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


logger = logging.getLogger(__name__)

#
# def create_hook(func, hook):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         hook(*args, **kwargs)
#         return func(*args, **kwargs)
#
#     return wrapper
#
#
# def hook_pdfminer_pdf_page_init(*args):
#     attrs = args[3]
#     try:
#         while isinstance(attrs["MediaBox"], PDFMinerPDFObjRef):
#             attrs["MediaBox"] = pdftypes_resolve1(attrs["MediaBox"])
#     except Exception:
#         logger.exception(f"try to fix mediabox failed: {attrs}")
#
#
# PDFMinerPDFPage.__init__ = create_hook(
#     PDFMinerPDFPage.__init__, hook_pdfminer_pdf_page_init
# )


def indirect(obj):
    if isinstance(obj, tuple) and obj[0] == "xref":
        return int(obj[1].split(" ")[0])


def get_glyph_cbox(face, g):
    face.load_glyph(g, freetype.FT_LOAD_NO_SCALE)
    cbox = face.glyph.outline.get_bbox()
    return cbox.xMin, cbox.yMin, cbox.xMax, cbox.yMax


def get_char_cbox(face, idx):
    g = face.get_char_index(idx)
    return get_glyph_cbox(face, g)


def get_name_cbox(face, name):
    if name:
        if isinstance(name, str):
            name = name.encode("utf-8")
        g = face.get_name_index(name)
        return get_glyph_cbox(face, g)
    return (0, 0, 0, 0)


def font_encoding_lookup(doc, idx, key):
    obj = doc.xref_get_key(idx, key)
    if obj[0] == "name":
        enc_name = obj[1][1:]
        if enc_vector := get_type1_encoding(enc_name):
            return enc_name, enc_vector


def parse_font_encoding(doc, idx):
    if encoding := font_encoding_lookup(doc, idx, "Encoding/BaseEncoding"):
        return encoding
    if encoding := font_encoding_lookup(doc, idx, "Encoding"):
        return encoding
    return ("Custom", get_type1_encoding("StandardEncoding"))


def get_truetype_ansi_bbox_list(face):
    scale = 1000 / face.units_per_EM
    bbox_list = [get_char_cbox(face, code) for code in WinAnsiEncoding]
    bbox_list = [[v * scale for v in bbox] for bbox in bbox_list]
    return bbox_list


def collect_face_cmap(face):
    umap = []  # unicode maps
    lmap = []  # legacy maps
    for cmap in face.charmaps:
        if cmap.encoding_name == "FT_ENCODING_UNICODE":
            umap.append(cmap)
        else:
            lmap.append(cmap)
    return umap, lmap


def get_truetype_custom_bbox_list(face):
    umap, lmap = collect_face_cmap(face)
    if umap:
        face.set_charmap(umap[0])
    elif lmap:
        face.set_charmap(lmap[0])
    else:
        return []
    scale = 1000 / face.units_per_EM
    bbox_list = [get_char_cbox(face, code) for code in range(256)]
    bbox_list = [[v * scale for v in bbox] for bbox in bbox_list]
    return bbox_list


def parse_font_file(doc, idx, encoding, differences):
    bbox_list = []
    data = doc.xref_stream(idx)
    face = freetype.Face(BytesIO(data))
    if face.get_format() == b"TrueType":
        if encoding[0] == "WinAnsiEncoding":
            return get_truetype_ansi_bbox_list(face)
        elif encoding[0] == "Custom":
            return get_truetype_custom_bbox_list(face)
    glyph_name_set = set()
    for x in range(0, face.num_glyphs):
        glyph_name_set.add(face.get_glyph_name(x).decode("U8"))
    scale = 1000 / face.units_per_EM
    enc_name, enc_vector = encoding
    _, lmap = collect_face_cmap(face)
    abbr = enc_name.removesuffix("Encoding")
    if lmap and abbr in ["Custom", "MacRoman", "Standard", "WinAnsi", "MacExpert"]:
        face.set_charmap(lmap[0])
    for i, x in enumerate(enc_vector):
        if x in glyph_name_set:
            v = get_name_cbox(face, x.encode("U8"))
        else:
            v = get_char_cbox(face, i)
        bbox_list.append(v)
    if differences:
        for code, name in differences:
            bbox_list[code] = get_name_cbox(face, name.encode("U8"))
    norm_bbox_list = [[v * scale for v in box] for box in bbox_list]
    return norm_bbox_list


def parse_encoding(obj_str):
    delta = []
    current = 0
    for x in re.finditer(
        r"(?P<p>[\[\]])|(?P<c>\d+)|(?P<n>/[^\s/\[\]()<>]+)|(?P<s>.)", obj_str
    ):
        key = x.lastgroup
        val = x.group()
        if key == "c":
            current = int(val)
        if key == "n":
            delta.append((current, val[1:]))
            current += 1
    return delta


def parse_mapping(text):
    mapping = []
    for x in re.finditer(r"<(?P<num>[a-fA-F0-9]+)>", text):
        mapping.append(x.group("num"))
    return mapping


def update_cmap_pair(cmap, data):
    for start_str, stop_str, value_str in batched(data, 3):
        start = int(start_str, 16)
        stop = int(stop_str, 16)
        try:
            value = base64.b16decode(value_str, True).decode("UTF-16-BE")
            for code in range(start, stop + 1):
                cmap[code] = value
        except Exception:
            pass  # to skip surrogate pairs (D800-DFFF)


def update_cmap_code(cmap, data):
    for code_str, value_str in batched(data, 2):
        code = int(code_str, 16)
        try:
            value = base64.b16decode(value_str, True).decode("UTF-16-BE")
            cmap[code] = value
        except Exception:
            pass  # to skip surrogate pairs (D800-DFFF)


def parse_cmap(cmap_str):
    cmap = {}
    for x in re.finditer(
        r"\s+beginbfrange\s*(?P<r>(<[0-9a-fA-F]+>\s*)+)endbfrange\s+", cmap_str
    ):
        update_cmap_pair(cmap, parse_mapping(x.group("r")))
    for x in re.finditer(
        r"\s+beginbfchar\s*(?P<c>(<[0-9a-fA-F]+>\s*)+)endbfchar", cmap_str
    ):
        update_cmap_code(cmap, parse_mapping(x.group("c")))
    return cmap


def get_code(cmap, c):
    for k, v in cmap.items():
        if v == c:
            return k
    return -1


def get_bbox(bbox, size, c, x, y):
    x_min, y_min, x_max, y_max = bbox[c]
    factor = 1 / 1000 * size
    x_min = x_min * factor
    y_min = -y_min * factor
    x_max = x_max * factor
    y_max = -y_max * factor
    ll = (x + x_min, y + y_min)
    lr = (x + x_max, y + y_min)
    ul = (x + x_min, y + y_max)
    ur = (x + x_max, y + y_max)
    return pymupdf.Quad(ll, lr, ul, ur)


# 常见 Unicode 空格字符的代码点
unicode_spaces = [
    "\u0020",  # 半角空格
    "\u00a0",  # 不间断空格
    "\u1680",  # Ogham 空格标记
    "\u2000",  # En Quad
    "\u2001",  # Em Quad
    "\u2002",  # En Space
    "\u2003",  # Em Space
    "\u2004",  # 三分之一 Em 空格
    "\u2005",  # 四分之一 Em 空格
    "\u2006",  # 六分之一 Em 空格
    "\u2007",  # 数样间距
    "\u2008",  # 行首前导空格
    "\u2009",  # 瘦弱空格
    "\u200a",  # hair space
    "\u202f",  # 窄不间断空格
    "\u205f",  # 数学中等空格
    "\u3000",  # 全角空格
    "\u200b",  # 零宽度空格
    "\u2060",  # 零宽度非断空格
    "\t",  # 水平制表符
]

# 构建正则表达式
pattern = "^[" + "".join(unicode_spaces) + "]+$"

# 编译正则
space_regex = re.compile(pattern)


def get_rotation_angle(matrix):
    """
    根据 PDF 的字符矩阵计算旋转角度（单位：度）
    matrix: tuple/list, 格式 (a, b, c, d, e, f)
    """
    a, b, c, d, e, f = matrix
    # 旋转角度：arctan2(b, a)
    angle_rad = math.atan2(b, a)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


class ILCreater:
    stage_name = "Parse PDF and Create Intermediate Representation"

    def __init__(self, translation_config: TranslationConfig):
        self.progress = None
        self.current_page: il_version_1.Page = None
        self.mupdf: pymupdf.Document = None
        self.model = translation_config.doc_layout_model
        self.docs = il_version_1.Document(page=[])
        self.stroking_color_space_name = None
        self.non_stroking_color_space_name = None
        self.passthrough_per_char_instruction: list[tuple[str, str]] = []
        self.translation_config = translation_config
        self.passthrough_per_char_instruction_stack: list[list[tuple[str, str]]] = []
        self.xobj_id = 0
        self.xobj_inc = 0
        self.xobj_map: dict[int, il_version_1.PdfXobject] = {}
        self.xobj_stack = []
        self.current_page_font_name_id_map = {}
        self.current_page_font_char_bounding_box_map = {}
        self.current_available_fonts = {}
        self.mupdf_font_map: dict[int, pymupdf.Font] = {}
        self.graphic_state_pool = {}
        self.enable_graphic_element_process = (
            translation_config.enable_graphic_element_process
        )
        self.render_order = 0
        self.current_clip_paths: list[tuple] = []
        self.clip_paths_stack: list[list[tuple]] = []
        # For valid character collection
        self.font_mapper = FontMapper(translation_config)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self._page_valid_chars_buffer: list[str] | None = None

    def transform_clip_path(
        self,
        clip_path,
        source_ctm: tuple[float, float, float, float, float, float],
        target_ctm: tuple[float, float, float, float, float, float],
    ):
        """Transform clip path coordinates from source CTM to target CTM."""
        if source_ctm == target_ctm:
            return clip_path

        # Calculate transformation matrix: inverse(target_ctm) * source_ctm
        inv_target_ctm = invert_matrix(target_ctm)
        transform_matrix = mult_matrix(source_ctm, inv_target_ctm)

        transformed_path = []
        for path_element in clip_path:
            if len(path_element) == 1:
                # Path operation without coordinates (e.g., 'h' for close path)
                transformed_path.append(path_element)
            else:
                # Path operation with coordinates
                op = path_element[0]
                coords = path_element[1:]
                transformed_coords = []

                # Transform coordinate pairs
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        x, y = coords[i], coords[i + 1]
                        transformed_point = apply_matrix_pt(transform_matrix, (x, y))
                        transformed_coords.extend(transformed_point)
                    else:
                        # Handle odd number of coordinates (shouldn't happen in well-formed paths)
                        transformed_coords.append(coords[i])

                transformed_path.append([op] + transformed_coords)

        return transformed_path

    def get_render_order_and_increase(self):
        self.render_order += 1
        return self.render_order

    def get_render_order(self):
        return self.render_order

    def on_finish(self):
        self.progress.__exit__(None, None, None)

    def is_graphic_operation(self, operator: str):
        if not self.enable_graphic_element_process:
            return False

        return re.match(
            "^(m|l|c|v|y|re|h|S|s|f|f*|F|B|B*|b|b*|n|Do)$",
            operator,
        )

    def is_passthrough_per_char_operation(self, operator: str):
        return re.match(
            "^(sc|SC|sh|scn|SCN|g|G|rg|RG|k|K|cs|CS|gs|ri|w|J|j|M|i)$",
            operator,
        )

    def can_remove_old_passthrough_per_char_instruction(self, operator: str):
        return re.match(
            "^(sc|SC|sh|scn|SCN|g|G|rg|RG|k|K|cs|CS|ri|w|J|j|M|i|d)$",
            operator,
        )

    def on_line_dash(self, dash, phase):
        dash_str = f"[{' '.join(f'{arg}' for arg in dash)}]"
        self.on_passthrough_per_char("d", [dash_str, str(phase)])

    def on_passthrough_per_char(self, operator: str, args: list[str]):
        if not self.is_passthrough_per_char_operation(operator) and operator not in (
            "W n",
            "W* n",
            "d",
            "W",
            "W*",
        ):
            logger.error("Unknown passthrough_per_char operation: %s", operator)
            return
        # logger.debug("xobj_id: %d, on_passthrough_per_char: %s ( %s )", self.xobj_id, operator, args)
        args = [self.parse_arg(arg) for arg in args]
        if self.can_remove_old_passthrough_per_char_instruction(operator):
            for _i, value in enumerate(self.passthrough_per_char_instruction.copy()):
                op, arg = value
                if op == operator:
                    self.passthrough_per_char_instruction.remove(value)
                    break
        self.passthrough_per_char_instruction.append((operator, " ".join(args)))
        pass

    def remove_latest_passthrough_per_char_instruction(self):
        if self.passthrough_per_char_instruction:
            self.passthrough_per_char_instruction.pop()

    def parse_arg(self, arg: str):
        if isinstance(arg, PSLiteral):
            return f"/{arg.name}"
        if not isinstance(arg, str):
            return str(arg)
        return arg

    def pop_passthrough_per_char_instruction(self):
        if self.passthrough_per_char_instruction_stack:
            self.passthrough_per_char_instruction = (
                self.passthrough_per_char_instruction_stack.pop()
            )
        else:
            self.passthrough_per_char_instruction = []
            logging.error(
                "pop_passthrough_per_char_instruction error on page: %s",
                self.current_page.page_number,
            )

        if self.clip_paths_stack:
            self.current_clip_paths = self.clip_paths_stack.pop()
        else:
            self.current_clip_paths = []

    def push_passthrough_per_char_instruction(self):
        self.passthrough_per_char_instruction_stack.append(
            self.passthrough_per_char_instruction.copy(),
        )
        self.clip_paths_stack.append(self.current_clip_paths.copy())

    # pdf32000 page 171
    def on_stroking_color_space(self, color_space_name):
        self.stroking_color_space_name = color_space_name

    def on_non_stroking_color_space(self, color_space_name):
        self.non_stroking_color_space_name = color_space_name

    def on_new_stream(self):
        self.stroking_color_space_name = None
        self.non_stroking_color_space_name = None
        self.passthrough_per_char_instruction = []
        self.current_clip_paths = []

    def push_xobj(self):
        self.xobj_stack.append(
            (
                self.xobj_id,
                self.current_clip_paths.copy(),
                self.current_available_fonts.copy(),
            ),
        )
        self.current_clip_paths = []

    def pop_xobj(self):
        (self.xobj_id, self.current_clip_paths, self.current_available_fonts) = (
            self.xobj_stack.pop()
        )

    def on_xobj_begin(self, bbox, xref_id):
        logger.debug(f"on_xobj_begin: {bbox} @ {xref_id}")
        self.push_passthrough_per_char_instruction()
        self.push_xobj()
        self.xobj_inc += 1
        self.xobj_id = self.xobj_inc
        xobject = il_version_1.PdfXobject(
            box=il_version_1.Box(
                x=float(bbox[0]),
                y=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
            ),
            xobj_id=self.xobj_id,
            xref_id=xref_id,
            pdf_font=[],
        )
        self.current_page.pdf_xobject.append(xobject)
        self.xobj_map[self.xobj_id] = xobject
        xobject.pdf_font.extend(self.current_available_fonts.values())
        return self.xobj_id

    def on_xobj_end(self, xobj_id, base_op):
        self.pop_passthrough_per_char_instruction()
        self.pop_xobj()
        xobj = self.xobj_map[xobj_id]
        base_op = zstd_helper.zstd_compress(base_op)
        xobj.base_operations = il_version_1.BaseOperations(value=base_op)
        self.xobj_inc += 1

    def on_page_start(self):
        self.current_page = il_version_1.Page(
            pdf_font=[],
            pdf_character=[],
            page_layout=[],
            pdf_curve=[],
            pdf_form=[],
            pdf_image=[],
            pdf_drawing=[],
            pdf_text_logo=[],
            # currently don't support UserUnit page parameter
            # pdf32000 page 79
            unit="point",
        )
        self.current_page_font_name_id_map = {}
        self.current_page_font_char_bounding_box_map = {}
        self.passthrough_per_char_instruction_stack = []
        self.xobj_stack = []
        self.non_stroking_color_space_name = None
        self.stroking_color_space_name = None
        self.current_clip_paths = []
        self.clip_paths_stack = []
        self.docs.page.append(self.current_page)
        # Prepare per-page buffer for valid characters on translated pages
        self._page_valid_chars_buffer = []

    def on_page_end(self):
        # Extract images and drawings from the current page
        self.extract_page_images_and_drawings()
        # Post-process characters to detect text logos that span multiple characters
        self._post_process_text_logos()
        # Accumulate this page's valid characters and tokens into shared context
        try:
            if (
                self._page_valid_chars_buffer is not None
                and len(self._page_valid_chars_buffer) > 0
            ):
                page_text = "".join(self._page_valid_chars_buffer)
                char_count = len(page_text)
                try:
                    token_count = len(
                        self.tokenizer.encode(page_text, disallowed_special=())
                    )
                except Exception as e:
                    logger.warning("Failed to compute token count for page: %s", e)
                    token_count = 0
                self.translation_config.shared_context_cross_split_part.add_valid_counts(
                    char_count, token_count
                )
        except Exception as e:
            logger.warning("Failed to accumulate page valid stats: %s", e)
        finally:
            self._page_valid_chars_buffer = []
        self.progress.advance(1)

    def extract_page_images_and_drawings(self):
        """Extract raster images, vector drawings, and text logos from the current page."""
        try:
            # Get the current page from PyMuPDF document
            page = self.mupdf[self.current_page.page_number]

            # Extract raster images
            self.extract_raster_images(page)

            # Extract vector drawings
            self.extract_vector_drawings(page)

            # Extract text logos using PyMuPDF's structured text extraction
            self.extract_text_logos(page)

        except Exception as e:
            logger.warning("Failed to extract images and drawings from page %d: %s",
                           self.current_page.page_number, e)

    def extract_raster_images(self, page):
        """Extract raster images from the page using PyMuPDF with robust error handling."""
        try:
            seen_images = set()  # Track images by (page, bbox) to avoid duplicates

            images = page.get_images(full=True)
            for img_index, img_info in enumerate(images):
                try:
                    xref = img_info[0]
                    bbox = page.get_image_bbox(img_info)
                    if bbox is None:
                        continue

                    # Check for duplicate images
                    image_key = (self.current_page.page_number, tuple(bbox))
                    if image_key in seen_images:
                        continue
                    seen_images.add(image_key)

                    # Try multiple approaches for problematic images
                    img_bytes = None
                    pix = None

                    try:
                        pix = pymupdf.Pixmap(self.mupdf, xref)

                        if pix.colorspace is None:
                            # Try to handle images with no colorspace
                            try:
                                if pix.n >= 3:  # Has color channels
                                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                                elif pix.n == 1:  # Grayscale
                                    pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
                                else:
                                    # Try direct RGB conversion
                                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                            except:
                                # If all conversions fail, try to extract raw data
                                try:
                                    img_stream = self.mupdf.extract_image(xref)
                                    if img_stream and 'image' in img_stream:
                                        img_bytes = img_stream['image']
                                        logger.info(f"Extracted raw image data for problematic image on page {self.current_page.page_number}")
                                except:
                                    logger.warning(f"Skipping image on page {self.current_page.page_number}: cannot extract raw data")
                                    continue
                        elif pix.n < 5:  # Convert to RGB if necessary
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                        # If we have a pixmap, convert to bytes
                        if pix and not img_bytes:
                            img_bytes = pix.tobytes("png")

                    except Exception as pix_error:
                        # Try alternative extraction method
                        try:
                            img_stream = self.mupdf.extract_image(xref)
                            if img_stream and 'image' in img_stream:
                                img_bytes = img_stream['image']
                                logger.info(f"Extracted image using alternative method on page {self.current_page.page_number}")
                            else:
                                logger.warning(f"Skipping image on page {self.current_page.page_number}: no extractable data - {pix_error}")
                                continue
                        except Exception as alt_error:
                            logger.warning(f"Skipping image on page {self.current_page.page_number}: all extraction methods failed - {alt_error}")
                            continue

                    if img_bytes:
                        # Convert to base64
                        import base64
                        encoded_data = base64.b64encode(img_bytes).decode("ascii")

                        # Determine format (prefer PNG for consistency)
                        image_format = "PNG"

                        # Create IL image object
                        pdf_image = il_version_1.PdfImage(
                            box=il_version_1.Box(
                                x=float(bbox.x0),
                                y=float(bbox.y0),
                                x2=float(bbox.x1),
                                y2=float(bbox.y1),
                            ),
                            image_data=encoded_data,
                            image_format=image_format,
                            xobj_id=self.xobj_id,
                            render_order=self.get_render_order_and_increase(),
                        )

                        self.current_page.pdf_image.append(pdf_image)
                        logger.info(f"Extracted image on page {self.current_page.page_number} at {bbox}")

                    if pix:
                        pix = None  # Free memory

                except Exception as e:
                    logger.warning("Failed to extract image %d: %s", img_index, e)

        except Exception as e:
            logger.warning("Failed to extract raster images: %s", e)

    def extract_vector_drawings(self, page):
        """Extract vector drawings from the page using PyMuPDF."""
        try:
            drawings = page.get_drawings()
            if drawings:
                logger.info(f"Found {len(drawings)} drawings on page {self.current_page.page_number}")

            for drawing_index, drawing in enumerate(drawings):
                try:
                    # Create IL drawing object with the full drawing data
                    # We'll store the drawing data and handle rendering in pdf_creater.py
                    pdf_drawing = il_version_1.PdfDrawing(
                        box=il_version_1.Box(
                            x=float(drawing.get("rect", pymupdf.Rect(0, 0, 0, 0)).x0),
                            y=float(drawing.get("rect", pymupdf.Rect(0, 0, 0, 0)).y0),
                            x2=float(drawing.get("rect", pymupdf.Rect(0, 0, 0, 0)).x1),
                            y2=float(drawing.get("rect", pymupdf.Rect(0, 0, 0, 0)).y1),
                        ),
                        graphic_state=self.create_graphic_state(
                            self.passthrough_per_char_instruction, include_clipping=True
                        ),
                        drawing_commands=str(drawing),  # Store the full drawing dict as string
                        xobj_id=self.xobj_id,
                        render_order=self.get_render_order_and_increase(),
                    )

                    self.current_page.pdf_drawing.append(pdf_drawing)
                    logger.info(f"Extracted drawing {drawing_index} on page {self.current_page.page_number}")

                except Exception as e:
                    logger.warning("Failed to extract drawing %d: %s", drawing_index, e)

        except Exception as e:
            logger.warning("Failed to extract vector drawings: %s", e)

    def extract_text_logos(self, page):
        """Extract text logos using PyMuPDF's structured text extraction."""
        try:
            # Get full page text for reference
            full_text = page.get_text()
            text_dict = page.get_text("dict")

            seen_texts = set()  # Track text content to avoid duplicates

            for block in text_dict["blocks"]:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        spans = line["spans"]
                        if not spans:
                            continue

                        # Group consecutive spans with the same style
                        grouped_spans = []
                        current_group = [spans[0]]
                        current_bold = self._is_bold_span(spans[0])
                        current_italic = self._is_italic_span(spans[0])
                        current_size = spans[0]["size"]

                        for span in spans[1:]:
                            bold = self._is_bold_span(span)
                            italic = self._is_italic_span(span)
                            size = span["size"]
                            if bold == current_bold and italic == current_italic and size == current_size:
                                current_group.append(span)
                            else:
                                grouped_spans.append((current_group, current_bold, current_italic, current_size))
                                current_group = [span]
                                current_bold = bold
                                current_italic = italic
                                current_size = size
                        grouped_spans.append((current_group, current_bold, current_italic, current_size))

                        # Now create elements for each group
                        for group, bold, italic, size in grouped_spans:
                            group_text = "".join(span["text"] for span in group)
                            text = group_text.strip()
                            if not text:
                                continue

                            # Fix truncated text
                            text = self._fix_truncated_text(text, full_text)

                            # Skip duplicates and significant substring overlaps
                            skip_element = False
                            if text in seen_texts:
                                skip_element = True
                            else:
                                # Check if this text significantly overlaps with existing text
                                for seen_text in seen_texts:
                                    if text in seen_text and len(text) > 10:
                                        overlap_ratio = len(text) / len(seen_text)
                                        if overlap_ratio > 0.5:
                                            skip_element = True
                                            break

                            if skip_element:
                                continue

                            seen_texts.add(text)

                            # Check if this is a logo
                            if self._is_text_logo_from_pymupdf(text, size):
                                # Calculate bounding box for the entire group
                                min_x = min(span["bbox"][0] for span in group)
                                min_y = min(span["bbox"][1] for span in group)
                                max_x = max(span["bbox"][2] for span in group)
                                max_y = max(span["bbox"][3] for span in group)

                                # Create graphics state
                                gs = self.create_graphic_state(
                                    self.passthrough_per_char_instruction, include_clipping=True
                                )

                                bbox = il_version_1.Box(x=min_x, y=min_y, x2=max_x, y2=max_y)

                                pdf_style = il_version_1.PdfStyle(
                                    font_id=f"pymupdf_font_{size}_{bold}_{italic}",
                                    font_size=size,
                                    graphic_state=gs,
                                )

                                # Create text logo object
                                text_logo = il_version_1.PdfTextLogo(
                                    box=bbox,
                                    pdf_style=pdf_style,
                                    text=text,
                                    xobj_id=self.xobj_id,
                                    render_order=self.get_render_order_and_increase(),
                                )

                                self.current_page.pdf_text_logo.append(text_logo)
                                logger.info(f"Extracted text logo: '{text}' at ({bbox.x}, {bbox.y})")

        except Exception as e:
            logger.warning("Failed to extract text logos: %s", e)

    def _is_bold_span(self, span):
        """Check if a span is bold."""
        return (("Bold" in span["font"]) or
                ("-B" in span["font"]) or
                (span["font"].endswith("B")) or
                (span["flags"] & 16))

    def _is_italic_span(self, span):
        """Check if a span is italic."""
        return (("Italic" in span["font"]) or
                ("-I" in span["font"]) or
                (span["font"].endswith("I")) or
                (span["flags"] & 2))

    def _fix_truncated_text(self, truncated_text, full_page_text):
        """Try to find the complete text for a truncated string."""
        if len(truncated_text) < 10:
            return truncated_text

        pos = full_page_text.find(truncated_text)
        if pos == -1:
            return truncated_text

        start_pos = pos
        end_pos = pos + len(truncated_text) + 50
        if end_pos > len(full_page_text):
            end_pos = len(full_page_text)

        extended_text = full_page_text[start_pos:end_pos]
        lines = extended_text.split('\n')
        if lines and len(lines[0]) > len(truncated_text):
            complete_text = lines[0].strip()
            return complete_text

        return truncated_text

    def _is_text_logo_from_pymupdf(self, text, size):
        """Determine if text should be treated as a logo based on PyMuPDF extraction."""
        if not text or len(text.strip()) == 0:
            return False

        # Known publisher names that should be preserved as logos
        publisher_logos = [
            "APPLIED ENERGY", "APPLIED", "ENERGY", "ELSEVIER", "SCIENCEDIRECT", "SCIENCE DIRECT",
            "NATURE", "IEEE", "ACM", "SPRINGER", "WILEY", "TAYLOR & FRANCIS"
        ]

        text_upper = text.upper().strip()
        if text_upper in publisher_logos:
            return True

        # Check for large font size (logos are typically larger)
        if size > 12:  # Lower threshold for large text
            if (text_upper == text.strip() and  # All caps
                len(text.strip()) > 1 and  # Not too short
                not any(c.isdigit() for c in text)):  # No numbers
                return True

        return False

    def convert_drawing_to_pdf_commands(self, items):
        """Convert PyMuPDF drawing items to PDF drawing commands."""
        commands = []
        try:
            for item in items:
                if len(item) < 2:
                    continue

                op = item[0]
                args = item[1:]

                if op == "l":  # line to
                    if len(args) >= 2:
                        commands.append(f"{args[0]:.6f} {args[1]:.6f} l")
                elif op == "c":  # curve to
                    if len(args) >= 6:
                        commands.append(f"{args[0]:.6f} {args[1]:.6f} {args[2]:.6f} {args[3]:.6f} {args[4]:.6f} {args[5]:.6f} c")
                elif op == "m":  # move to
                    if len(args) >= 2:
                        commands.append(f"{args[0]:.6f} {args[1]:.6f} m")
                elif op == "h":  # close path
                    commands.append("h")
                elif op == "re":  # rectangle
                    if len(args) >= 4:
                        commands.append(f"{args[0]:.6f} {args[1]:.6f} {args[2]:.6f} {args[3]:.6f} re")
                elif op == "q":  # quadratic curve (convert to cubic)
                    if len(args) >= 4:
                        # Convert quadratic to cubic bezier
                        x0, y0, x1, y1, x2, y2 = args[:6]
                        # Simple conversion (this is approximate)
                        cx1 = x0 + (x1 - x0) * 2/3
                        cy1 = y0 + (y1 - y0) * 2/3
                        cx2 = x2 + (x1 - x2) * 2/3
                        cy2 = y2 + (y1 - y2) * 2/3
                        commands.append(f"{cx1:.6f} {cy1:.6f} {cx2:.6f} {cy2:.6f} {x2:.6f} {y2:.6f} c")

            return " ".join(commands) if commands else ""

        except Exception as e:
            logger.warning("Failed to convert drawing commands: %s", e)
            return ""

    def on_page_crop_box(
        self,
        x0: float | int,
        y0: float | int,
        x1: float | int,
        y1: float | int,
    ):
        box = il_version_1.Box(x=float(x0), y=float(y0), x2=float(x1), y2=float(y1))
        self.current_page.cropbox = il_version_1.Cropbox(box=box)

    def on_page_media_box(
        self,
        x0: float | int,
        y0: float | int,
        x1: float | int,
        y1: float | int,
    ):
        box = il_version_1.Box(x=float(x0), y=float(y0), x2=float(x1), y2=float(y1))
        self.current_page.mediabox = il_version_1.Mediabox(box=box)

    def on_page_number(self, page_number: int):
        assert isinstance(page_number, int)
        assert page_number >= 0
        self.current_page.page_number = page_number

    def on_page_base_operation(self, operation: str):
        operation = zstd_helper.zstd_compress(operation)
        self.current_page.base_operations = il_version_1.BaseOperations(value=operation)

    def on_page_resource_font(self, font: PDFFont, xref_id: int, font_id: str):
        font_name = font.fontname
        logger.debug(f"handle font {font_name} @ {xref_id} in {self.xobj_id}")
        if isinstance(font_name, bytes):
            try:
                font_name = font_name.decode("utf-8")
            except UnicodeDecodeError:
                font_name = "BASE64:" + base64.b64encode(font_name).decode("utf-8")
        encoding_length = 1
        if isinstance(font, PDFCIDFont):
            try:
                # pdf 32000:2008 page 273
                # Table 118 - Predefined CJK CMap names
                _, encoding = self.mupdf.xref_get_key(xref_id, "Encoding")
                if encoding == "/Identity-H" or encoding == "/Identity-V":
                    encoding_length = 2
                elif encoding == "/WinAnsiEncoding":
                    encoding_length = 1
                else:
                    _, to_unicode_id = self.mupdf.xref_get_key(xref_id, "ToUnicode")
                    if to_unicode_id is not None:
                        to_unicode_bytes = self.mupdf.xref_stream(
                            int(to_unicode_id.split(" ")[0]),
                        )
                        code_range = re.search(
                            b"begincodespacerange\n?.*<(\\d+?)>.*",
                            to_unicode_bytes,
                        ).group(1)
                        encoding_length = len(code_range) // 2
            except Exception:
                if (
                    font.unicode_map
                    and font.unicode_map.cid2unichr
                    and max(font.unicode_map.cid2unichr.keys()) > 255
                ):
                    encoding_length = 2
                else:
                    encoding_length = 1
        try:
            if xref_id in self.mupdf_font_map:
                mupdf_font = self.mupdf_font_map[xref_id]
            else:
                mupdf_font = pymupdf.Font(
                    fontbuffer=self.mupdf.extract_font(xref_id)[3]
                )
                mupdf_font.has_glyph = functools.lru_cache(maxsize=10240, typed=True)(
                    mupdf_font.has_glyph,
                )
            bold = mupdf_font.is_bold
            italic = mupdf_font.is_italic
            monospaced = mupdf_font.is_monospaced
            serif = mupdf_font.is_serif
            self.mupdf_font_map[xref_id] = mupdf_font
        except Exception:
            bold = None
            italic = None
            monospaced = None
            serif = None

        # Fallback bold/italic detection based on font name
        if not bold:
            bold = (("Bold" in font_name) or ("-B" in font_name))
        if not italic:
            italic = (("Italic" in font_name) or ("-I" in font_name) or ("Oblique" in font_name))

        logger.debug(f"Font {font_name} @ {xref_id}: bold={bold}, italic={italic}")

        il_font_metadata = il_version_1.PdfFont(
            name=font_name,
            xref_id=xref_id,
            font_id=font_id,
            encoding_length=encoding_length,
            bold=bold,
            italic=italic,
            monospace=monospaced,
            serif=serif,
            ascent=font.ascent,
            descent=font.descent,
            pdf_font_char_bounding_box=[],
        )
        try:
            if xref_id is None:
                logger.warning("xref_id is None for font %s", font_name)
            else:
                bbox_list, cmap = self.parse_font_xobj_id(xref_id)
                font_char_bounding_box_map = {}
                if not cmap:
                    cmap = {x: x for x in range(257)}
                for char_id, char_bbox in enumerate(bbox_list):
                    font_char_bounding_box_map[char_id] = char_bbox
                for char_id in cmap:
                    if char_id < 0 or char_id >= len(bbox_list):
                        continue
                    bbox = bbox_list[char_id]
                    x, y, x2, y2 = bbox
                    if (
                        x == 0
                        and y == 0
                        and x2 == 500
                        and y2 == 698
                        or x == 0
                        and y == 0
                        and x2 == 0
                        and y2 == 0
                    ):
                        # ignore default bounding box
                        continue
                    il_font_metadata.pdf_font_char_bounding_box.append(
                        il_version_1.PdfFontCharBoundingBox(
                            x=x,
                            y=y,
                            x2=x2,
                            y2=y2,
                            char_id=char_id,
                        )
                    )
                    font_char_bounding_box_map[char_id] = bbox
                if self.xobj_id in self.xobj_map:
                    if self.xobj_id not in self.current_page_font_char_bounding_box_map:
                        self.current_page_font_char_bounding_box_map[self.xobj_id] = {}
                    self.current_page_font_char_bounding_box_map[self.xobj_id][xref_id] = (
                        font_char_bounding_box_map
                    )
                else:
                    self.current_page_font_char_bounding_box_map[xref_id] = (
                        font_char_bounding_box_map
                    )
        except Exception as e:
            if xref_id is None:
                logger.error("failed to parse font xobj id None: %s", e)
            else:
                logger.error("failed to parse font xobj id %d: %s", xref_id, e)
        self.current_page_font_name_id_map[xref_id] = font_id
        self.current_available_fonts[font_id] = il_font_metadata

        fonts = self.current_page.pdf_font
        if self.xobj_id in self.xobj_map:
            fonts = self.xobj_map[self.xobj_id].pdf_font
        should_remove = []
        for f in fonts:
            if f.font_id == font_id:
                should_remove.append(f)
        for sr in should_remove:
            fonts.remove(sr)
        fonts.append(il_font_metadata)

    def parse_font_xobj_id(self, xobj_id: int):
        if xobj_id is None:
            return [], {}

        bbox_list = []
        encoding = parse_font_encoding(self.mupdf, xobj_id)
        differences = []
        font_differences = self.mupdf.xref_get_key(xobj_id, "Encoding/Differences")
        if font_differences:
            differences = parse_encoding(font_differences[1])
        for file_key in ["FontFile", "FontFile2", "FontFile3"]:
            font_file = self.mupdf.xref_get_key(xobj_id, f"FontDescriptor/{file_key}")
            if file_idx := indirect(font_file):
                bbox_list = parse_font_file(
                    self.mupdf,
                    file_idx,
                    encoding,
                    differences,
                )
        cmap = {}
        to_unicode = self.mupdf.xref_get_key(xobj_id, "ToUnicode")
        if to_unicode_idx := indirect(to_unicode):
            cmap = parse_cmap(self.mupdf.xref_stream(to_unicode_idx).decode("U8"))
        if not bbox_list:
            obj_type, obj_val = self.mupdf.xref_get_key(xobj_id, "BaseFont")
            if obj_type == "name":
                bbox_list = get_base14_bbox(obj_val[1:])
        if cid_bbox := get_cidfont_bbox(self.mupdf, xobj_id):
            bbox_list = cid_bbox
        return bbox_list, cmap

    def create_graphic_state(
        self,
        gs: babeldoc.pdfminer.pdfinterp.PDFGraphicState | list[tuple[str, str]],
        include_clipping: bool = False,
        target_ctm: tuple[float, float, float, float, float, float] = None,
        clip_paths=None,
    ):
        if clip_paths is None:
            clip_paths = self.current_clip_paths
        passthrough_instruction = getattr(gs, "passthrough_instruction", gs)

        def filter_clipping(op):
            return op not in ("W n", "W* n")

        def pass_all(_op):
            return True

        if include_clipping:
            filter_clipping = pass_all

        passthrough_per_char_instruction_parts = [
            f"{arg} {op}" for op, arg in passthrough_instruction if filter_clipping(op)
        ]

        # Add transformed clipping paths if requested and target CTM is provided
        if include_clipping and target_ctm and clip_paths:
            for clip_path, source_ctm, evenodd in clip_paths:
                try:
                    # Transform clip path from source CTM to target CTM
                    transformed_path = self.transform_clip_path(
                        clip_path, source_ctm, target_ctm
                    )

                    # Generate clipping instruction
                    op = "W* n" if evenodd else "W n"
                    args = []
                    for p in transformed_path:
                        if len(p) == 1:
                            args.append(p[0])
                        elif len(p) > 1:
                            args.extend([f"{x:F}" for x in p[1:]])
                            args.append(p[0])

                    if args:
                        clipping_instruction = f"{' '.join(args)} {op}"
                        passthrough_per_char_instruction_parts.append(
                            clipping_instruction
                        )

                except Exception as e:
                    logger.warning("Error transforming clip path: %s", e)

        passthrough_per_char_instruction = " ".join(
            passthrough_per_char_instruction_parts
        )

        # 可能会影响部分 graphic state 准确度。不过 BabelDOC 仅使用 passthrough_per_char_instruction
        # 所以应该是没啥影响
        # 但是池化 graphic state 后可以减少内存占用
        if passthrough_per_char_instruction not in self.graphic_state_pool:
            self.graphic_state_pool[passthrough_per_char_instruction] = (
                il_version_1.GraphicState(
                    passthrough_per_char_instruction=passthrough_per_char_instruction
                )
            )
        graphic_state = self.graphic_state_pool[passthrough_per_char_instruction]

        return graphic_state

    def on_lt_char(self, char: LTChar):
        if char.aw_font_id is None:
            return
        try:
            rotation_angle = get_rotation_angle(char.matrix)
            if not (-0.1 <= rotation_angle <= 0.1 or 89.9 <= rotation_angle <= 90.1):
                return
        except Exception:
            logger.warning(
                "Failed to get rotation angle for char %s",
                char.get_text(),
            )

        char_unicode = char.get_text()

        # For now, let's collect all large text and see what we get
        if char.size > 12 and char_unicode.strip():
            logger.info(f"Large text detected: '{char_unicode}' (size: {char.size}, pos: {char.bbox})")

        # Check if this text should be treated as a logo
        if self._is_text_logo(char, char_unicode):
            self._process_text_logo(char, char_unicode)
            return

        # Collect valid characters for statistics
        try:
            self._collect_valid_char(char_unicode)
        except Exception as e:
            logger.warning("Error collecting valid char: %s", e)
        gs = self.create_graphic_state(char.graphicstate)
        # Get font from current page or xobject
        font = None
        pdf_font = None
        for pdf_font in self.xobj_map.get(char.xobj_id, self.current_page).pdf_font:
            if pdf_font.font_id == char.aw_font_id:
                font = pdf_font
                break

        # Get descent from font
        descent = 0
        if font and hasattr(font, "descent"):
            descent = font.descent * char.size / 1000

        char_id = char.cid

        char_bounding_box = None
        try:
            if (
                font_bounding_box_map
                := self.current_page_font_char_bounding_box_map.get(
                    char.xobj_id, self.current_page_font_char_bounding_box_map
                ).get(font.xref_id)
            ):
                char_bounding_box = font_bounding_box_map.get(char_id, None)
            else:
                char_bounding_box = None
        except Exception:
            # logger.debug(
            #     "Failed to get font bounding box for char %s",
            #     char.get_text(),
            # )
            char_bounding_box = None

        char_unicode = char.get_text()
        # if "(cid:" not in char_unicode and len(char_unicode) > 1:
        #     return
        if space_regex.match(char_unicode):
            char_unicode = " "
        advance = char.adv
        bbox = il_version_1.Box(
            x=char.bbox[0],
            y=char.bbox[1],
            x2=char.bbox[2],
            y2=char.bbox[3],
        )
        if bbox.x2 < bbox.x or bbox.y2 < bbox.y:
            logger.warning(
                "Invalid bounding box for character %s: %s",
                char_unicode,
                bbox,
            )

        if char.matrix[0] == 0 and char.matrix[3] == 0:
            vertical = True
            visual_bbox = il_version_1.Box(
                x=char.bbox[0] - descent,
                y=char.bbox[1],
                x2=char.bbox[2] - descent,
                y2=char.bbox[3],
            )
        else:
            vertical = False
            # Add descent to y coordinates
            visual_bbox = il_version_1.Box(
                x=char.bbox[0],
                y=char.bbox[1] + descent,
                x2=char.bbox[2],
                y2=char.bbox[3] + descent,
            )
        visual_bbox = il_version_1.VisualBbox(box=visual_bbox)
        pdf_style = il_version_1.PdfStyle(
            font_id=char.aw_font_id,
            font_size=char.size,
            graphic_state=gs,
        )

        if font:
            font_xref_id = font.xref_id
            if font_xref_id in self.mupdf_font_map:
                mupdf_font = self.mupdf_font_map[font_xref_id]
                # if "(cid:" not in char_unicode:
                #     if mupdf_cid := mupdf_font.has_glyph(ord(char_unicode)):
                #         char_id = mupdf_cid

        pdf_char = il_version_1.PdfCharacter(
            box=bbox,
            pdf_character_id=char_id,
            advance=advance,
            char_unicode=char_unicode,
            vertical=vertical,
            pdf_style=pdf_style,
            xobj_id=char.xobj_id,
            visual_bbox=visual_bbox,
            render_order=char.render_order,
            sub_render_order=0,
        )
        if self.translation_config.ocr_workaround:
            pdf_char.pdf_style.graphic_state = BLACK
            pdf_char.render_order = None
        if pdf_style.font_size == 0.0:
            logger.warning(
                "Font size is 0.0 for character %s. Skip it.",
                char_unicode,
            )
            return

        if char_bounding_box and len(char_bounding_box) == 4:
            x_min, y_min, x_max, y_max = char_bounding_box
            factor = 1 / 1000 * pdf_style.font_size
            x_min = x_min * factor
            y_min = y_min * factor
            x_max = x_max * factor
            y_max = y_max * factor
            ll = (char.bbox[0] + x_min, char.bbox[1] + y_min)
            ur = (char.bbox[0] + x_max, char.bbox[1] + y_max)

            volume = (ur[0] - ll[0]) * (ur[1] - ll[1])
            if volume > 1:
                pdf_char.visual_bbox = il_version_1.VisualBbox(
                    il_version_1.Box(ll[0], ll[1], ur[0], ur[1])
                )

        self.current_page.pdf_character.append(pdf_char)

        if self.translation_config.show_char_box:
            self.current_page.pdf_rectangle.append(
                il_version_1.PdfRectangle(
                    box=pdf_char.visual_bbox.box,
                    graphic_state=YELLOW,
                    debug_info=True,
                    line_width=0.2,
                )
            )

    def _collect_valid_char(self, ch: str):
        """Append a valid character into the current page buffer according to rules.
        Rules:
        - Include whitespace matched by space_regex directly.
        - Ignore categories that are never normal text: {Cc, Cs, Co, Cn}.
        - Apply inverted criteria from formular_helper.py (21-28):
          empty -> invalid, contains '(cid:' -> invalid,
          not has_char(ch) -> invalid unless len(ch) > 1 and all(has_char(x)).
        """
        if self._page_valid_chars_buffer is None:
            return
        if space_regex.match(ch):
            self._page_valid_chars_buffer.append(ch)
            return
        try:
            cat = unicodedata.category(ch[0]) if ch else None
        except Exception:
            cat = None
        if cat in {"Cc", "Cs", "Co", "Cn"}:
            return
        is_invalid = False
        if not ch:
            is_invalid = True
        elif "(cid:" in ch:
            is_invalid = True
        else:
            try:
                if not self.font_mapper.has_char(ch):
                    if len(ch) > 1 and all(self.font_mapper.has_char(x) for x in ch):
                        is_invalid = False
                    else:
                        is_invalid = True
            except Exception:
                is_invalid = True
        if not is_invalid:
            self._page_valid_chars_buffer.append(ch)

    def on_lt_curve(self, curve: babeldoc.pdfminer.layout.LTCurve):
        if not self.enable_graphic_element_process:
            return
        bbox = il_version_1.Box(
            x=curve.bbox[0],
            y=curve.bbox[1],
            x2=curve.bbox[2],
            y2=curve.bbox[3],
        )
        # Extract CTM from curve object if it exists
        curve_ctm = getattr(curve, "ctm", None)
        gs = self.create_graphic_state(
            curve.passthrough_instruction,
            include_clipping=True,
            target_ctm=curve_ctm,
            clip_paths=curve.clip_paths,
        )
        paths = []
        for point in curve.original_path:
            op = point[0]
            if len(point) == 1:
                paths.append(
                    il_version_1.PdfPath(
                        op=op,
                        x=None,
                        y=None,
                        has_xy=False,
                    )
                )
                continue
            for p in point[1:-1]:
                paths.append(
                    il_version_1.PdfPath(
                        op="",
                        x=p[0],
                        y=p[1],
                        has_xy=True,
                    )
                )
            paths.append(
                il_version_1.PdfPath(
                    op=point[0],
                    x=point[-1][0],
                    y=point[-1][1],
                    has_xy=True,
                )
            )

        fill_background = curve.fill
        stroke_path = curve.stroke
        evenodd = curve.evenodd
        # Extract CTM from curve object if it exists
        ctm = getattr(curve, "ctm", None)

        # Extract raw path from curve object if it exists
        raw_path = getattr(curve, "raw_path", None)
        raw_pdf_paths = None
        if raw_path is not None:
            raw_pdf_paths = []
            for path in raw_path:
                if path[0] == "h":  # h command (close path)
                    raw_pdf_paths.append(
                        il_version_1.PdfOriginalPath(
                            pdf_path=il_version_1.PdfPath(
                                x=0.0,
                                y=0.0,
                                op=path[0],
                                has_xy=False,
                            )
                        )
                    )
                else:  # commands with coordinates (m, l, c, v, y, etc.)
                    for p in batched(path[1:-2], 2, strict=True):
                        raw_pdf_paths.append(
                            il_version_1.PdfOriginalPath(
                                pdf_path=il_version_1.PdfPath(
                                    x=float(p[0]),
                                    y=float(p[1]),
                                    op="",
                                    has_xy=True,
                                )
                            )
                        )
                    # Last point in the path
                    raw_pdf_paths.append(
                        il_version_1.PdfOriginalPath(
                            pdf_path=il_version_1.PdfPath(
                                x=float(path[-2]),
                                y=float(path[-1]),
                                op=path[0],
                                has_xy=True,
                            )
                        )
                    )

        curve_obj = il_version_1.PdfCurve(
            box=bbox,
            graphic_state=gs,
            pdf_path=paths,
            fill_background=fill_background,
            stroke_path=stroke_path,
            evenodd=evenodd,
            debug_info=False,  # Set to False by default for original graphics
            xobj_id=curve.xobj_id,
            render_order=curve.render_order,
            ctm=list(ctm) if ctm is not None else None,
            pdf_original_path=raw_pdf_paths,
        )
        self.current_page.pdf_curve.append(curve_obj)
        pass

    def on_xobj_form(
        self,
        ctm: tuple[float, float, float, float, float, float],
        xobj_id: int,
        xref_id: int,
        form_type: Literal["image", "form"],
        do_args: str,
        bbox: tuple[float, float, float, float],
        matrix: tuple[float, float, float, float, float, float],
    ):
        logger.debug(f"on_xobj_form: {do_args}[{bbox}] @ {xref_id} in {self.xobj_id}")
        matrix = mult_matrix(matrix, ctm)
        (x, y, w, h) = guarded_bbox(bbox)
        bounds = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
        bbox = get_bound(apply_matrix_pt(matrix, (p, q)) for (p, q) in bounds)

        gs = self.create_graphic_state(
            self.passthrough_per_char_instruction, include_clipping=True, target_ctm=ctm
        )

        figure_bbox = il_version_1.Box(
            x=bbox[0],
            y=bbox[1],
            x2=bbox[2],
            y2=bbox[3],
        )
        pdf_matrix = il_version_1.PdfMatrix(
            a=ctm[0],
            b=ctm[1],
            c=ctm[2],
            d=ctm[3],
            e=ctm[4],
            f=ctm[5],
        )
        affine_transform = decompose_ctm(ctm)
        xobj_form = il_version_1.PdfXobjForm(
            xref_id=xref_id,
            do_args=do_args,
        )
        pdf_form_subtype = il_version_1.PdfFormSubtype(
            pdf_xobj_form=xobj_form,
        )
        new_form = il_version_1.PdfForm(
            xobj_id=xobj_id,
            box=figure_bbox,
            pdf_matrix=pdf_matrix,
            graphic_state=gs,
            pdf_affine_transform=affine_transform,
            render_order=self.get_render_order_and_increase(),
            form_type=form_type,
            pdf_form_subtype=pdf_form_subtype,
            ctm=list(ctm),
        )
        self.current_page.pdf_form.append(new_form)

    def on_pdf_clip_path(
        self,
        clip_path,
        evenodd: bool,
        ctm: tuple[float, float, float, float, float, float],
    ):
        try:
            self.current_clip_paths.append((clip_path.copy(), ctm, evenodd))
        except Exception as e:
            logger.warning("Error in on_pdf_clip_path: %s", e)

    def create_il(self):
        pages = [
            page
            for page in self.docs.page
            if self.translation_config.should_translate_page(page.page_number + 1)
        ]
        self.docs.page = pages
        return self.docs

    def on_total_pages(self, total_pages: int):
        assert isinstance(total_pages, int)
        assert total_pages > 0
        self.docs.total_pages = total_pages
        total = 0
        for page in range(total_pages):
            if self.translation_config.should_translate_page(page + 1) is False:
                continue
            total += 1
        self.progress = self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        )

    def on_pdf_figure(self, figure: LTFigure):
        box = il_version_1.Box(
            figure.bbox[0],
            figure.bbox[1],
            figure.bbox[2],
            figure.bbox[3],
        )
        self.current_page.pdf_figure.append(il_version_1.PdfFigure(box=box))

    def on_inline_image_begin(self):
        """Begin processing inline image"""
        # Store current state for inline image processing
        self._inline_image_state = {
            "ctm": None,
            "parameters": {},
        }

    def on_inline_image_end(self, stream_obj, ctm):
        """End processing inline image and create PdfForm"""
        import base64
        import json

        from babeldoc.format.pdf.babelpdf.utils import guarded_bbox
        from babeldoc.format.pdf.document_il.utils.matrix_helper import decompose_ctm
        from babeldoc.pdfminer.utils import apply_matrix_pt
        from babeldoc.pdfminer.utils import get_bound

        # Extract image parameters from stream dictionary
        image_dict = stream_obj.attrs if hasattr(stream_obj, "attrs") else {}

        # Build parameters dictionary
        parameters = {}
        for key, value in image_dict.items():
            if hasattr(value, "name"):
                parameters[key] = value.name
            else:
                parameters[key] = str(value)

        # Get image data (encoded as base64)
        image_data = ""
        if hasattr(stream_obj, "data") and stream_obj.data is not None:
            image_data = base64.b64encode(stream_obj.data).decode("ascii")
        elif hasattr(stream_obj, "rawdata") and stream_obj.rawdata is not None:
            image_data = base64.b64encode(stream_obj.rawdata).decode("ascii")

        # Create inline form with parameters as JSON string
        inline_form = il_version_1.PdfInlineForm(
            form_data=image_data, image_parameters=json.dumps(parameters)
        )

        # Calculate bounding box - inline images are typically 1x1 unit square in user space
        bbox = (0, 0, 1, 1)
        (x, y, w, h) = guarded_bbox(bbox)
        bounds = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
        final_bbox = get_bound(apply_matrix_pt(ctm, (p, q)) for (p, q) in bounds)

        # Create graphics state
        gs = self.create_graphic_state(
            self.passthrough_per_char_instruction, include_clipping=True, target_ctm=ctm
        )

        # Create PdfMatrix from CTM
        pdf_matrix = il_version_1.PdfMatrix(
            a=ctm[0], b=ctm[1], c=ctm[2], d=ctm[3], e=ctm[4], f=ctm[5]
        )

        # Create affine transform
        affine_transform = decompose_ctm(ctm)

        # Create PdfFormSubtype with inline form
        pdf_form_subtype = il_version_1.PdfFormSubtype(pdf_inline_form=inline_form)

        # Create PdfForm for the inline image
        pdf_form = il_version_1.PdfForm(
            box=il_version_1.Box(
                x=final_bbox[0],
                y=final_bbox[1],
                x2=final_bbox[2],
                y2=final_bbox[3],
            ),
            graphic_state=gs,
            pdf_matrix=pdf_matrix,
            pdf_affine_transform=affine_transform,
            pdf_form_subtype=pdf_form_subtype,
            xobj_id=self.xobj_id,
            ctm=list(ctm),
            render_order=self.get_render_order_and_increase(),
            form_type="image",
        )

        # Add to current page
        self.current_page.pdf_form.append(pdf_form)

    def _is_text_logo(self, char: LTChar, text: str) -> bool:
        """Determine if text should be treated as a logo based on various criteria."""
        if not text or len(text.strip()) == 0:
            return False

        # Known publisher names that should be preserved as logos
        publisher_logos = [
            "APPLIED ENERGY", "APPLIED", "ENERGY", "ELSEVIER", "SCIENCEDIRECT", "SCIENCE DIRECT",
            "NATURE", "IEEE", "ACM", "SPRINGER", "WILEY", "TAYLOR & FRANCIS"
        ]

        # Check for exact matches with known publishers
        text_upper = text.upper().strip()
        if text_upper in publisher_logos:
            logger.info(f"Detected known publisher logo: {text_upper}")
            return True

        # Check for large font size (logos are typically larger)
        if char.size > 10:  # Even lower threshold for large text
            # Additional criteria for logo-like text
            if (text_upper == text.strip() and  # All caps
                len(text.strip()) > 1 and  # Not too short
                not any(c.isdigit() for c in text)):  # No numbers (likely not page numbers)
                logger.info(f"Detected logo-like text: '{text}' (size: {char.size})")
                return True

        return False

    def _process_text_logo(self, char: LTChar, text: str):
        """Process text that has been identified as a logo."""
        try:
            gs = self.create_graphic_state(char.graphicstate)

            # Get font from current page or xobject
            font = None
            for pdf_font in self.xobj_map.get(char.xobj_id, self.current_page).pdf_font:
                if pdf_font.font_id == char.aw_font_id:
                    font = pdf_font
                    break

            # Get descent from font
            descent = 0
            if font and hasattr(font, "descent"):
                descent = font.descent * char.size / 1000

            bbox = il_version_1.Box(
                x=char.bbox[0],
                y=char.bbox[1],
                x2=char.bbox[2],
                y2=char.bbox[3],
            )

            if char.matrix[0] == 0 and char.matrix[3] == 0:
                vertical = True
                visual_bbox = il_version_1.Box(
                    x=char.bbox[0] - descent,
                    y=char.bbox[1],
                    x2=char.bbox[2] - descent,
                    y2=char.bbox[3],
                )
            else:
                vertical = False
                visual_bbox = il_version_1.Box(
                    x=char.bbox[0],
                    y=char.bbox[1] + descent,
                    x2=char.bbox[2],
                    y2=char.bbox[3] + descent,
                )

            pdf_style = il_version_1.PdfStyle(
                font_id=char.aw_font_id,
                font_size=char.size,
                graphic_state=gs,
            )

            # Create text logo object
            text_logo = il_version_1.PdfTextLogo(
                box=bbox,
                pdf_style=pdf_style,
                text=text,
                xobj_id=char.xobj_id,
                render_order=self.get_render_order_and_increase(),
            )

            self.current_page.pdf_text_logo.append(text_logo)
            logger.info(f"Added text logo: '{text}' at position ({text_logo.box.x}, {text_logo.box.y})")

        except Exception as e:
            logger.warning("Failed to process text logo '%s': %s", text, e)

    def _post_process_text_logos(self):
        """Post-process characters to detect text logos that span multiple characters."""
        try:
            # Group characters by similar properties (font, size, position)
            char_groups = {}
            for char in self.current_page.pdf_character:
                if char.pdf_style.font_size > 10:  # Only consider large text
                    key = (
                        char.pdf_style.font_id,
                        round(char.pdf_style.font_size, 1),
                        round(char.box.y, 1),  # Same line
                        char.xobj_id
                    )
                    if key not in char_groups:
                        char_groups[key] = []
                    char_groups[key].append(char)

            # Process each group to find logo text
            for group_key, chars in char_groups.items():
                if len(chars) < 2:  # Need at least 2 characters for a logo
                    continue

                # Sort characters by x position
                chars.sort(key=lambda c: c.box.x)

                # Build text from consecutive characters
                current_text = ""
                current_chars = []
                last_x_end = None

                for char in chars:
                    char_text = char.char_unicode.strip()
                    if not char_text:
                        continue

                    # Check if this character is consecutive with the previous
                    if (last_x_end is None or
                        abs(char.box.x - last_x_end) < char.pdf_style.font_size * 0.5):  # Allow some spacing
                        current_text += char_text
                        current_chars.append(char)
                    else:
                        # Process the current text group
                        if current_text and len(current_text) > 1:
                            self._check_and_create_text_logo(current_text, current_chars)
                        # Start new group
                        current_text = char_text
                        current_chars = [char]

                    last_x_end = char.box.x2

                # Process the last group
                if current_text and len(current_text) > 1:
                    self._check_and_create_text_logo(current_text, current_chars)

        except Exception as e:
            logger.warning("Failed to post-process text logos: %s", e)

    def _check_and_create_text_logo(self, text: str, chars: list):
        """Check if text forms a logo and create it if so."""
        try:
            # Known publisher names that should be preserved as logos
            publisher_logos = [
                "APPLIED ENERGY", "APPLIED", "ENERGY", "ELSEVIER", "SCIENCEDIRECT", "SCIENCE DIRECT",
                "NATURE", "IEEE", "ACM", "SPRINGER", "WILEY", "TAYLOR & FRANCIS"
            ]

            text_upper = text.upper().strip()
            if text_upper in publisher_logos:
                logger.info(f"Detected multi-character logo: '{text_upper}'")

                # Calculate bounding box for the entire text
                min_x = min(c.box.x for c in chars)
                min_y = min(c.box.y for c in chars)
                max_x2 = max(c.box.x2 for c in chars)
                max_y2 = max(c.box.y2 for c in chars)

                # Use the first character's style
                first_char = chars[0]
                gs = first_char.pdf_style.graphic_state

                bbox = il_version_1.Box(x=min_x, y=min_y, x2=max_x2, y2=max_y2)

                pdf_style = il_version_1.PdfStyle(
                    font_id=first_char.pdf_style.font_id,
                    font_size=first_char.pdf_style.font_size,
                    graphic_state=gs,
                )

                # Create text logo object
                text_logo = il_version_1.PdfTextLogo(
                    box=bbox,
                    pdf_style=pdf_style,
                    text=text,
                    xobj_id=first_char.xobj_id,
                    render_order=self.get_render_order_and_increase(),
                )

                self.current_page.pdf_text_logo.append(text_logo)
                logger.info(f"Created multi-character text logo: '{text}' at ({bbox.x}, {bbox.y})")

                # Remove the individual characters that form this logo
                for char in chars:
                    if char in self.current_page.pdf_character:
                        self.current_page.pdf_character.remove(char)

        except Exception as e:
            logger.warning("Failed to create text logo for '%s': %s", text, e)
