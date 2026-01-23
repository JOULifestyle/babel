# Text Overflow Logic Analysis and Improvement Plan

## Current Implementation Overview

The BabelDOC typesetting system handles text overflow through a multi-stage process in the `Typesetter` class:

### 1. Scale-Based Fitting (`_find_optimal_scale_and_layout`)
- **Method**: Iteratively reduces scale (1.0 → 0.9 → 0.8 → ... → 0.1) until text fits
- **Overflow Control**: `allow_overflow` parameter enables special handling for structured paragraphs
- **Structured Paragraphs**: When `preserve_line_structure=True`, allows horizontal overflow at scale 0.7
- **Fallback**: Tries disabling English line breaks if initial attempts fail

### 2. Line Layout Logic (`_layout_typesetting_units`)
- **Line Breaking**: Decides breaks based on box width, English lookahead, and punctuation rules
- **Structured vs Normal**:
  - Normal paragraphs: Fit within box by scaling
  - Structured paragraphs: Allow horizontal overflow to page width (595pt)
- **Vertical Overflow**: Detects when text extends below box bottom

### 3. Positioning (`_flush_line`)
- Places text lines with proper spacing and indentation
- Handles formula padding and CJK character spacing

## Identified Issues

### 1. Inconsistent Overflow Strategy
**Problem**: Mixed approaches create unpredictable behavior
- Scale reduction for fitting
- Horizontal overflow for structured content
- Vertical overflow detection
- No clear priority or fallback hierarchy

**Impact**: Text may overflow horizontally, vertically, or be squeezed inappropriately

### 2. Hard-coded Constraints
**Problem**: Fixed values don't adapt to content or page layout
- Page width: 595pt (A4 assumption)
- Scale jump: Abrupt change from variable scale to 0.7
- No consideration of actual page dimensions

**Impact**: Incorrect overflow on different page sizes or orientations

### 3. Limited Graceful Degradation
**Problem**: Binary choices between fitting and overflow
- No intermediate options like font size adjustment
- No word spacing or letter spacing modifications
- No hyphenation or other text compression techniques

**Impact**: Poor readability when text can't fit naturally

### 4. Complex Conditional Logic
**Problem**: Multiple nested conditions make behavior hard to predict
```python
if scale < 0.7:
    if allow_overflow and getattr(paragraph, 'preserve_line_structure', False):
        # Allow overflow
    else:
        # Continue scaling
```

**Impact**: Difficult to debug and maintain

### 5. Vertical Overflow Handling
**Problem**: Vertical overflow detection exists but lacks resolution strategies
- Only sets `all_units_fit = False`
- No attempt to expand box vertically
- No pagination or text flow to next area

**Impact**: Text gets cut off or improperly positioned

## Proposed Improvements

### 1. Unified Overflow Strategy
**Design**: Implement a configurable overflow policy system

```python
class OverflowPolicy:
    NONE = "none"                    # No overflow allowed
    HORIZONTAL = "horizontal"        # Allow horizontal expansion
    VERTICAL = "vertical"           # Allow vertical expansion
    SCALE_ONLY = "scale_only"       # Only use scaling
    HYBRID = "hybrid"               # Scale first, then overflow
```

**Benefits**:
- Clear, predictable behavior
- Easy to configure per paragraph type
- Separates concerns

### 2. Dynamic Page Awareness
**Design**: Replace hard-coded values with dynamic page detection

```python
def get_effective_page_width(self, paragraph, page) -> float:
    """Calculate available width considering margins and page dimensions"""
    page_width = getattr(page, 'width', 595.0)  # Default A4
    left_margin = getattr(paragraph, 'left_margin', 0)
    right_margin = getattr(paragraph, 'right_margin', 0)
    return page_width - left_margin - right_margin
```

**Benefits**:
- Works with any page size
- Respects document margins
- Future-proof for different layouts

### 3. Graceful Degradation Pipeline
**Design**: Implement progressive text fitting strategies

```python
def apply_overflow_strategies(self, units, box, policy):
    strategies = [
        self._try_fit_with_spacing_adjustment,
        self._try_fit_with_font_size_adjustment,
        self._try_fit_with_hyphenation,
        self._try_allow_horizontal_overflow,
        self._try_allow_vertical_expansion,
        self._try_force_fit_with_minimum_scale
    ]

    for strategy in strategies:
        if strategy(units, box):
            return True
    return False
```

**Benefits**:
- Maintains readability as long as possible
- Progressive fallback ensures text is always visible
- Configurable strategy order

### 4. Improved Vertical Overflow Handling
**Design**: Add vertical expansion and pagination support

```python
def handle_vertical_overflow(self, paragraph, page, overflow_amount):
    """Handle text that exceeds vertical bounds"""
    if self.can_expand_downward(paragraph, page, overflow_amount):
        self.expand_paragraph_box(paragraph, overflow_amount)
        return True
    elif self.can_create_new_paragraph(paragraph, page):
        self.split_paragraph_at_page_break(paragraph, page)
        return True
    return False
```

**Benefits**:
- Prevents text cutoff
- Maintains document flow
- Handles multi-page content properly

### 5. Configuration-Driven Behavior
**Design**: Make overflow behavior configurable per document type

```python
overflow_config = {
    'default': OverflowPolicy.HYBRID,
    'structured': OverflowPolicy.HORIZONTAL,
    'captions': OverflowPolicy.SCALE_ONLY,
    'headers': OverflowPolicy.NONE
}
```

**Benefits**:
- Tailored behavior for different content types
- Easy to adjust without code changes
- Consistent handling across similar documents

## Implementation Plan

### Phase 1: Core Refactoring
1. Extract overflow logic into dedicated `OverflowManager` class
2. Implement `OverflowPolicy` enum and configuration
3. Replace hard-coded values with dynamic calculations
4. Add comprehensive logging for overflow decisions

### Phase 2: Enhanced Strategies
1. Implement spacing adjustment strategy
2. Add font size adjustment (within reasonable bounds)
3. Implement hyphenation support for long words
4. Add vertical expansion logic

### Phase 3: Advanced Features
1. Add pagination support for long content
2. Implement content-aware overflow decisions
3. Add overflow metrics and reporting
4. Create overflow visualization for debugging

### Phase 4: Testing and Validation
1. Create comprehensive test cases for different overflow scenarios
2. Validate behavior across different document types
3. Performance testing for large documents
4. User acceptance testing with real-world PDFs

## Migration Strategy

### Backward Compatibility
- Keep existing `allow_overflow` parameter as deprecated
- Map old behavior to new policies automatically
- Add migration warnings in logs

### Gradual Rollout
- Start with low-risk paragraphs (body text)
- Gradually enable for structured content
- Monitor for regressions before full deployment

## Success Metrics

1. **Text Visibility**: 100% of translated text should be visible (no cutoff)
2. **Readability**: Maintain acceptable font sizes (minimum 0.7 scale)
3. **Layout Preservation**: Structured content maintains intended layout
4. **Performance**: No significant impact on processing time
5. **Maintainability**: Clear, documented overflow logic

## Risk Assessment

### Low Risk
- Dynamic page width calculation
- Improved logging and debugging
- Configuration-driven policies

### Medium Risk
- Vertical expansion (may affect page layout)
- New overflow strategies (may change text appearance)

### High Risk
- Pagination support (complex interaction with existing layout)
- Major refactoring of core logic (potential for bugs)

## Conclusion

The current overflow logic, while functional, suffers from inconsistent behavior and limited adaptability. The proposed improvements provide a more robust, configurable, and maintainable system that better handles the diverse requirements of PDF translation while maintaining text visibility and readability.