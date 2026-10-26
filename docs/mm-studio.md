# MM photography workspace

Updated: 2026-10-22

The desktop opens the MM workspace directly. The visual reference supplied for this customization drives its charcoal surfaces, warm selection accent, library sidebar, independent vertical photo strip, central loupe, and right inspector with histogram, Light, Curve and collapsible adjustment sections. Existing engine commands continue to own edits, selection, printing and export.

## Navigation and responsive layout

- Detail and Compare show a vertical photo strip. F6 and the filmstrip button share the standard visibility setting. Keyboard or toolbar selection scrolls the selected photo into view once, while manual scrolling remains available.
- At widths below 1080 points, Detail hides library navigation to preserve the canvas. Biblioteca opens a grid with the library visible even at the desktop minimum width of 900 points; the edit inspector gives its space to the grid.
- Presets use a separate drawer on wide screens and the inspector region on smaller screens.
- Startup bypasses the previous decorative home screen. Missing EXIF displays an empty-data message instead of invented camera settings.

## Performance and validation

The strip uses `ScrollArea::show_rows` and the session's shared visible IDs. Only viewport rows build widgets and request thumbnails; the complete ID list is not copied each frame. Selection reconciliation is cached by visible generation and active ID. Album counts reuse the existing incremental cache.

`tests_studio` covers a 3000-photo catalog (fewer than 15 film cells at 1600 × 1000), selection follow, empty filters, laptop navigation, F6 visibility, slider changes and undo, and inspector bounds. This establishes bounded strip work; it is not a benchmark of RAW decoding, export, GPU performance or full catalog startup.

Visual QA uses generated demo images through the real headless UI at 1600 × 1000 and 900 × 700. No reference photograph or screenshot asset is embedded in the product.
