# Data Directory

This project separates inputs from derived outputs.

- `raw/` is for source files, downloaded event data, and untouched extracts.
- `processed/` is for cleaned tables, model-ready action data, and exported summaries.

Keep raw files immutable where possible. If a transformation is needed, write the result into `processed/` and document the step in code rather than editing source files by hand.
