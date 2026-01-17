# Vinzy DataDiff 📊

A comprehensive Python library for comparing pandas DataFrames, identifying differences at the row and cell level, and exporting comparison results in various formats.

## Features

- **Row-level comparison**: Identify added, removed, modified, and unchanged rows
- **Cell-level tracking**: Get detailed cell-by-cell differences for modified rows
- **Key-based or index-based comparison**: Compare by unique key columns or by DataFrame index
- **Schema comparison**: Detect column additions, removals, and data type changes
- **Numerical tolerance**: Compare floating-point numbers with configurable tolerance
- **String comparison options**: Case-insensitive and whitespace-ignoring comparisons
- **Multiple export formats**: Excel, CSV, JSON, and HTML reports
- **Convenience functions**: Quick one-liner comparisons for simple use cases

## Installation

```bash
# Basic installation
pip install vinzy_datadiff

# With Excel export support
pip install vinzy_datadiff[excel]

# Development installation
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from vinzy_datadiff import compare_dataframes, quick_diff, DataFrameDiff

# Sample DataFrames
df1 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40]
})

df2 = pd.DataFrame({
    'id': [1, 2, 3, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Eve'],
    'age': [26, 30, 35, 28]
})

# Quick comparison
result = compare_dataframes(df1, df2, key_columns=['id'])
print(result.summary)
# {'total_rows_df1': 4, 'total_rows_df2': 4, 'added_rows': 1,
#  'removed_rows': 1, 'modified_rows': 1, 'unchanged_rows': 2, 'identical': False}

# Even quicker
summary = quick_diff(df1, df2, key_columns=['id'])
print(summary)
# {'added': 1, 'removed': 1, 'modified': 1, 'unchanged': 2, 'identical': False}
```

## Detailed Usage

### Class-based Comparison

```python
from vinzy_datadiff import DataFrameDiff

# Create differ with options
differ = DataFrameDiff(
    ignore_index=False,          # Use DataFrame index for comparison
    ignore_column_order=True,    # Ignore column order differences
    case_sensitive=False,        # Case-insensitive string comparison
    ignore_whitespace=True,      # Strip whitespace from strings
    treat_null_as_equal=True     # NaN/None values are considered equal
)

# Compare DataFrames
result = differ.compare(
    df1, df2,
    key_columns=['id'],          # Columns to identify unique rows
    compare_columns=['name', 'age'],  # Columns to compare (optional)
    tolerance=0.01,              # Numerical tolerance
    track_cell_changes=True      # Track individual cell changes
)
```

### Accessing Results

```python
# Get different row types
added_rows = result.get_added()
removed_rows = result.get_removed()
modified_rows = result.get_modified()
unchanged_rows = result.get_unchanged()

# Get all differences with change type
all_diffs = result.get_all_differences()
# Returns DataFrame with '_change_type' column

# Get cell-level differences
cell_diffs = result.get_cell_diffs()
for diff in cell_diffs:
    print(f"Row {diff.row_key}, Column '{diff.column}': {diff.old_value} → {diff.new_value}")

# Get as DataFrame
cell_diff_df = result.get_cell_diffs_df()
```

### Schema Comparison

```python
# Compare only schema (columns and dtypes)
schema_diff = differ.compare_schema(df1, df2)
print(f"Added columns: {schema_diff.added_columns}")
print(f"Removed columns: {schema_diff.removed_columns}")
print(f"Dtype changes: {schema_diff.dtype_changes}")
```

### Export Results

```python
# Export to Excel (requires openpyxl)
result.export_to_excel('comparison.xlsx')

# Export to CSV (creates multiple files)
result.export_to_csv('comparison')  # Creates comparison_added.csv, etc.

# Export to JSON
result.export_to_json('comparison.json')

# Export to HTML with styling
result.export_to_html('comparison.html')

# Or use the differ directly
differ.export_differences(df1, df2, 'output.xlsx', key_columns=['id'], format='excel')
```

### Boolean Checks

```python
# Quick boolean checks
if result.has_changes():
    print("DataFrames are different")

if result.has_added_rows():
    print(f"Found {len(result.get_added())} new rows")

if result.has_column_changes():
    print("Column structure changed")

# Or check directly
if differ.are_identical(df1, df2, key_columns=['id']):
    print("DataFrames are identical")
```

### Numerical Tolerance

```python
df_float1 = pd.DataFrame({'value': [1.001, 2.002, 3.003]})
df_float2 = pd.DataFrame({'value': [1.000, 2.000, 3.000]})

# Strict comparison (default)
result_strict = compare_dataframes(df_float1, df_float2)
print(result_strict.summary['identical'])  # False

# With tolerance
result_tolerant = compare_dataframes(df_float1, df_float2, tolerance=0.01)
print(result_tolerant.summary['identical'])  # True
```

## API Reference

### Classes

- `DataFrameDiff` - Main comparison class
- `DiffResult` - Container for comparison results
- `CellDiff` - Represents a single cell difference
- `SchemaDiff` - Represents schema differences
- `DiffType` - Enum for difference types (ADDED, REMOVED, MODIFIED, UNCHANGED)

### Convenience Functions

- `compare_dataframes(df1, df2, ...)` - Quick DataFrame comparison
- `quick_diff(df1, df2, ...)` - Get a simple summary dictionary
- `are_dataframes_equal(df1, df2, ...)` - Check if DataFrames are equal

## Requirements

- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.20.0
- openpyxl >= 3.0.0 (optional, for Excel export)

## License

MIT License - see LICENSE file for details.
