"""
Vinzy DataDiff - A comprehensive DataFrame comparison library

This library provides tools for comparing pandas DataFrames, identifying
differences at the row and cell level, and exporting comparison results
in various formats.

Basic Usage:
    >>> from vinzy_datadiff import DataFrameDiff, compare_dataframes
    >>> 
    >>> # Quick comparison
    >>> result = compare_dataframes(df1, df2, key_columns=['id'])
    >>> print(result.summary)
    >>> 
    >>> # Class-based usage with options
    >>> differ = DataFrameDiff(ignore_index=True, case_sensitive=False)
    >>> result = differ.compare(df1, df2, key_columns=['id'], tolerance=0.01)
    >>> 
    >>> # Access different row types
    >>> added = result.get_added()
    >>> removed = result.get_removed()
    >>> modified = result.get_modified()
    >>> 
    >>> # Get cell-level differences
    >>> cell_diffs = result.get_cell_diffs()
    >>> 
    >>> # Export results
    >>> result.export_to_excel('comparison.xlsx')
    >>> result.export_to_html('comparison.html')
"""

from .datadiff import (
    # Main classes
    DataFrameDiff,
    DiffResult,
    DiffType,
    CellDiff,
    SchemaDiff,
    
    # Convenience functions
    compare_dataframes,
    quick_diff,
    are_dataframes_equal,
    
    # Version
    __version__,
)

__all__ = [
    # Main classes
    "DataFrameDiff",
    "DiffResult", 
    "DiffType",
    "CellDiff",
    "SchemaDiff",
    
    # Convenience functions
    "compare_dataframes",
    "quick_diff",
    "are_dataframes_equal",
    
    # Version
    "__version__",
]
