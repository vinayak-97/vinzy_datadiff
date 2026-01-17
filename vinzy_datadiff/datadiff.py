"""
DataFrame Diff Library - A comprehensive tool for comparing pandas DataFrames

Features:
- Row-level comparison (added, removed, modified, unchanged)
- Column-level changes detection
- Key-based or index-based comparison
- Numerical tolerance support
- Detailed cell-by-cell diff
- Schema validation and comparison
- Export to Excel, CSV, JSON, and HTML
- Configurable comparison options
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import json
from io import StringIO


__version__ = "0.1.0"
__all__ = ["DataFrameDiff", "DiffResult", "DiffType", "CellDiff", "SchemaDiff"]


class DiffType(Enum):
    """Types of differences between DataFrames"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class CellDiff:
    """Represents a single cell difference"""
    row_key: Any
    column: str
    old_value: Any
    new_value: Any
    diff_type: DiffType
    
    def __str__(self) -> str:
        return f"CellDiff({self.column}: {self.old_value!r} -> {self.new_value!r})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'row_key': self.row_key,
            'column': self.column,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'diff_type': self.diff_type.value
        }


@dataclass
class SchemaDiff:
    """Represents schema differences between DataFrames"""
    added_columns: List[str]
    removed_columns: List[str]
    common_columns: List[str]
    dtype_changes: Dict[str, Tuple[str, str]]
    column_order_changed: bool
    
    def __str__(self) -> str:
        return f"SchemaDiff(added: {len(self.added_columns)}, removed: {len(self.removed_columns)}, dtype_changes: {len(self.dtype_changes)})"
    
    def has_changes(self) -> bool:
        """Check if there are any schema changes"""
        return (len(self.added_columns) > 0 or 
                len(self.removed_columns) > 0 or 
                len(self.dtype_changes) > 0 or 
                self.column_order_changed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'added_columns': self.added_columns,
            'removed_columns': self.removed_columns,
            'common_columns': self.common_columns,
            'dtype_changes': self.dtype_changes,
            'column_order_changed': self.column_order_changed
        }


@dataclass
class DiffResult:
    """Container for DataFrame comparison results"""
    summary: Dict[str, Any]
    added_rows: pd.DataFrame
    removed_rows: pd.DataFrame
    modified_rows: pd.DataFrame
    unchanged_rows: pd.DataFrame
    column_changes: Dict[str, Any]
    cell_diffs: List[CellDiff] = field(default_factory=list)
    schema_diff: Optional[SchemaDiff] = None
    _detailed_modifications: Optional[pd.DataFrame] = field(default=None, repr=False)
    
    def __str__(self) -> str:
        return f"DiffResult(added: {len(self.added_rows)}, removed: {len(self.removed_rows)}, modified: {len(self.modified_rows)})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_added(self) -> pd.DataFrame:
        """Get added rows as a DataFrame"""
        return self.added_rows.copy()
    
    def get_removed(self) -> pd.DataFrame:
        """Get removed rows as a DataFrame"""
        return self.removed_rows.copy()
    
    def get_modified(self) -> pd.DataFrame:
        """Get modified rows as a DataFrame"""
        return self.modified_rows.copy()
    
    def get_unchanged(self) -> pd.DataFrame:
        """Get unchanged rows as a DataFrame"""
        return self.unchanged_rows.copy()
    
    def get_detailed_modifications(self) -> pd.DataFrame:
        """
        Get detailed modifications showing old and new values side by side.
        
        Returns a DataFrame with columns: key_columns, column_name, old_value, new_value
        """
        if self._detailed_modifications is not None:
            return self._detailed_modifications.copy()
        return pd.DataFrame()
    
    def get_cell_diffs(self) -> List[CellDiff]:
        """Get list of cell-level differences"""
        return list(self.cell_diffs)
    
    def get_cell_diffs_df(self) -> pd.DataFrame:
        """Get cell differences as a DataFrame"""
        if not self.cell_diffs:
            return pd.DataFrame(columns=['row_key', 'column', 'old_value', 'new_value', 'diff_type'])
        return pd.DataFrame([cd.to_dict() for cd in self.cell_diffs])
    
    def get_changes(self) -> pd.DataFrame:
        """Get all changed rows (added + modified) as a DataFrame"""
        if len(self.added_rows) == 0 and len(self.modified_rows) == 0:
            return pd.DataFrame()
        elif len(self.added_rows) == 0:
            return self.modified_rows.copy()
        elif len(self.modified_rows) == 0:
            return self.added_rows.copy()
        else:
            return pd.concat([self.added_rows, self.modified_rows], ignore_index=True)
    
    def get_all_differences(self) -> pd.DataFrame:
        """Get all differences (added + removed + modified) as a DataFrame with change type"""
        dfs = []
        
        if len(self.added_rows) > 0:
            added_df = self.added_rows.copy()
            added_df['_change_type'] = 'added'
            dfs.append(added_df)
        
        if len(self.removed_rows) > 0:
            removed_df = self.removed_rows.copy()
            removed_df['_change_type'] = 'removed'
            dfs.append(removed_df)
        
        if len(self.modified_rows) > 0:
            modified_df = self.modified_rows.copy()
            modified_df['_change_type'] = 'modified'
            dfs.append(modified_df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary as a dictionary"""
        return self.summary.copy()
    
    def get_column_changes(self) -> Dict[str, Any]:
        """Get column changes as a dictionary"""
        return self.column_changes.copy()
    
    def has_changes(self) -> bool:
        """Check if there are any changes between the DataFrames"""
        return not self.summary['identical']
    
    def has_added_rows(self) -> bool:
        """Check if there are any added rows"""
        return len(self.added_rows) > 0
    
    def has_removed_rows(self) -> bool:
        """Check if there are any removed rows"""
        return len(self.removed_rows) > 0
    
    def has_modified_rows(self) -> bool:
        """Check if there are any modified rows"""
        return len(self.modified_rows) > 0
    
    def has_column_changes(self) -> bool:
        """Check if there are any column structure changes"""
        if self.schema_diff:
            return self.schema_diff.has_changes()
        return (len(self.column_changes.get('added_columns', [])) > 0 or 
                len(self.column_changes.get('removed_columns', [])) > 0 or
                self.column_changes.get('column_order_changed', False))
    
    def get_schema_diff(self) -> Optional[SchemaDiff]:
        """Get schema difference object"""
        return self.schema_diff
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire result to a dictionary"""
        result = {
            'summary': self.summary,
            'added_rows': self.added_rows.to_dict('records') if len(self.added_rows) > 0 else [],
            'removed_rows': self.removed_rows.to_dict('records') if len(self.removed_rows) > 0 else [],
            'modified_rows': self.modified_rows.to_dict('records') if len(self.modified_rows) > 0 else [],
            'unchanged_rows': self.unchanged_rows.to_dict('records') if len(self.unchanged_rows) > 0 else [],
            'column_changes': self.column_changes,
            'cell_diffs': [cd.to_dict() for cd in self.cell_diffs]
        }
        if self.schema_diff:
            result['schema_diff'] = self.schema_diff.to_dict()
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert the entire result to a JSON string"""
        def json_serializer(obj):
            if isinstance(obj, (pd.Timestamp, np.datetime64)):
                return str(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if pd.isna(obj):
                return None
            return str(obj)
        
        return json.dumps(self.to_dict(), default=json_serializer, indent=indent)
    
    def export_to_excel(self, filename: str) -> None:
        """Export all differences to an Excel file with separate sheets"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Create summary sheet first
                summary_df = pd.DataFrame(list(self.summary.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                if len(self.added_rows) > 0:
                    self.added_rows.to_excel(writer, sheet_name='Added_Rows', index=False)
                if len(self.removed_rows) > 0:
                    self.removed_rows.to_excel(writer, sheet_name='Removed_Rows', index=False)
                if len(self.modified_rows) > 0:
                    self.modified_rows.to_excel(writer, sheet_name='Modified_Rows', index=False)
                if len(self.unchanged_rows) > 0:
                    self.unchanged_rows.to_excel(writer, sheet_name='Unchanged_Rows', index=False)
                
                # Add cell diffs if available
                cell_diff_df = self.get_cell_diffs_df()
                if len(cell_diff_df) > 0:
                    cell_diff_df.to_excel(writer, sheet_name='Cell_Differences', index=False)
                
                # Add detailed modifications if available
                if self._detailed_modifications is not None and len(self._detailed_modifications) > 0:
                    self._detailed_modifications.to_excel(writer, sheet_name='Detailed_Changes', index=False)
                
        except ImportError:
            raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
    
    def export_to_csv(self, base_filename: str) -> List[str]:
        """
        Export differences to separate CSV files
        
        Returns:
            List of created file paths
        """
        created_files = []
        
        if len(self.added_rows) > 0:
            path = f"{base_filename}_added.csv"
            self.added_rows.to_csv(path, index=False)
            created_files.append(path)
        if len(self.removed_rows) > 0:
            path = f"{base_filename}_removed.csv"
            self.removed_rows.to_csv(path, index=False)
            created_files.append(path)
        if len(self.modified_rows) > 0:
            path = f"{base_filename}_modified.csv"
            self.modified_rows.to_csv(path, index=False)
            created_files.append(path)
        if len(self.unchanged_rows) > 0:
            path = f"{base_filename}_unchanged.csv"
            self.unchanged_rows.to_csv(path, index=False)
            created_files.append(path)
        
        # Export summary
        summary_df = pd.DataFrame(list(self.summary.items()), columns=['Metric', 'Value'])
        summary_path = f"{base_filename}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        created_files.append(summary_path)
        
        # Export cell diffs if available
        cell_diff_df = self.get_cell_diffs_df()
        if len(cell_diff_df) > 0:
            path = f"{base_filename}_cell_diffs.csv"
            cell_diff_df.to_csv(path, index=False)
            created_files.append(path)
        
        return created_files
    
    def export_to_json(self, filename: str) -> None:
        """Export all differences to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    def export_to_html(self, filename: str, include_styles: bool = True) -> None:
        """
        Export differences to an HTML file with formatted tables
        
        Args:
            filename: Output HTML file path
            include_styles: Whether to include CSS styling
        """
        html_parts = []
        
        if include_styles:
            html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <title>DataFrame Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; margin: 10px 0 30px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .added { background-color: #d4edda; }
        .removed { background-color: #f8d7da; }
        .modified { background-color: #fff3cd; }
        .section { margin-bottom: 30px; }
    </style>
</head>
<body>
""")
        else:
            html_parts.append("<html><body>")
        
        html_parts.append("<h1>DataFrame Comparison Report</h1>")
        
        # Summary section
        html_parts.append('<div class="summary">')
        html_parts.append("<h2>Summary</h2>")
        html_parts.append("<ul>")
        for key, value in self.summary.items():
            html_parts.append(f"<li><strong>{key}:</strong> {value}</li>")
        html_parts.append("</ul></div>")
        
        # Added rows
        if len(self.added_rows) > 0:
            html_parts.append('<div class="section added">')
            html_parts.append(f"<h2>Added Rows ({len(self.added_rows)})</h2>")
            html_parts.append(self.added_rows.to_html(index=False, classes='table'))
            html_parts.append("</div>")
        
        # Removed rows
        if len(self.removed_rows) > 0:
            html_parts.append('<div class="section removed">')
            html_parts.append(f"<h2>Removed Rows ({len(self.removed_rows)})</h2>")
            html_parts.append(self.removed_rows.to_html(index=False, classes='table'))
            html_parts.append("</div>")
        
        # Modified rows
        if len(self.modified_rows) > 0:
            html_parts.append('<div class="section modified">')
            html_parts.append(f"<h2>Modified Rows ({len(self.modified_rows)})</h2>")
            html_parts.append(self.modified_rows.to_html(index=False, classes='table'))
            html_parts.append("</div>")
        
        html_parts.append("</body></html>")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))


class DataFrameDiff:
    """
    Main class for comparing DataFrames.
    
    Features:
        - Row-level comparison (added, removed, modified, unchanged)
        - Column structure change detection
        - Key-based or index-based comparison
        - Numerical tolerance support
        - Cell-level detailed differences
        - Multiple export formats
    
    Example:
        >>> differ = DataFrameDiff()
        >>> result = differ.compare(df1, df2, key_columns=['id'])
        >>> print(result.summary)
    """
    
    def __init__(
        self, 
        ignore_index: bool = False, 
        ignore_column_order: bool = False,
        case_sensitive: bool = True,
        ignore_whitespace: bool = False,
        treat_null_as_equal: bool = True
    ):
        """
        Initialize DataFrameDiff
        
        Args:
            ignore_index: If True, ignore DataFrame index when comparing
            ignore_column_order: If True, ignore column order when comparing
            case_sensitive: If True, string comparisons are case-sensitive
            ignore_whitespace: If True, strip whitespace from strings before comparing
            treat_null_as_equal: If True, NaN/None values are considered equal
        """
        self.ignore_index = ignore_index
        self.ignore_column_order = ignore_column_order
        self.case_sensitive = case_sensitive
        self.ignore_whitespace = ignore_whitespace
        self.treat_null_as_equal = treat_null_as_equal
    
    def compare(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        key_columns: Optional[List[str]] = None,
        compare_columns: Optional[List[str]] = None,
        tolerance: float = 0.0,
        include_unchanged: bool = True,
        track_cell_changes: bool = True
    ) -> DiffResult:
        """
        Compare two DataFrames and return differences
        
        Args:
            df1: First DataFrame (considered as "before" / source)
            df2: Second DataFrame (considered as "after" / target)
            key_columns: Columns to use as unique identifiers for rows.
                        If None, uses index-based comparison.
            compare_columns: Columns to compare. If None, compare all common columns.
            tolerance: Tolerance for numerical comparisons (relative and absolute).
            include_unchanged: Whether to include unchanged rows in the result.
            track_cell_changes: Whether to track individual cell-level changes.
            
        Returns:
            DiffResult object containing all differences
            
        Raises:
            TypeError: If inputs are not pandas DataFrames
            ValueError: If key_columns or compare_columns don't exist in DataFrames
        """
        # Validate inputs
        self._validate_inputs(df1, df2, key_columns, compare_columns)
        
        # Handle empty DataFrames
        if len(df1) == 0 and len(df2) == 0:
            return self._create_empty_result(df1, df2)
        
        # Prepare DataFrames
        df1_prep, df2_prep = self._prepare_dataframes(df1, df2, compare_columns, key_columns)
        
        # Get schema differences
        schema_diff = self._get_schema_diff(df1, df2)
        column_changes = schema_diff.to_dict()
        
        # If no key columns specified, use index-based comparison
        if key_columns is None:
            return self._compare_by_index(
                df1_prep, df2_prep, tolerance, column_changes, 
                schema_diff, include_unchanged, track_cell_changes
            )
        else:
            return self._compare_by_keys(
                df1_prep, df2_prep, key_columns, tolerance, column_changes,
                schema_diff, include_unchanged, track_cell_changes
            )
    
    def get_added_rows(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                       key_columns: Optional[List[str]] = None,
                       compare_columns: Optional[List[str]] = None,
                       tolerance: float = 0.0) -> pd.DataFrame:
        """Get only the added rows between two DataFrames"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.get_added()
    
    def get_removed_rows(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                         key_columns: Optional[List[str]] = None,
                         compare_columns: Optional[List[str]] = None,
                         tolerance: float = 0.0) -> pd.DataFrame:
        """Get only the removed rows between two DataFrames"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.get_removed()
    
    def get_modified_rows(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                          key_columns: Optional[List[str]] = None,
                          compare_columns: Optional[List[str]] = None,
                          tolerance: float = 0.0) -> pd.DataFrame:
        """Get only the modified rows between two DataFrames"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.get_modified()
    
    def get_unchanged_rows(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                           key_columns: Optional[List[str]] = None,
                           compare_columns: Optional[List[str]] = None,
                           tolerance: float = 0.0) -> pd.DataFrame:
        """Get only the unchanged rows between two DataFrames"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.get_unchanged()
    
    def get_all_changes(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                        key_columns: Optional[List[str]] = None,
                        compare_columns: Optional[List[str]] = None,
                        tolerance: float = 0.0) -> pd.DataFrame:
        """Get all changes (added + modified) between two DataFrames"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.get_changes()
    
    def get_all_differences(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                            key_columns: Optional[List[str]] = None,
                            compare_columns: Optional[List[str]] = None,
                            tolerance: float = 0.0) -> pd.DataFrame:
        """Get all differences (added + removed + modified) with change type"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.get_all_differences()
    
    def are_identical(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                      key_columns: Optional[List[str]] = None,
                      compare_columns: Optional[List[str]] = None,
                      tolerance: float = 0.0) -> bool:
        """Check if two DataFrames are identical"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return not result.has_changes()
    
    def has_changes(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                    key_columns: Optional[List[str]] = None,
                    compare_columns: Optional[List[str]] = None,
                    tolerance: float = 0.0) -> bool:
        """Check if there are any changes between DataFrames"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.has_changes()
    
    def has_added_rows(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                       key_columns: Optional[List[str]] = None,
                       compare_columns: Optional[List[str]] = None,
                       tolerance: float = 0.0) -> bool:
        """Check if there are any added rows"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.has_added_rows()
    
    def has_removed_rows(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                         key_columns: Optional[List[str]] = None,
                         compare_columns: Optional[List[str]] = None,
                         tolerance: float = 0.0) -> bool:
        """Check if there are any removed rows"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.has_removed_rows()
    
    def has_modified_rows(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                          key_columns: Optional[List[str]] = None,
                          compare_columns: Optional[List[str]] = None,
                          tolerance: float = 0.0) -> bool:
        """Check if there are any modified rows"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        return result.has_modified_rows()
    
    def print_summary(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                      key_columns: Optional[List[str]] = None,
                      compare_columns: Optional[List[str]] = None,
                      tolerance: float = 0.0) -> None:
        """Print a human-readable summary of the differences"""
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        print("DataFrame Comparison Summary")
        print("=" * 30)
        print(f"Total rows in DataFrame 1: {result.summary['total_rows_df1']}")
        print(f"Total rows in DataFrame 2: {result.summary['total_rows_df2']}")
        print(f"Added rows: {result.summary['added_rows']}")
        print(f"Removed rows: {result.summary['removed_rows']}")
        print(f"Modified rows: {result.summary['modified_rows']}")
        print(f"Unchanged rows: {result.summary['unchanged_rows']}")
        print(f"DataFrames identical: {result.summary['identical']}")
        
        if result.column_changes['added_columns']:
            print(f"Added columns: {result.column_changes['added_columns']}")
        if result.column_changes['removed_columns']:
            print(f"Removed columns: {result.column_changes['removed_columns']}")
    
    def export_differences(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        filename: str,
        key_columns: Optional[List[str]] = None,
        compare_columns: Optional[List[str]] = None,
        tolerance: float = 0.0,
        format: str = 'excel'
    ) -> Union[None, List[str]]:
        """
        Export differences to file
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            filename: Output filename (or base filename for CSV)
            key_columns: Key columns for comparison
            compare_columns: Columns to compare
            tolerance: Numerical tolerance
            format: Export format ('excel', 'csv', 'json', 'html')
            
        Returns:
            For CSV format, returns list of created files. None otherwise.
        """
        result = self.compare(df1, df2, key_columns, compare_columns, tolerance)
        
        format_lower = format.lower()
        if format_lower == 'excel':
            result.export_to_excel(filename)
        elif format_lower == 'csv':
            return result.export_to_csv(filename)
        elif format_lower == 'json':
            result.export_to_json(filename)
        elif format_lower == 'html':
            result.export_to_html(filename)
        else:
            raise ValueError("Format must be 'excel', 'csv', 'json', or 'html'")
        return None
    
    def compare_schema(self, df1: pd.DataFrame, df2: pd.DataFrame) -> SchemaDiff:
        """
        Compare only the schema (columns and dtypes) of two DataFrames
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            SchemaDiff object with schema differences
        """
        return self._get_schema_diff(df1, df2)
    
    def _create_empty_result(self, df1: pd.DataFrame, df2: pd.DataFrame) -> DiffResult:
        """Create an empty DiffResult for empty DataFrames"""
        schema_diff = self._get_schema_diff(df1, df2)
        return DiffResult(
            summary={
                'total_rows_df1': 0,
                'total_rows_df2': 0,
                'added_rows': 0,
                'removed_rows': 0,
                'modified_rows': 0,
                'unchanged_rows': 0,
                'identical': not schema_diff.has_changes()
            },
            added_rows=pd.DataFrame(),
            removed_rows=pd.DataFrame(),
            modified_rows=pd.DataFrame(),
            unchanged_rows=pd.DataFrame(),
            column_changes=schema_diff.to_dict(),
            cell_diffs=[],
            schema_diff=schema_diff
        )
    
    def _validate_inputs(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        key_columns: Optional[List[str]], 
        compare_columns: Optional[List[str]]
    ) -> None:
        """Validate input parameters"""
        if not isinstance(df1, pd.DataFrame):
            raise TypeError(f"First argument must be a pandas DataFrame, got {type(df1).__name__}")
        if not isinstance(df2, pd.DataFrame):
            raise TypeError(f"Second argument must be a pandas DataFrame, got {type(df2).__name__}")
        
        if key_columns is not None:
            if not isinstance(key_columns, (list, tuple)):
                raise TypeError("key_columns must be a list or tuple of column names")
            if len(key_columns) == 0:
                raise ValueError("key_columns cannot be empty. Use None for index-based comparison.")
            
            missing_keys_df1 = [col for col in key_columns if col not in df1.columns]
            missing_keys_df2 = [col for col in key_columns if col not in df2.columns]
            
            if missing_keys_df1:
                raise ValueError(f"Key columns {missing_keys_df1} not found in first DataFrame. "
                               f"Available columns: {list(df1.columns)}")
            if missing_keys_df2:
                raise ValueError(f"Key columns {missing_keys_df2} not found in second DataFrame. "
                               f"Available columns: {list(df2.columns)}")
            
            # Check for duplicate keys
            df1_duplicates = df1[key_columns].duplicated().sum()
            df2_duplicates = df2[key_columns].duplicated().sum()
            if df1_duplicates > 0:
                warnings.warn(f"First DataFrame has {df1_duplicates} duplicate key(s). "
                            "Only the first occurrence will be compared.")
            if df2_duplicates > 0:
                warnings.warn(f"Second DataFrame has {df2_duplicates} duplicate key(s). "
                            "Only the first occurrence will be compared.")
        
        if compare_columns is not None:
            if not isinstance(compare_columns, (list, tuple)):
                raise TypeError("compare_columns must be a list or tuple of column names")
            
            missing_cols_df1 = [col for col in compare_columns if col not in df1.columns]
            missing_cols_df2 = [col for col in compare_columns if col not in df2.columns]
            
            if missing_cols_df1:
                raise ValueError(f"Compare columns {missing_cols_df1} not found in first DataFrame")
            if missing_cols_df2:
                raise ValueError(f"Compare columns {missing_cols_df2} not found in second DataFrame")
    
    def _prepare_dataframes(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        compare_columns: Optional[List[str]],
        key_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare DataFrames for comparison"""
        df1_prep = df1.copy()
        df2_prep = df2.copy()
        
        # Reset index if ignoring it
        if self.ignore_index:
            df1_prep = df1_prep.reset_index(drop=True)
            df2_prep = df2_prep.reset_index(drop=True)
        
        # Select columns to compare
        if compare_columns:
            common_cols = list(compare_columns)
            # Ensure key columns are included even if not in compare_columns
            if key_columns:
                for key_col in key_columns:
                    if key_col not in common_cols:
                        common_cols.insert(0, key_col)
        else:
            common_cols = list(set(df1_prep.columns) & set(df2_prep.columns))
        
        # Reorder columns if ignoring column order
        if self.ignore_column_order:
            common_cols = sorted(common_cols)
        
        # Ensure we have at least some columns to compare
        if len(common_cols) == 0:
            warnings.warn("No common columns found between DataFrames")
            return pd.DataFrame(), pd.DataFrame()
        
        # Apply string preprocessing if needed
        df1_result = df1_prep[common_cols].copy()
        df2_result = df2_prep[common_cols].copy()
        
        if not self.case_sensitive or self.ignore_whitespace:
            for col in common_cols:
                if df1_result[col].dtype == 'object':
                    if self.ignore_whitespace:
                        df1_result[col] = df1_result[col].apply(
                            lambda x: x.strip() if isinstance(x, str) else x
                        )
                        df2_result[col] = df2_result[col].apply(
                            lambda x: x.strip() if isinstance(x, str) else x
                        )
                    if not self.case_sensitive:
                        df1_result[col] = df1_result[col].apply(
                            lambda x: x.lower() if isinstance(x, str) else x
                        )
                        df2_result[col] = df2_result[col].apply(
                            lambda x: x.lower() if isinstance(x, str) else x
                        )
        
        return df1_result, df2_result
    
    def _get_schema_diff(self, df1: pd.DataFrame, df2: pd.DataFrame) -> SchemaDiff:
        """Get comprehensive schema differences between DataFrames"""
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        common_cols = cols1 & cols2
        
        # Check for dtype changes in common columns
        dtype_changes = {}
        for col in common_cols:
            dtype1 = str(df1[col].dtype)
            dtype2 = str(df2[col].dtype)
            if dtype1 != dtype2:
                dtype_changes[col] = (dtype1, dtype2)
        
        return SchemaDiff(
            added_columns=list(cols2 - cols1),
            removed_columns=list(cols1 - cols2),
            common_columns=list(common_cols),
            dtype_changes=dtype_changes,
            column_order_changed=list(df1.columns) != list(df2.columns) if not self.ignore_column_order else False
        )
    
    def _compare_by_index(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        tolerance: float, 
        column_changes: Dict[str, Any],
        schema_diff: SchemaDiff,
        include_unchanged: bool,
        track_cell_changes: bool
    ) -> DiffResult:
        """Compare DataFrames using index-based matching"""
        # Handle empty prepared DataFrames
        if len(df1.columns) == 0 or len(df2.columns) == 0:
            return self._create_empty_result(df1, df2)
        
        # Get all unique indices
        all_idx = df1.index.union(df2.index)
        idx1_set = set(df1.index)
        idx2_set = set(df2.index)
        
        # Added rows (exist in df2 but not df1)
        added_idx = [idx for idx in all_idx if idx in idx2_set and idx not in idx1_set]
        added_rows = df2.loc[added_idx] if added_idx else pd.DataFrame(columns=df2.columns)
        
        # Removed rows (exist in df1 but not df2)
        removed_idx = [idx for idx in all_idx if idx in idx1_set and idx not in idx2_set]
        removed_rows = df1.loc[removed_idx] if removed_idx else pd.DataFrame(columns=df1.columns)
        
        # Find common indices
        common_idx = [idx for idx in all_idx if idx in idx1_set and idx in idx2_set]
        
        # Modified and unchanged rows
        modified_idx = []
        unchanged_idx = []
        cell_diffs = []
        detailed_mods = []
        
        for idx in common_idx:
            row1 = df1.loc[idx]
            row2 = df2.loc[idx]
            
            if self._rows_equal(row1, row2, tolerance):
                unchanged_idx.append(idx)
            else:
                modified_idx.append(idx)
                
                # Track cell-level changes
                if track_cell_changes:
                    for col in df1.columns:
                        val1 = row1[col]
                        val2 = row2[col]
                        if not self._values_equal(val1, val2, tolerance):
                            cell_diffs.append(CellDiff(
                                row_key=idx,
                                column=col,
                                old_value=val1,
                                new_value=val2,
                                diff_type=DiffType.MODIFIED
                            ))
                            detailed_mods.append({
                                'row_key': idx,
                                'column': col,
                                'old_value': val1,
                                'new_value': val2
                            })
        
        modified_df = df2.loc[modified_idx] if modified_idx else pd.DataFrame(columns=df2.columns)
        unchanged_df = df2.loc[unchanged_idx] if (include_unchanged and unchanged_idx) else pd.DataFrame(columns=df2.columns)
        detailed_modifications = pd.DataFrame(detailed_mods) if detailed_mods else None
        
        # Create summary
        summary = {
            'total_rows_df1': len(df1),
            'total_rows_df2': len(df2),
            'added_rows': len(added_rows),
            'removed_rows': len(removed_rows),
            'modified_rows': len(modified_df),
            'unchanged_rows': len(unchanged_idx),
            'total_cell_changes': len(cell_diffs),
            'identical': len(modified_df) == 0 and len(added_rows) == 0 and len(removed_rows) == 0
        }
        
        return DiffResult(
            summary=summary,
            added_rows=added_rows,
            removed_rows=removed_rows,
            modified_rows=modified_df,
            unchanged_rows=unchanged_df,
            column_changes=column_changes,
            cell_diffs=cell_diffs,
            schema_diff=schema_diff,
            _detailed_modifications=detailed_modifications
        )
    
    def _compare_by_keys(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        key_columns: List[str], 
        tolerance: float, 
        column_changes: Dict[str, Any],
        schema_diff: SchemaDiff,
        include_unchanged: bool,
        track_cell_changes: bool
    ) -> DiffResult:
        """Compare DataFrames using specified key columns"""
        # Handle empty prepared DataFrames
        if len(df1.columns) == 0 or len(df2.columns) == 0:
            return self._create_empty_result(df1, df2)
        
        # Create composite keys - handle both single and multiple key columns
        def make_key(row):
            if len(key_columns) == 1:
                return row[key_columns[0]]
            return tuple(row[col] for col in key_columns)
        
        df1_keys = df1.apply(make_key, axis=1)
        df2_keys = df2.apply(make_key, axis=1)
        
        # Create lookup dictionaries for faster access
        df1_lookup = {}
        for idx, key in df1_keys.items():
            if key not in df1_lookup:  # Only keep first occurrence
                df1_lookup[key] = idx
        
        df2_lookup = {}
        for idx, key in df2_keys.items():
            if key not in df2_lookup:  # Only keep first occurrence
                df2_lookup[key] = idx
        
        # Find different types of keys
        keys1_set = set(df1_lookup.keys())
        keys2_set = set(df2_lookup.keys())
        
        common_keys = keys1_set & keys2_set
        added_keys = keys2_set - keys1_set
        removed_keys = keys1_set - keys2_set
        
        # Get rows for each category
        added_idx = [df2_lookup[k] for k in added_keys]
        removed_idx = [df1_lookup[k] for k in removed_keys]
        
        added_rows = df2.loc[added_idx] if added_idx else pd.DataFrame(columns=df2.columns)
        removed_rows = df1.loc[removed_idx] if removed_idx else pd.DataFrame(columns=df1.columns)
        
        # For common keys, check if values changed
        modified_idx = []
        unchanged_idx = []
        cell_diffs = []
        detailed_mods = []
        
        # Get columns to compare (excluding key columns)
        compare_cols = [col for col in df1.columns if col not in key_columns]
        
        for key in common_keys:
            idx1 = df1_lookup[key]
            idx2 = df2_lookup[key]
            row1 = df1.loc[idx1]
            row2 = df2.loc[idx2]
            
            if len(compare_cols) == 0 or self._rows_equal(row1[compare_cols], row2[compare_cols], tolerance):
                unchanged_idx.append(idx2)
            else:
                modified_idx.append(idx2)
                
                # Track cell-level changes
                if track_cell_changes:
                    for col in compare_cols:
                        val1 = row1[col]
                        val2 = row2[col]
                        if not self._values_equal(val1, val2, tolerance):
                            cell_diffs.append(CellDiff(
                                row_key=key,
                                column=col,
                                old_value=val1,
                                new_value=val2,
                                diff_type=DiffType.MODIFIED
                            ))
                            detailed_mods.append({
                                'row_key': str(key),
                                'column': col,
                                'old_value': val1,
                                'new_value': val2
                            })
        
        modified_df = df2.loc[modified_idx] if modified_idx else pd.DataFrame(columns=df2.columns)
        unchanged_df = df2.loc[unchanged_idx] if (include_unchanged and unchanged_idx) else pd.DataFrame(columns=df2.columns)
        detailed_modifications = pd.DataFrame(detailed_mods) if detailed_mods else None
        
        # Create summary
        summary = {
            'total_rows_df1': len(df1),
            'total_rows_df2': len(df2),
            'added_rows': len(added_rows),
            'removed_rows': len(removed_rows),
            'modified_rows': len(modified_df),
            'unchanged_rows': len(unchanged_idx),
            'total_cell_changes': len(cell_diffs),
            'identical': len(modified_df) == 0 and len(added_rows) == 0 and len(removed_rows) == 0
        }
        
        return DiffResult(
            summary=summary,
            added_rows=added_rows,
            removed_rows=removed_rows,
            modified_rows=modified_df,
            unchanged_rows=unchanged_df,
            column_changes=column_changes,
            cell_diffs=cell_diffs,
            schema_diff=schema_diff,
            _detailed_modifications=detailed_modifications
        )
    
    def _values_equal(self, val1: Any, val2: Any, tolerance: float) -> bool:
        """Check if two individual values are equal"""
        # Handle None/NaN
        val1_null = pd.isna(val1)
        val2_null = pd.isna(val2)
        
        if val1_null and val2_null:
            return self.treat_null_as_equal
        if val1_null or val2_null:
            return False
        
        # Numeric comparison with tolerance
        if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
            if tolerance > 0:
                return np.isclose(val1, val2, rtol=tolerance, atol=tolerance)
            return val1 == val2
        
        # Direct comparison for other types
        try:
            return val1 == val2
        except (ValueError, TypeError):
            return str(val1) == str(val2)
    
    def _rows_equal(self, row1: pd.Series, row2: pd.Series, tolerance: float) -> bool:
        """Check if two rows are equal within tolerance"""
        try:
            # Handle empty rows
            if len(row1) == 0 and len(row2) == 0:
                return True
            if len(row1) != len(row2):
                return False
            
            for col in row1.index:
                if not self._values_equal(row1[col], row2[col], tolerance):
                    return False
            return True
        except Exception:
            # Fallback to direct comparison
            try:
                return row1.equals(row2)
            except Exception:
                return False


# Convenience functions for quick comparisons
def compare_dataframes(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    key_columns: Optional[List[str]] = None,
    tolerance: float = 0.0,
    **kwargs
) -> DiffResult:
    """
    Convenience function to compare two DataFrames.
    
    Args:
        df1: First DataFrame (before)
        df2: Second DataFrame (after)
        key_columns: Columns to use as unique identifiers
        tolerance: Numerical tolerance for comparisons
        **kwargs: Additional arguments passed to DataFrameDiff
        
    Returns:
        DiffResult object with comparison results
        
    Example:
        >>> result = compare_dataframes(df1, df2, key_columns=['id'])
        >>> print(result.summary)
    """
    differ = DataFrameDiff(**kwargs)
    return differ.compare(df1, df2, key_columns=key_columns, tolerance=tolerance)


def quick_diff(df1: pd.DataFrame, df2: pd.DataFrame, key_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get a quick summary of differences between two DataFrames.
    
    Returns a dictionary with counts of added, removed, and modified rows.
    
    Example:
        >>> summary = quick_diff(df1, df2, key_columns=['id'])
        >>> print(summary)
        {'added': 1, 'removed': 1, 'modified': 1, 'unchanged': 2, 'identical': False}
    """
    result = compare_dataframes(df1, df2, key_columns=key_columns)
    return {
        'added': result.summary['added_rows'],
        'removed': result.summary['removed_rows'],
        'modified': result.summary['modified_rows'],
        'unchanged': result.summary['unchanged_rows'],
        'identical': result.summary['identical']
    }


def are_dataframes_equal(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    key_columns: Optional[List[str]] = None,
    tolerance: float = 0.0
) -> bool:
    """
    Check if two DataFrames are equal.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        key_columns: Optional key columns for row matching
        tolerance: Numerical tolerance
        
    Returns:
        True if DataFrames are identical, False otherwise
    """
    result = compare_dataframes(df1, df2, key_columns=key_columns, tolerance=tolerance)
    return result.summary['identical']


# Example usage and testing
if __name__ == "__main__":
    # Create sample DataFrames for testing
    df1 = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 40],
        'city': ['NY', 'LA', 'Chicago', 'Miami']
    })
    
    df2 = pd.DataFrame({
        'id': [1, 2, 3, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Eve'],
        'age': [26, 30, 35, 28],
        'city': ['NY', 'LA', 'Chicago', 'Seattle']
    })
    
    print("=" * 60)
    print("DataDiff Library Demo")
    print("=" * 60)
    
    # Example 1: Basic class usage
    print("\n📊 Example 1: Basic Comparison")
    print("-" * 40)
    differ = DataFrameDiff()
    result = differ.compare(df1, df2, key_columns=['id'])
    print(f"Result: {result}")
    print(f"\nSummary:")
    for key, value in result.summary.items():
        print(f"  {key}: {value}")
    
    # Example 2: Get specific row types
    print("\n📊 Example 2: Row Categories")
    print("-" * 40)
    print(f"Added rows:\n{result.get_added()}")
    print(f"\nRemoved rows:\n{result.get_removed()}")
    print(f"\nModified rows:\n{result.get_modified()}")
    
    # Example 3: Cell-level differences
    print("\n📊 Example 3: Cell-Level Differences")
    print("-" * 40)
    for cell_diff in result.get_cell_diffs():
        print(f"  Row {cell_diff.row_key}, Column '{cell_diff.column}': "
              f"{cell_diff.old_value!r} → {cell_diff.new_value!r}")
    
    # Example 4: Convenience functions
    print("\n📊 Example 4: Quick Diff Function")
    print("-" * 40)
    summary = quick_diff(df1, df2, key_columns=['id'])
    print(f"Quick summary: {summary}")
    
    # Example 5: Tolerance comparison
    print("\n📊 Example 5: Numerical Tolerance")
    print("-" * 40)
    df_float1 = pd.DataFrame({'value': [1.001, 2.002, 3.003]})
    df_float2 = pd.DataFrame({'value': [1.000, 2.000, 3.000]})
    
    print(f"Strict comparison: {are_dataframes_equal(df_float1, df_float2, tolerance=0.0)}")
    print(f"Tolerant (0.01): {are_dataframes_equal(df_float1, df_float2, tolerance=0.01)}")
    
    # Example 6: Case-insensitive comparison
    print("\n📊 Example 6: Case-Insensitive Comparison")
    print("-" * 40)
    df_case1 = pd.DataFrame({'name': ['Alice', 'BOB', 'Charlie']})
    df_case2 = pd.DataFrame({'name': ['alice', 'bob', 'charlie']})
    
    differ_strict = DataFrameDiff(case_sensitive=True)
    differ_loose = DataFrameDiff(case_sensitive=False)
    
    print(f"Case-sensitive: {differ_strict.are_identical(df_case1, df_case2)}")
    print(f"Case-insensitive: {differ_loose.are_identical(df_case1, df_case2)}")
    
    # Example 7: Schema comparison
    print("\n📊 Example 7: Schema Comparison")
    print("-" * 40)
    df_schema1 = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    df_schema2 = pd.DataFrame({'a': [1], 'b': ['x'], 'd': [4]})  # 'b' type changed, 'c' removed, 'd' added
    
    schema_diff = differ.compare_schema(df_schema1, df_schema2)
    print(f"Added columns: {schema_diff.added_columns}")
    print(f"Removed columns: {schema_diff.removed_columns}")
    print(f"Dtype changes: {schema_diff.dtype_changes}")
    
    # Example 8: Export options
    print("\n📊 Example 8: Export Options")
    print("-" * 40)
    print("Available export formats: Excel, CSV, JSON, HTML")
    print("Use result.export_to_excel('output.xlsx')")
    print("Use result.export_to_csv('output')")
    print("Use result.export_to_json('output.json')")
    print("Use result.export_to_html('output.html')")
    
    # Example 9: JSON output
    print("\n📊 Example 9: JSON Output")
    print("-" * 40)
    json_output = result.to_json()
    print(f"JSON length: {len(json_output)} characters")
    print("(Use result.to_json() to get full JSON string)")
    
    print("\n" + "=" * 60)
    print("✅ DataDiff Library Ready for Use!")
    print("=" * 60)