"""
Tests for the Vinzy DataDiff library
"""

import pytest
import pandas as pd
import numpy as np
from vinzy_datadiff import (
    DataFrameDiff,
    DiffResult,
    DiffType,
    CellDiff,
    SchemaDiff,
    compare_dataframes,
    quick_diff,
    are_dataframes_equal,
)


class TestDataFrameDiff:
    """Tests for the DataFrameDiff class"""
    
    @pytest.fixture
    def sample_dfs(self):
        """Create sample DataFrames for testing"""
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
        
        return df1, df2
    
    def test_basic_comparison(self, sample_dfs):
        """Test basic DataFrame comparison"""
        df1, df2 = sample_dfs
        differ = DataFrameDiff()
        result = differ.compare(df1, df2, key_columns=['id'])
        
        assert isinstance(result, DiffResult)
        assert result.summary['total_rows_df1'] == 4
        assert result.summary['total_rows_df2'] == 4
        assert result.summary['added_rows'] == 1
        assert result.summary['removed_rows'] == 1
        assert result.summary['modified_rows'] == 1
        assert result.summary['unchanged_rows'] == 2
        assert result.summary['identical'] == False
    
    def test_identical_dataframes(self):
        """Test comparison of identical DataFrames"""
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        differ = DataFrameDiff()
        result = differ.compare(df, df.copy(), key_columns=['id'])
        
        assert result.summary['identical'] == True
        assert result.summary['added_rows'] == 0
        assert result.summary['removed_rows'] == 0
        assert result.summary['modified_rows'] == 0
        assert result.summary['unchanged_rows'] == 3
    
    def test_empty_dataframes(self):
        """Test comparison of empty DataFrames"""
        df1 = pd.DataFrame(columns=['id', 'value'])
        df2 = pd.DataFrame(columns=['id', 'value'])
        differ = DataFrameDiff()
        result = differ.compare(df1, df2)
        
        assert result.summary['total_rows_df1'] == 0
        assert result.summary['total_rows_df2'] == 0
        assert result.summary['identical'] == True
    
    def test_added_rows_only(self):
        """Test when rows are only added"""
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        df2 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        differ = DataFrameDiff()
        result = differ.compare(df1, df2, key_columns=['id'])
        
        assert result.summary['added_rows'] == 1
        assert result.summary['removed_rows'] == 0
        assert result.summary['modified_rows'] == 0
        assert len(result.get_added()) == 1
        assert result.get_added()['id'].iloc[0] == 3
    
    def test_removed_rows_only(self):
        """Test when rows are only removed"""
        df1 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        df2 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        differ = DataFrameDiff()
        result = differ.compare(df1, df2, key_columns=['id'])
        
        assert result.summary['added_rows'] == 0
        assert result.summary['removed_rows'] == 1
        assert result.summary['modified_rows'] == 0
        assert len(result.get_removed()) == 1
        assert result.get_removed()['id'].iloc[0] == 3
    
    def test_modified_rows_only(self):
        """Test when rows are only modified"""
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        df2 = pd.DataFrame({'id': [1, 2], 'value': [15, 25]})
        differ = DataFrameDiff()
        result = differ.compare(df1, df2, key_columns=['id'])
        
        assert result.summary['added_rows'] == 0
        assert result.summary['removed_rows'] == 0
        assert result.summary['modified_rows'] == 2
        assert result.summary['unchanged_rows'] == 0
    
    def test_tolerance_comparison(self):
        """Test numerical tolerance in comparison"""
        df1 = pd.DataFrame({'id': [1, 2], 'value': [1.001, 2.002]})
        df2 = pd.DataFrame({'id': [1, 2], 'value': [1.000, 2.000]})
        
        # Without tolerance
        differ = DataFrameDiff()
        result_strict = differ.compare(df1, df2, key_columns=['id'], tolerance=0.0)
        assert result_strict.summary['modified_rows'] == 2
        
        # With tolerance
        result_tolerant = differ.compare(df1, df2, key_columns=['id'], tolerance=0.01)
        assert result_tolerant.summary['modified_rows'] == 0
        assert result_tolerant.summary['identical'] == True
    
    def test_case_insensitive_comparison(self):
        """Test case-insensitive string comparison"""
        df1 = pd.DataFrame({'id': [1], 'name': ['Alice']})
        df2 = pd.DataFrame({'id': [1], 'name': ['ALICE']})
        
        # Case-sensitive (default)
        differ_sensitive = DataFrameDiff(case_sensitive=True)
        result = differ_sensitive.compare(df1, df2, key_columns=['id'])
        assert result.summary['modified_rows'] == 1
        
        # Case-insensitive
        differ_insensitive = DataFrameDiff(case_sensitive=False)
        result = differ_insensitive.compare(df1, df2, key_columns=['id'])
        assert result.summary['modified_rows'] == 0
    
    def test_whitespace_handling(self):
        """Test whitespace handling in comparison"""
        df1 = pd.DataFrame({'id': [1], 'name': ['Alice']})
        df2 = pd.DataFrame({'id': [1], 'name': ['  Alice  ']})
        
        # With whitespace consideration (default)
        differ_strict = DataFrameDiff(ignore_whitespace=False)
        result = differ_strict.compare(df1, df2, key_columns=['id'])
        assert result.summary['modified_rows'] == 1
        
        # Ignoring whitespace
        differ_loose = DataFrameDiff(ignore_whitespace=True)
        result = differ_loose.compare(df1, df2, key_columns=['id'])
        assert result.summary['modified_rows'] == 0
    
    def test_cell_level_tracking(self, sample_dfs):
        """Test cell-level change tracking"""
        df1, df2 = sample_dfs
        differ = DataFrameDiff()
        result = differ.compare(df1, df2, key_columns=['id'], track_cell_changes=True)
        
        cell_diffs = result.get_cell_diffs()
        assert len(cell_diffs) > 0
        assert all(isinstance(cd, CellDiff) for cd in cell_diffs)
        
        # Check cell diff DataFrame
        cell_df = result.get_cell_diffs_df()
        assert 'column' in cell_df.columns
        assert 'old_value' in cell_df.columns
        assert 'new_value' in cell_df.columns
    
    def test_schema_comparison(self):
        """Test schema comparison"""
        df1 = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        df2 = pd.DataFrame({'a': [1], 'b': ['x'], 'd': [4]})
        
        differ = DataFrameDiff()
        schema_diff = differ.compare_schema(df1, df2)
        
        assert isinstance(schema_diff, SchemaDiff)
        assert 'd' in schema_diff.added_columns
        assert 'c' in schema_diff.removed_columns
        assert 'b' in schema_diff.dtype_changes
        assert schema_diff.has_changes() == True
    
    def test_index_based_comparison(self):
        """Test comparison using DataFrame index"""
        df1 = pd.DataFrame({'value': [10, 20, 30]}, index=['a', 'b', 'c'])
        df2 = pd.DataFrame({'value': [10, 25, 30]}, index=['a', 'b', 'd'])
        
        differ = DataFrameDiff()
        result = differ.compare(df1, df2)  # No key_columns, uses index
        
        assert result.summary['added_rows'] == 1  # 'd'
        assert result.summary['removed_rows'] == 1  # 'c'
        assert result.summary['modified_rows'] == 1  # 'b'
    
    def test_multiple_key_columns(self):
        """Test comparison with multiple key columns"""
        df1 = pd.DataFrame({
            'region': ['US', 'US', 'EU'],
            'product': ['A', 'B', 'A'],
            'sales': [100, 200, 150]
        })
        df2 = pd.DataFrame({
            'region': ['US', 'US', 'EU'],
            'product': ['A', 'B', 'A'],
            'sales': [110, 200, 150]
        })
        
        differ = DataFrameDiff()
        result = differ.compare(df1, df2, key_columns=['region', 'product'])
        
        assert result.summary['modified_rows'] == 1  # US-A modified
        assert result.summary['unchanged_rows'] == 2
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        differ = DataFrameDiff()
        
        # Non-DataFrame inputs
        with pytest.raises(TypeError):
            differ.compare([1, 2, 3], pd.DataFrame())
        
        # Missing key columns
        df = pd.DataFrame({'a': [1, 2]})
        with pytest.raises(ValueError):
            differ.compare(df, df, key_columns=['missing'])
    
    def test_to_dict(self, sample_dfs):
        """Test conversion to dictionary"""
        df1, df2 = sample_dfs
        differ = DataFrameDiff()
        result = differ.compare(df1, df2, key_columns=['id'])
        
        result_dict = result.to_dict()
        assert 'summary' in result_dict
        assert 'added_rows' in result_dict
        assert 'removed_rows' in result_dict
        assert 'modified_rows' in result_dict
    
    def test_to_json(self, sample_dfs):
        """Test JSON serialization"""
        df1, df2 = sample_dfs
        differ = DataFrameDiff()
        result = differ.compare(df1, df2, key_columns=['id'])
        
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert 'summary' in json_str


class TestConvenienceFunctions:
    """Tests for convenience functions"""
    
    @pytest.fixture
    def sample_dfs(self):
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        df2 = pd.DataFrame({'id': [1, 2], 'value': [10, 25]})
        return df1, df2
    
    def test_compare_dataframes(self, sample_dfs):
        """Test compare_dataframes function"""
        df1, df2 = sample_dfs
        result = compare_dataframes(df1, df2, key_columns=['id'])
        
        assert isinstance(result, DiffResult)
        assert result.summary['modified_rows'] == 1
    
    def test_quick_diff(self, sample_dfs):
        """Test quick_diff function"""
        df1, df2 = sample_dfs
        summary = quick_diff(df1, df2, key_columns=['id'])
        
        assert isinstance(summary, dict)
        assert 'added' in summary
        assert 'removed' in summary
        assert 'modified' in summary
        assert 'unchanged' in summary
        assert 'identical' in summary
        assert summary['modified'] == 1
    
    def test_are_dataframes_equal(self):
        """Test are_dataframes_equal function"""
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        df2 = df1.copy()
        df3 = pd.DataFrame({'id': [1, 2], 'value': [10, 25]})
        
        assert are_dataframes_equal(df1, df2, key_columns=['id']) == True
        assert are_dataframes_equal(df1, df3, key_columns=['id']) == False


class TestDiffResult:
    """Tests for DiffResult class methods"""
    
    @pytest.fixture
    def sample_result(self):
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        df2 = pd.DataFrame({
            'id': [1, 2, 4],
            'value': [15, 20, 40]
        })
        differ = DataFrameDiff()
        return differ.compare(df1, df2, key_columns=['id'])
    
    def test_has_methods(self, sample_result):
        """Test has_* boolean methods"""
        assert sample_result.has_changes() == True
        assert sample_result.has_added_rows() == True
        assert sample_result.has_removed_rows() == True
        assert sample_result.has_modified_rows() == True
    
    def test_get_changes(self, sample_result):
        """Test get_changes method"""
        changes = sample_result.get_changes()
        # Should include added and modified
        assert len(changes) == 2  # 1 added + 1 modified
    
    def test_get_all_differences(self, sample_result):
        """Test get_all_differences method"""
        all_diffs = sample_result.get_all_differences()
        assert '_change_type' in all_diffs.columns
        assert len(all_diffs) == 3  # 1 added + 1 removed + 1 modified


class TestNullHandling:
    """Tests for NULL/NaN value handling"""
    
    def test_null_as_equal(self):
        """Test treating NaN values as equal"""
        df1 = pd.DataFrame({'id': [1, 2], 'value': [np.nan, 20]})
        df2 = pd.DataFrame({'id': [1, 2], 'value': [np.nan, 20]})
        
        differ = DataFrameDiff(treat_null_as_equal=True)
        result = differ.compare(df1, df2, key_columns=['id'])
        
        assert result.summary['identical'] == True
    
    def test_null_as_different(self):
        """Test treating NaN values as different"""
        df1 = pd.DataFrame({'id': [1], 'value': [np.nan]})
        df2 = pd.DataFrame({'id': [1], 'value': [np.nan]})
        
        differ = DataFrameDiff(treat_null_as_equal=False)
        result = differ.compare(df1, df2, key_columns=['id'])
        
        # When treat_null_as_equal=False, NaN != NaN
        assert result.summary['modified_rows'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
