import pandas as pd
import Levenshtein
import re
from typing import Dict, List, Any

def normalize_text(text: Any) -> str:
    """
    Normalize text for comparison:
    - Convert to lowercase
    - Remove punctuation and special characters (keep alphanumeric)
    - Strip whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def calculate_similarity(val1: Any, val2: Any) -> float:
    """
    Calculate similarity score between two values (0 to 100).
    Uses Levenshtein ratio.
    """
    s1 = normalize_text(val1)
    s2 = normalize_text(val2)
    
    if not s1 and not s2:
        return 100.0
    if not s1 or not s2:
        return 0.0
        
    return Levenshtein.ratio(s1, s2) * 100

def compare_ocr_with_reference(ocr_data: Dict[str, Any], reference_file_path: str) -> Dict[str, Any]:
    """
    Compare OCR extracted fields with a reference Excel/CSV file.
    
    Strategy:
    1. Load reference file.
    2. Try to find the 'best match' row in the reference file.
       - The best match is determined by the highest average similarity across all common keys.
       - A key identifier (like 'Certificate No' or 'Unique Doc Reference') is prioritized if found.
    3. Calculate match percentage for that best row.
    
    Args:
        ocr_data: Dictionary of extracted fields from OCR.
        reference_file_path: Path to the uploaded reference file.
        
    Returns:
        Dictionary containing:
        - match_score: Overall percentage match.
        - matched_row_index: Index of the matching row in reference file.
        - field_comparisons: Detailed comparison for each field.
    """
    # 1. Load Reference File
    try:
        if reference_file_path.endswith('.csv'):
            df = pd.read_csv(reference_file_path)
        else:
            df = pd.read_excel(reference_file_path)
    except Exception as e:
        return {"error": f"Failed to load reference file: {str(e)}"}

    # Normalize column names for easier mapping
    # Map common variations to standard keys (similar to OCR keys)
    column_map = {}
    for col in df.columns:
        norm_col = col.lower().strip().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
        column_map[norm_col] = col # store original name

    # 2. Identify Comparison Keys
    # We only compare keys that exist in both OCR data and Reference File columns (roughly)
    common_keys = []
    ocr_keys_normalized = {}
    
    for key, value in ocr_data.items():
        if isinstance(value, (dict, list)): continue # Skip nested complex structures for now
        
        norm_key = key.lower().strip().replace(' ', '_')
        ocr_keys_normalized[norm_key] = key
        
        # Fuzzy match key to column names
        best_col_match = None
        best_col_score = 0
        
        for map_key in column_map.keys():
            score = Levenshtein.ratio(norm_key, map_key)
            if score > 0.85: # High threshold for column mapping
                if score > best_col_score:
                    best_col_score = score
                    best_col_match = map_key
        
        if best_col_match:
            common_keys.append((key, column_map[best_col_match]))

    if not common_keys:
        return {
            "error": "No common fields found between OCR data and Reference file to compare.",
            "available_ocr_keys": list(ocr_data.keys()),
            "available_ref_columns": list(df.columns)
        }

    # 3. Find Best Matching Row
    # We iterate through all rows and calculate a score based on 'Certificate No' or avg similarity
    
    best_row_idx = -1
    best_row_average_score = -1
    best_field_details = []
    
    # Priority Key for anchoring (e.g., Certificate No)
    priority_keys = ['certificate_no', 'unique_doc_reference', 'account_reference']
    anchor_col = None
    anchor_ocr_key = None
    
    # Check if we have an anchor
    for p_key in priority_keys:
        for o_key, c_name in common_keys:
            if p_key in o_key.lower():
                anchor_col = c_name
                anchor_ocr_key = o_key
                break
        if anchor_col: break
    
    # Optimization: Filter by anchor if possible
    search_df = df
    if anchor_col and anchor_ocr_key in ocr_data:
        anchor_val = normalize_text(ocr_data[anchor_ocr_key])
        # Simple exact contains check for filtering to speed up
        # (Can be removed if dataset is small)
        pass

    for idx, row in search_df.iterrows():
        row_scores = []
        field_details = []
        
        for ocr_key, col_name in common_keys:
            ocr_val = ocr_data.get(ocr_key, "")
            ref_val = row[col_name]
            
            score = calculate_similarity(ocr_val, ref_val)
            row_scores.append(score)
            
            field_details.append({
                "field": ocr_key,
                "reference_column": col_name,
                "ocr_value": ocr_val,
                "reference_value": str(ref_val) if pd.notna(ref_val) else "",
                "match_score": round(score, 1),
                "is_match": score > 90 # Threshold for boolean match
            })
            
        avg_score = sum(row_scores) / len(row_scores) if row_scores else 0
        
        # Bonus for anchor match
        if anchor_col:
             # Find the detail for anchor
             anchor_detail = next((d for d in field_details if d['reference_column'] == anchor_col), None)
             if anchor_detail and anchor_detail['is_match']:
                 avg_score += 20 # heavily weight the anchor
        
        if avg_score > best_row_average_score:
            best_row_average_score = avg_score
            best_row_idx = idx
            best_field_details = field_details

    # 4. Final Result Compilation
    
    # Normalize score back to 0-100 max even with bonus
    final_score = min(100, best_row_average_score) 
    # Recalculate pure average for display without bonus
    pure_scores = [f['match_score'] for f in best_field_details]
    pure_avg = sum(pure_scores) / len(pure_scores) if pure_scores else 0

    return {
        "match_found": best_row_idx != -1 and pure_avg > 50, # Arbitrary threshold for "Found"
        "overall_match_score": round(pure_avg, 1),
        "matched_row_index": best_row_idx,
        "details": best_field_details,
        "total_fields_compared": len(common_keys)
    }
