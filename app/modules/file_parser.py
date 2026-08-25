"""
Multi-format file parser module.
Supports: CSV, Excel, ODS, TXT, JSON, HTML, PDF, DOCX, Parquet, Feather
"""

import pandas as pd
import io
from pathlib import Path
from typing import Optional, Tuple, Union
import chardet


def detect_encoding(file_content: bytes) -> str:
    """Detect file encoding using chardet."""
    result = chardet.detect(file_content)
    return result.get('encoding', 'utf-8') or 'utf-8'


def parse_csv(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Parse CSV file."""
    encoding = detect_encoding(file_content)
    
    # Try different delimiters
    for delimiter in [',', ';', '\t', '|']:
        try:
            df = pd.read_csv(
                io.BytesIO(file_content),
                encoding=encoding,
                delimiter=delimiter,
                **kwargs
            )
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    
    # Fallback with default delimiter
    return pd.read_csv(io.BytesIO(file_content), encoding=encoding, **kwargs)


def parse_excel(file_content: bytes, file_extension: str, **kwargs) -> pd.DataFrame:
    """Parse Excel file (xlsx or xls)."""
    engine = 'openpyxl' if file_extension == '.xlsx' else 'xlrd'
    return pd.read_excel(io.BytesIO(file_content), engine=engine, **kwargs)


def parse_ods(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Parse OpenDocument Spreadsheet."""
    return pd.read_excel(io.BytesIO(file_content), engine='odf', **kwargs)


def parse_json(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Parse JSON file."""
    encoding = detect_encoding(file_content)
    text = file_content.decode(encoding)
    
    # Try different orientations
    for orient in [None, 'records', 'columns', 'index', 'split', 'values']:
        try:
            if orient:
                df = pd.read_json(io.StringIO(text), orient=orient, **kwargs)
            else:
                df = pd.read_json(io.StringIO(text), **kwargs)
            if not df.empty:
                return df
        except Exception:
            continue
    
    return pd.read_json(io.StringIO(text), **kwargs)


def parse_html(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Parse HTML tables."""
    encoding = detect_encoding(file_content)
    text = file_content.decode(encoding)
    
    tables = pd.read_html(io.StringIO(text), **kwargs)
    if tables:
        # Return the largest table
        return max(tables, key=lambda t: t.size)
    raise ValueError("No tables found in HTML file")


def parse_pdf(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Parse PDF tables using pdfplumber (primary) or tabula (fallback)."""
    try:
        import pdfplumber
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            all_tables = []
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        all_tables.append(df)
            
            if all_tables:
                return pd.concat(all_tables, ignore_index=True)
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback to tabula
    try:
        import tabula
        
        tables = tabula.read_pdf(io.BytesIO(file_content), pages='all', **kwargs)
        if tables:
            return pd.concat(tables, ignore_index=True)
    except ImportError:
        raise ImportError("Please install pdfplumber or tabula-py for PDF support")
    except Exception as e:
        raise ValueError(f"Could not extract tables from PDF: {e}")
    
    raise ValueError("No tables found in PDF file")


def parse_docx(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Parse Word document tables."""
    try:
        from docx import Document
        
        doc = Document(io.BytesIO(file_content))
        all_tables = []
        
        for table in doc.tables:
            data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                data.append(row_data)
            
            if data:
                df = pd.DataFrame(data[1:], columns=data[0])
                all_tables.append(df)
        
        if all_tables:
            return pd.concat(all_tables, ignore_index=True)
        raise ValueError("No tables found in Word document")
    
    except ImportError:
        raise ImportError("Please install python-docx for Word document support")


def parse_parquet(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Parse Parquet file."""
    return pd.read_parquet(io.BytesIO(file_content), **kwargs)


def parse_feather(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Parse Feather file."""
    return pd.read_feather(io.BytesIO(file_content), **kwargs)


def parse_text(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Parse text file (TSV or space-separated)."""
    encoding = detect_encoding(file_content)
    text = file_content.decode(encoding)
    
    # Try tab-separated first
    try:
        df = pd.read_csv(io.StringIO(text), delimiter='\t', **kwargs)
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    
    # Try space-separated
    try:
        df = pd.read_csv(io.StringIO(text), delim_whitespace=True, **kwargs)
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    
    # Try comma
    return pd.read_csv(io.StringIO(text), **kwargs)


def parse_file(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Main entry point for file parsing.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (DataFrame, status_message)
    """
    if uploaded_file is None:
        raise ValueError("No file provided")
    
    file_name = uploaded_file.name
    file_extension = Path(file_name).suffix.lower()
    file_content = uploaded_file.read()
    
    parsers = {
        '.csv': lambda: parse_csv(file_content),
        '.xlsx': lambda: parse_excel(file_content, '.xlsx'),
        '.xls': lambda: parse_excel(file_content, '.xls'),
        '.ods': lambda: parse_ods(file_content),
        '.json': lambda: parse_json(file_content),
        '.html': lambda: parse_html(file_content),
        '.htm': lambda: parse_html(file_content),
        '.pdf': lambda: parse_pdf(file_content),
        '.docx': lambda: parse_docx(file_content),
        '.parquet': lambda: parse_parquet(file_content),
        '.feather': lambda: parse_feather(file_content),
        '.txt': lambda: parse_text(file_content),
        '.tsv': lambda: parse_text(file_content),
    }
    
    if file_extension not in parsers:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    try:
        df = parsers[file_extension]()
        rows, cols = df.shape
        status = f"✅ Successfully loaded {rows:,} rows × {cols} columns from {file_name}"
        return df, status
    except Exception as e:
        raise ValueError(f"Error parsing {file_name}: {str(e)}")


def get_supported_extensions() -> list:
    """Return list of supported file extensions."""
    return [
        'csv', 'xlsx', 'xls', 'ods', 'json', 'html', 'htm',
        'pdf', 'docx', 'parquet', 'feather', 'txt', 'tsv'
    ]
 
