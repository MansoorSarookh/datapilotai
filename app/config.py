"""
Application configuration and constants.
"""

# Supported file formats
SUPPORTED_FORMATS = {
    'csv': 'CSV (Comma Separated Values)',
    'xlsx': 'Excel Workbook',
    'xls': 'Excel 97-2003',
    'ods': 'OpenDocument Spreadsheet',
    'tsv': 'Tab Separated Values',
    'txt': 'Text File',
    'json': 'JSON File',
    'html': 'HTML Table',
    'pdf': 'PDF Document',
    'docx': 'Word Document',
    'parquet': 'Apache Parquet',
    'feather': 'Feather Format',
}

# File extensions by category
SPREADSHEET_FORMATS = ['.csv', '.xlsx', '.xls', '.ods', '.tsv', '.txt']
DOCUMENT_FORMATS = ['.pdf', '.docx', '.html']
DATA_FORMATS = ['.json', '.parquet', '.feather']

# Chart types
UNIVARIATE_CHARTS = [
    'Histogram',
    'Box Plot',
    'Violin Plot',
    'Bar Chart',
    'Pie Chart',
    'KDE Plot',
]

BIVARIATE_CHARTS = [
    'Scatter Plot',
    'Line Chart',
    'Grouped Bar Chart',
    'Correlation Heatmap',
    'Joint Plot',
]

MULTIVARIATE_CHARTS = [
    '3D Scatter Plot',
    'Parallel Coordinates',
    'Bubble Chart',
    'Sunburst Chart',
    'Treemap',
    'Pair Plot',
]

TIME_SERIES_CHARTS = [
    'Line with Range Slider',
    'Area Chart',
    'Candlestick Chart',
    'Animated Time Series',
    'Rolling Statistics',
]

# Color themes
PLOTLY_THEMES = [
    'plotly_dark',
    'plotly',
    'plotly_white',
    'ggplot2',
    'seaborn',
    'simple_white',
    'presentation',
]

# Default colors
DEFAULT_COLOR_PALETTE = [
    '#667eea',  # Indigo
    '#764ba2',  # Purple
    '#f093fb',  # Pink
    '#f5576c',  # Red
    '#4facfe',  # Blue
    '#00f2fe',  # Cyan
    '#43e97b',  # Green
    '#38f9d7',  # Teal
    '#fa709a',  # Rose
    '#fee140',  # Yellow
]

# App settings
APP_TITLE = "🧠 DataPilot AI — The AI Data Intelligence Copilot"
APP_ICON = "🧠"
MAX_UPLOAD_SIZE_MB = 500
MAX_ROWS_DISPLAY = 1000
SAMPLE_SIZE_FOR_PREVIEW = 10

# Export settings
EXPORT_FORMATS = {
    'png': 'PNG Image',
    'svg': 'SVG Vector',
    'html': 'Interactive HTML',
    'gif': 'Animated GIF',
}

# Statistical thresholds
CATEGORICAL_THRESHOLD = 20  # Max unique values to consider as categorical
HIGH_CARDINALITY_THRESHOLD = 100  # When to warn about high cardinality
MISSING_VALUE_THRESHOLD = 0.5  # 50% missing triggers warning
 
