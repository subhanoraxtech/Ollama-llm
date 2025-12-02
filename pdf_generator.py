"""
PDF Generator for Data Export
Generates professional PDF reports from pandas DataFrames
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import pandas as pd
from datetime import datetime
import io


def generate_pdf_report(df: pd.DataFrame, title: str = "Data Report", filename: str = None) -> bytes:
    """
    Generate a professional PDF report from a pandas DataFrame
    Intelligently handles wide datasets by selecting important columns
    
    Args:
        df: pandas DataFrame to export
        title: Title of the report
        filename: Optional filename (not used, returns bytes)
    
    Returns:
        bytes: PDF file content as bytes
    """
    # Create a BytesIO buffer to store PDF
    buffer = io.BytesIO()
    
    # Always use landscape for better readability
    pagesize = landscape(A4)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=pagesize,
        rightMargin=20,
        leftMargin=20,
        topMargin=40,
        bottomMargin=20
    )
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    # Add title
    title_para = Paragraph(title, title_style)
    elements.append(title_para)
    
    # Add metadata
    num_cols = len(df.columns)
    metadata_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Rows: {len(df):,} | Columns: {num_cols}"
    metadata_para = Paragraph(metadata_text, subtitle_style)
    elements.append(metadata_para)
    
    # SMART COLUMN SELECTION for wide datasets
    if num_cols > 12:
        # Too many columns - select the most important ones
        elements.append(Paragraph(
            f"<b>Note:</b> Dataset has {num_cols} columns. Showing most important columns. Download CSV for full data.",
            ParagraphStyle('Note', parent=styles['Normal'], fontSize=8, textColor=colors.red, alignment=TA_CENTER)
        ))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Select important columns intelligently
        important_cols = []
        
        # Priority 1: ID/Key columns (first few columns usually)
        for col in df.columns[:3]:
            important_cols.append(col)
        
        # Priority 2: Name columns
        name_cols = [c for c in df.columns if any(x in c.lower() for x in ['name', 'customer', 'defendant', 'client'])]
        for col in name_cols[:2]:
            if col not in important_cols:
                important_cols.append(col)
        
        # Priority 3: Contact columns
        contact_cols = [c for c in df.columns if any(x in c.lower() for x in ['email', 'phone', 'address', 'city', 'state'])]
        for col in contact_cols[:3]:
            if col not in important_cols and len(important_cols) < 10:
                important_cols.append(col)
        
        # Priority 4: Financial columns
        money_cols = [c for c in df.columns if any(x in c.lower() for x in ['balance', 'amount', 'payment', 'total', 'due', 'price'])]
        for col in money_cols[:2]:
            if col not in important_cols and len(important_cols) < 12:
                important_cols.append(col)
        
        # Use selected columns
        df = df[important_cols]
        num_cols = len(df.columns)
    
    # Prepare table data
    max_rows_per_page = 35  # Reduced for better readability
    total_rows = len(df)
    
    # Process in chunks
    for chunk_start in range(0, total_rows, max_rows_per_page):
        chunk_end = min(chunk_start + max_rows_per_page, total_rows)
        df_chunk = df.iloc[chunk_start:chunk_end]
        
        # Convert DataFrame to list of lists for reportlab
        data = [df_chunk.columns.tolist()]  # Header row
        
        # Add data rows (convert all to strings and truncate long values)
        for idx, row in df_chunk.iterrows():
            row_data = []
            for val in row:
                # Convert to string and truncate if too long
                str_val = str(val)
                if len(str_val) > 40:
                    str_val = str_val[:37] + "..."
                row_data.append(str_val)
            data.append(row_data)
        
        # Calculate column widths dynamically
        available_width = pagesize[0] - 40  # Account for margins
        
        # Smart column width allocation
        if num_cols <= 6:
            col_width = available_width / num_cols
        elif num_cols <= 10:
            col_width = available_width / num_cols * 0.95
        else:
            col_width = available_width / num_cols * 0.9
        
        # Minimum width check
        if col_width < 0.6 * inch:
            col_width = 0.6 * inch
        
        col_widths = [col_width] * num_cols
        
        # Create table
        table = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Style the table
        table_style = TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
            
            # Data rows styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#1f4788')),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ])
        
        table.setStyle(table_style)
        elements.append(table)
        
        # Add page break if there are more chunks
        if chunk_end < total_rows:
            elements.append(PageBreak())
            # Add continuation header
            continuation_text = f"<b>{title}</b> - Continued (Rows {chunk_end + 1} - {min(chunk_end + max_rows_per_page, total_rows)})"
            elements.append(Paragraph(continuation_text, subtitle_style))
            elements.append(Spacer(1, 0.1 * inch))
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def generate_summary_pdf(df: pd.DataFrame, title: str = "Data Summary", stats: dict = None) -> bytes:
    """
    Generate a summary PDF with statistics and preview
    
    Args:
        df: pandas DataFrame
        title: Report title
        stats: Optional dictionary of statistics to include
    
    Returns:
        bytes: PDF file content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f4788'),
        alignment=TA_CENTER
    )
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Statistics section
    if stats:
        elements.append(Paragraph("Summary Statistics", styles['Heading2']))
        elements.append(Spacer(1, 0.1 * inch))
        
        for key, value in stats.items():
            stat_text = f"<b>{key}:</b> {value}"
            elements.append(Paragraph(stat_text, styles['Normal']))
        
        elements.append(Spacer(1, 0.2 * inch))
    
    # Data preview (first 20 rows)
    elements.append(Paragraph("Data Preview (First 20 Rows)", styles['Heading2']))
    elements.append(Spacer(1, 0.1 * inch))
    
    preview_df = df.head(20)
    data = [preview_df.columns.tolist()]
    
    for idx, row in preview_df.iterrows():
        row_data = [str(val)[:30] for val in row]  # Truncate values
        data.append(row_data)
    
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    doc.build(elements)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
