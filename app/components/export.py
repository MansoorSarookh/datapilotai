"""
Export module - Handles downloading charts as PNG, SVG, HTML, and GIF.
"""

import io
import base64
from typing import Optional
import plotly.graph_objects as go
import streamlit as st


def fig_to_bytes(fig: go.Figure, format: str = 'png', width: int = 1200, height: int = 700) -> bytes:
    """Convert Plotly figure to bytes."""
    return fig.to_image(format=format, width=width, height=height, scale=2)


def fig_to_html(fig: go.Figure) -> str:
    """Convert Plotly figure to HTML string."""
    return fig.to_html(include_plotlyjs='cdn', full_html=True)


def create_download_button(
    fig: go.Figure,
    filename: str,
    format: str,
    button_text: str,
    key: str
) -> None:
    """Create a Streamlit download button for chart export."""
    
    if format in ['png', 'svg', 'jpeg', 'webp']:
        try:
            img_bytes = fig_to_bytes(fig, format=format)
            mime_type = f'image/{format}'
            
            st.download_button(
                label=button_text,
                data=img_bytes,
                file_name=f"{filename}.{format}",
                mime=mime_type,
                key=key,
            )
        except Exception as e:
            st.error(f"Export failed: {e}. Make sure 'kaleido' is installed.")
    
    elif format == 'html':
        html_str = fig_to_html(fig)
        
        st.download_button(
            label=button_text,
            data=html_str,
            file_name=f"{filename}.html",
            mime='text/html',
            key=key,
        )


def create_export_panel(fig: go.Figure, chart_name: str, key_prefix: str = '') -> None:
    """Create a panel with multiple export options — PNG, SVG, JPEG, HTML, Hi-Res."""

    with st.expander("📥 Download Chart", expanded=False):
        cols = st.columns(5)

        with cols[0]:
            try:
                img_bytes = fig.to_image(format='png', width=1200, height=700, scale=2)
                st.download_button(
                    label='🖼️ PNG',
                    data=img_bytes,
                    file_name=f'{chart_name}.png',
                    mime='image/png',
                    key=f'{key_prefix}_png',
                    use_container_width=True,
                )
            except Exception:
                st.button('🖼️ PNG', disabled=True, key=f'{key_prefix}_png_d', use_container_width=True)

        with cols[1]:
            try:
                img_bytes = fig.to_image(format='jpeg', width=1200, height=700, scale=2)
                st.download_button(
                    label='📷 JPEG',
                    data=img_bytes,
                    file_name=f'{chart_name}.jpg',
                    mime='image/jpeg',
                    key=f'{key_prefix}_jpeg',
                    use_container_width=True,
                )
            except Exception:
                st.button('📷 JPEG', disabled=True, key=f'{key_prefix}_jpeg_d', use_container_width=True)

        with cols[2]:
            try:
                img_bytes = fig.to_image(format='svg', width=1200, height=700)
                st.download_button(
                    label='🎨 SVG',
                    data=img_bytes,
                    file_name=f'{chart_name}.svg',
                    mime='image/svg+xml',
                    key=f'{key_prefix}_svg',
                    use_container_width=True,
                )
            except Exception:
                st.button('🎨 SVG', disabled=True, key=f'{key_prefix}_svg_d', use_container_width=True)

        with cols[3]:
            html_str = fig.to_html(include_plotlyjs=False, full_html=True)
            st.download_button(
                label='🌐 HTML',
                data=html_str.encode('utf-8'),
                file_name=f'{chart_name}.html',
                mime='text/html',
                key=f'{key_prefix}_html',
                use_container_width=True,
            )

        with cols[4]:
            try:
                hires_bytes = fig.to_image(format='png', width=3000, height=1800, scale=3)
                st.download_button(
                    label='🔍 Hi-Res',
                    data=hires_bytes,
                    file_name=f'{chart_name}_4K.png',
                    mime='image/png',
                    key=f'{key_prefix}_hires',
                    use_container_width=True,
                )
            except Exception:
                st.button('🔍 Hi-Res', disabled=True, key=f'{key_prefix}_hires_d', use_container_width=True)


def export_dataframe(df, filename: str, format: str = 'csv') -> bytes:
    """Export DataFrame to bytes."""
    if format == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format == 'excel':
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        return buffer.getvalue()
    elif format == 'json':
        return df.to_json(orient='records', indent=2).encode('utf-8')
    else:
        return df.to_csv(index=False).encode('utf-8')


def create_data_export_panel(df, filename: str = 'data', key_prefix: str = '') -> None:
    """Create panel for exporting data."""
    
    st.markdown("##### 📊 Download Data")
    
    cols = st.columns(3)
    
    with cols[0]:
        csv_data = export_dataframe(df, filename, 'csv')
        st.download_button(
            label='📄 CSV',
            data=csv_data,
            file_name=f'{filename}.csv',
            mime='text/csv',
            key=f'{key_prefix}_csv'
        )
    
    with cols[1]:
        try:
            excel_data = export_dataframe(df, filename, 'excel')
            st.download_button(
                label='📊 Excel',
                data=excel_data,
                file_name=f'{filename}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key=f'{key_prefix}_excel'
            )
        except Exception:
            st.button('📊 Excel', disabled=True, key=f'{key_prefix}_excel_disabled')
    
    with cols[2]:
        json_data = export_dataframe(df, filename, 'json')
        st.download_button(
            label='📋 JSON',
            data=json_data,
            file_name=f'{filename}.json',
            mime='application/json',
            key=f'{key_prefix}_json'
        )


def create_animated_gif(frames: list, duration: float = 0.5) -> bytes:
    """Create animated GIF from list of image frames."""
    try:
        import imageio
        from PIL import Image
        
        images = []
        for frame_bytes in frames:
            img = Image.open(io.BytesIO(frame_bytes))
            images.append(img)
        
        buffer = io.BytesIO()
        imageio.mimsave(buffer, images, format='GIF', duration=duration, loop=0)
        return buffer.getvalue()
    
    except ImportError:
        raise ImportError("Please install imageio and Pillow for GIF export")


def capture_animation_frames(fig: go.Figure, num_frames: int = 10) -> list:
    """Capture frames from an animated Plotly figure."""
    frames = []
    
    if hasattr(fig, 'frames') and fig.frames:
        # Get frames from animated figure
        for i, frame in enumerate(fig.frames[:num_frames]):
            temp_fig = go.Figure(data=frame.data, layout=fig.layout)
            frame_bytes = temp_fig.to_image(format='png', width=800, height=500)
            frames.append(frame_bytes)
    else:
        # Single frame
        frame_bytes = fig.to_image(format='png', width=800, height=500)
        frames.append(frame_bytes)
    
    return frames

