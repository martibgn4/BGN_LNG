import pandas as pd
from datetime import datetime

# Replace with your actual DataFrame
def run_lng_on_water_processing(df):
    # Reshape with MultiIndex columns
    df_display = df.copy()
    df_display.columns = pd.MultiIndex.from_tuples(
        [('>0 days',  'Cargoes'), ('>0 days',  'Milion MT'),
         ('>20 days', 'Cargoes'), ('>20 days', 'Milion MT'),
         ('>30 days', 'Cargoes'), ('>30 days', 'Milion MT')],
        names=[f'LNG on Water by Country <br> on {datetime.today().date().strftime('%d-%b-%Y')}', 'Metric']
    )
    df_display = df_display.sort_values(('>0 days', 'Milion MT'), ascending=False)
    df_display.index.name = None

    # Totals row
    totals = df_display.sum(numeric_only=True).to_frame().T
    totals.index = ['Total']
    df_display_with_totals = df_display  #pd.concat([df_display, totals])

    fmt = {
        ('>0 days', 'Cargoes'): '{:,.0f}',
        ('>0 days', 'Milion MT'): '{:,.2f}',
        ('>20 days', 'Cargoes'): '{:,.0f}',
        ('>20 days', 'Milion MT'): '{:,.2f}',
        ('>30 days', 'Cargoes'): '{:,.0f}',
        ('>30 days', 'Milion MT'): '{:,.2f}',
    }

    # Styled table
    styled = (
        df_display_with_totals.style
        .format(fmt)
        .set_caption(f'LNG on Water by Country on {datetime.today().date()}')
        .background_gradient(
            cmap='Blues',
            # subset=pd.IndexSlice[df_display.index, :],  # exclude totals from gradient
            axis=0,
        )
        .set_table_attributes('id="lng-table" class="sortable"')
        .set_table_styles([
            {'selector': 'caption',
             'props': 'caption-side: top; font-size: 1.3em; font-weight: 600; '
                      'padding: 10px 0; color: #111827; text-align: left;'},
            {'selector': 'th, td',
             'props': 'border: 1px solid #e5e7eb; padding: 8px 14px;'},
            {'selector': 'thead tr:nth-child(1) th',
             'props': 'background-color: #1e3a8a; color: white; text-align: center; '
                      'font-weight: 600;'},
            {'selector': 'thead tr:nth-child(2) th',
             'props': 'background-color: #3b82f6; color: white; text-align: center; '
                      'font-weight: 500; font-size: 0.9em; cursor: pointer; user-select: none;'},
            # {'selector': 'thead tr:nth-child(2) th:hover',
            #  'props': 'background-color: #2563eb;'},
            {'selector': 'tbody th',
             'props': 'background-color: #f9fafb; font-weight: 600; text-align: left;'},
            {'selector': 'tbody td',
             'props': 'text-align: right; font-variant-numeric: tabular-nums;'},
            {'selector': 'tbody tr:hover td',
             'props': 'background-color: #fef3c7 !important;'},
            {'selector': 'tbody tr.totals-row th, tbody tr.totals-row td',
             'props': 'background-color: #1f2937 !important; color: white; '
                      'font-weight: 700; border-top: 2px solid #1f2937;'},
            {'selector': '',
             'props': 'border-collapse: collapse; margin: 12px 0; '
                      'box-shadow: 0 1px 3px rgba(0,0,0,0.08); '
                      'font-family: -apple-system, "Segoe UI", Roboto, sans-serif;'},
        ])
    )

    # Build a full standalone HTML page with search + sort
    table_html = styled.to_html()

    # Tag the totals row so JS can skip it during sorting
    # table_html = table_html.replace(
    #     '<tr>\n      <th id="T_', '<tr class="totals-marker"><th id="T_', 1  # placeholder, replaced below
    # )

    # Simpler: post-process to add a class to the last <tr>
    import re
    def mark_totals_row(html):
        # Add class="totals-row" to the last <tr> in <tbody>
        parts = html.rsplit('<tr>', 1)
        if len(parts) == 2:
            return parts[0] + '<tr class="totals-row">' + parts[1]
        return html
    # table_html = mark_totals_row(styled.to_html())

    full_html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>LNG on Water</title>
    <style>
      body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
             max-width: 1100px; margin: 30px auto; padding: 0 20px; color: #111827; }}
      #search {{ padding: 8px 12px; font-size: 1em; width: 280px;
                 border: 1px solid #d1d5db; border-radius: 6px; margin-bottom: 8px; }}
      .hint {{ color: #6b7280; font-size: 0.9em; margin-bottom: 4px; }}
    </style>
    </head>
    <body>
      <h1>LNG on Water by Country</h1>
      <input id="search" type="text" placeholder="Filter countries..." />
      <p class="hint">Click any column header (Vessels / Tonnes) to sort.</p>
      {table_html}
    
    <script>
    (function() {{
      const table = document.getElementById('lng-table');
      if (!table) return;
    
      // --- Sorting ---
      const subHeaders = table.querySelectorAll('thead tr:nth-child(2) th');
      let sortState = {{ col: null, asc: false }};
    
      subHeaders.forEach((th, idx) => {{
        th.addEventListener('click', () => {{
          const tbody = table.querySelector('tbody');
          const rows = Array.from(tbody.querySelectorAll('tr')).filter(r => !r.classList.contains('totals-row'));
          const totalsRow = tbody.querySelector('tr.totals-row');
    
          const asc = sortState.col === idx ? !sortState.asc : false;
          sortState = {{ col: idx, asc }};
    
          rows.sort((a, b) => {{
            const av = parseFloat(a.cells[idx].innerText.replace(/,/g, '')) || 0;
            const bv = parseFloat(b.cells[idx].innerText.replace(/,/g, '')) || 0;
            return asc ? av - bv : bv - av;
          }});
    
          rows.forEach(r => tbody.appendChild(r));
          if (totalsRow) tbody.appendChild(totalsRow);
    
          subHeaders.forEach(h => h.innerText = h.innerText.replace(/ [▲▼]$/, ''));
          th.innerText += asc ? ' ▲' : ' ▼';
        }});
      }});
    
      // --- Search filter ---
      const search = document.getElementById('search');
      search.addEventListener('input', () => {{
        const q = search.value.toLowerCase();
        table.querySelectorAll('tbody tr').forEach(row => {{
          if (row.classList.contains('totals-row')) return;
          const country = row.querySelector('th')?.innerText.toLowerCase() ?? '';
          row.style.display = country.includes(q) ? '' : 'none';
        }});
      }});
    }})();
    </script>
    </body>
    </html>
    """

    with open('C:\\Marti\\lng_on_water.html', 'w', encoding='utf-8') as f:
        f.write(full_html)

    print("Saved to lng_on_water.html")
    return table_html

if __name__ == '__main__':
    _ = run_lng_on_water_processing()