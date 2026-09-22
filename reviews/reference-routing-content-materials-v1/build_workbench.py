"""Offline view reusing the existing WP3/paired workbench CSS and ReviewCore."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def build(folder):
    html=(HERE/'workbench.html.in').read_text()
    replacements={
      '__SHARED_BASE_CSS__':(ROOT/'tools/wp3_candidate_review_ui/styles.css').read_text(),
      '__SHARED_PAIRED_CSS__':(ROOT/'tools/general_model_paired_review_ui/styles.css').read_text(),
      '__SHARED_CORE_JS__':(ROOT/'tools/wp3_candidate_review_ui/core.js').read_text(),
      '__REVIEW_DATA__':(folder/'review-data.json').read_text().replace('<','\\u003c'),
    }
    for key,value in replacements.items():html=html.replace(key,value)
    with (folder/'REVIEW.html').open('x',encoding='utf-8') as f:f.write(html)

if __name__=='__main__':build(HERE/sys.argv[1])
