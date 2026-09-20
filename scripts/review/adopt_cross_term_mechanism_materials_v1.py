#!/usr/bin/env python3
"""Record accepted cross-term texts and definitions; tokenizer CPU only."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from diagnostics.cross_term_mechanism_materials_v1 import adopt
if __name__=='__main__':print(json.dumps(adopt(),ensure_ascii=False,indent=2))
