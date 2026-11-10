"""Launch real-PCAP record ordering audits without changing captures.
Gọi CLI audit ordering record PCAP thật, không sửa capture.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nids_ml.cli.pcap_ordering import main

if __name__=='__main__':raise SystemExit(main())
