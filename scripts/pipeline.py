from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=".root", pythonpath=True)

from scripts.train import main


if __name__ == "__main__":
    main()
