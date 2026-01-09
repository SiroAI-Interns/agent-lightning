# Copyright (c) Microsoft. All rights reserved.

"""Windows compatibility patch for Agent Lightning.

This module patches the fcntl import issue on Windows by creating a dummy fcntl module
before importing agentlightning.
"""

import sys
from unittest.mock import MagicMock

# Create a dummy fcntl module for Windows compatibility
if sys.platform == "win32":
    # Mock fcntl module since it doesn't exist on Windows
    fcntl_mock = MagicMock()
    sys.modules["fcntl"] = fcntl_mock



