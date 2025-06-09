import os
import json
from openai import OpenAI
import pandas as pd
import sys
from pathlib import Path
import base64
import mimetypes
import subprocess
import time

try:
    # client = OpenAI()
    client = OpenAI(api_key="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")
except openai.APIKeyError:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit()

import os
import json
from openai import OpenAI
import pandas as pd
import sys
from pathlib import Path
import base64
import mimetypes
import subprocess
import time

try:
    # client = OpenAI()
    client = OpenAI(api_key="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")
except openai.APIKeyError:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit()

