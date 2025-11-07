from company_research.tools import *
print([name for name in globals() if name.endswith("_tool")])
