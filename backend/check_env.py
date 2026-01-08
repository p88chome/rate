import sys
import os

try:
    import langchain
    print(f"LangChain Version: {langchain.__version__}")
    print(f"LangChain Path: {langchain.__file__}")
except ImportError as e:
    print(f"LangChain Import Error: {e}")

try:
    import langchain_core
    print(f"LangChain Core Version: {langchain_core.__version__}")
except ImportError as e:
    print(f"LangChain Core Import Error: {e}")

print("\nInstalled Packages:")
import pkg_resources
for p in pkg_resources.working_set:
    if "langchain" in p.project_name:
        print(f"{p.project_name}=={p.version}")
