from email.utils import parseaddr
from importlib.metadata import metadata, version

__version__ = version("pydiffuser")
__author__ = parseaddr(metadata("pydiffuser")["Author-email"])[0]
