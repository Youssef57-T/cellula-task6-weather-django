from datetime import datetime

from django import template

register = template.Library()


@register.filter
def isotime(value, fmt="%d %b %Y, %H:%M"):
    """Format an ISO timestamp string from the forecast dict, e.g. '2025-12-25T06:00:00' -> '25 Dec 2025, 06:00'."""
    return datetime.fromisoformat(value).strftime(fmt) if value else ""
