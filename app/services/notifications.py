from __future__ import annotations
import asyncio
import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Any, Mapping

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from app.config import get_settings


logger = logging.getLogger("app.notifications")

settings = get_settings()


async def send_slack_alert(alert: Mapping[str, Any]) -> None:
    if httpx is None:
        return
    webhook = os.getenv("SMARTFUEL_SLACK_WEBHOOK")
    if not webhook:
        return
    payload = {
        "text": (
            f"⚠️ SmartFuel Alert\n"
            f"Sensor: {alert.get('sensor_no')}\n"
            f"Probability: {alert.get('fault_probability')}\n"
            f"Severity: {alert.get('severity')}\n"
            f"Date: {alert.get('date')}"
        )
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(webhook, json=payload)
            resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Slack notification failed: %s", exc)


def send_email_alert(alert: Mapping[str, Any]) -> None:
    recipients = [r.strip() for r in settings.smtp_recipients.split(",") if r.strip()]
    if not (settings.smtp_host and recipients):
        return

    msg = EmailMessage()
    msg["Subject"] = f"SmartFuel Critical Alert – Sensor {alert.get('sensor_no')}"
    msg["From"] = settings.smtp_sender
    msg["To"] = ", ".join(recipients)
    body = (
        "Critical alert detected:\n\n"
        f"Sensor: {alert.get('sensor_no')}\n"
        f"Probability: {alert.get('fault_probability')}\n"
        f"Severity: {alert.get('severity')}\n"
        f"Date: {alert.get('date')}\n"
        f"Source: {alert.get('source_file')}\n"
    )
    msg.set_content(body)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
            if settings.smtp_username and settings.smtp_password:
                server.starttls()
                server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(msg)
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Email notification failed: %s", exc)


async def notify_critical(alerts: list[Mapping[str, Any]]) -> None:
    tasks = [send_slack_alert(alert) for alert in alerts if alert.get("severity") == "critical"]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    for alert in alerts:
        if alert.get("severity") == "critical":
            send_email_alert(alert)
