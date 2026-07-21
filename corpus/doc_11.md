# Incident Response Runbook

## Severity Levels

SEV1: complete outage or confirmed customer data exposure. SEV2: major feature unavailable or severe degradation affecting many customers. SEV3: minor feature degraded, workaround available. SEV4: cosmetic or low-impact issue with no customer impact.

## Acknowledgement Targets

SEV1 must be acknowledged within 15 minutes of the page. SEV2 within 30 minutes. SEV3 and SEV4 are handled during business hours and do not page.

## Roles

Every SEV1 and SEV2 has a named Incident Commander. The Incident Commander does not debug. Their role is coordination, communication, and decision-making. A separate Communications Lead is named for SEV1 only and owns the status page and customer messaging.

The engineer who acknowledges the page is Incident Commander by default until they explicitly hand the role to someone else.

## Status Page

The status page is updated within 30 minutes of a SEV1 declaration and every 60 minutes thereafter until resolution.

## Postmortem

Every SEV1 and SEV2 requires a written postmortem published within five business days of resolution. Postmortems are blameless. They must include a timeline, contributing factors, and action items with named owners and due dates.
