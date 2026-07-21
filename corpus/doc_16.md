# Deployment and Release Process

## Branching

Meridian uses trunk-based development. Work is done on short-lived branches and merged into main. Branches living longer than three days are flagged in the weekly engineering review.

All merges to main require one approving review and a green continuous integration run.

## Deploy Windows

Production deploys run on Tuesdays and Thursdays between 9:00 AM and 3:00 PM Mountain Time. Deploys outside the window require approval from the group's engineering manager.

No production deploys are made during the winter closure. The freeze begins at 3:00 PM Mountain Time on the last business day before the closure and lifts when offices reopen.

## Canary

Every production deploy goes to a canary receiving 5 percent of traffic for 30 minutes. Error rate and latency are watched automatically during the canary window. A canary that trips its thresholds rolls back without human involvement.

## Rollback

Rollback must complete within 5 minutes. Any change that cannot be rolled back in 5 minutes must be behind a feature flag.

## Feature Flags

Feature flags are removed within 30 days of a feature reaching 100 percent of traffic. Stale flags are reported to the owning team weekly.
